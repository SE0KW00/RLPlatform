"""2단계: PPO — InstructGPT / TRL 스타일의 RLHF.

REINFORCE 의 문제:
  (a) 샘플 하나로 업데이트 한 번 → 샘플 효율이 나쁘다
  (b) 시퀀스 전체에 같은 advantage → 어떤 토큰이 좋았는지 구분하지 못한다(credit assignment)
  (c) 업데이트 크기 제약이 없어 정책이 한 번에 크게 망가질 수 있다

PPO 의 해법:
  (a) 같은 롤아웃으로 여러 epoch 업데이트 + importance ratio ρ = π_θ / π_old
  (b) critic V(s_t) 와 GAE 로 **토큰별** advantage 를 추정
  (c) ρ 를 [1-ε, 1+ε] 로 clip 해 한 번에 너무 멀리 가지 못하게 함

LLM RLHF 의 토큰별 보상 (중요!):
    r_t = -β · (log π_old(a_t|s_t) - log π_ref(a_t|s_t))         (모든 응답 토큰)
    r_T += R(x, y)                                                (마지막 토큰에만 점수)
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..generation import generate, masked_mean, masked_whiten, token_logprobs
from ..model import TinyGPT, last_token_index
from ..tasks import Task
from ..tokenizer import CharTokenizer
from ..utils import Logger
from .common import RewardFn, RLConfig, evaluate, make_reference, task_reward_fn


@dataclass
class PPOConfig(RLConfig):
    ppo_epochs: int = 4
    minibatch_size: int = 32
    clip_range: float = 0.2
    clip_range_value: float = 0.2
    vf_coef: float = 0.5
    gamma: float = 1.0
    lam: float = 0.95
    whiten_advantages: bool = True


def compute_gae(rewards, values, mask, gamma: float, lam: float):
    """마스크된 토큰 열에 대한 Generalized Advantage Estimation.

    δ_t = r_t + γ V(s_{t+1}) - V(s_t)
    A_t = δ_t + γλ A_{t+1}
    응답이 끝난 뒤(mask=0)의 가치는 0 으로 취급한다(에피소드 종료).
    """
    B, T = rewards.shape
    adv = torch.zeros_like(rewards)
    last = torch.zeros(B)
    for t in reversed(range(T)):
        nmask = mask[:, t + 1] if t + 1 < T else torch.zeros(B)
        next_v = values[:, t + 1] * nmask if t + 1 < T else torch.zeros(B)
        delta = rewards[:, t] + gamma * next_v - values[:, t]
        last = delta + gamma * lam * nmask * last
        adv[:, t] = last
    adv = adv * mask
    returns = adv + values
    return adv, returns


def train_ppo(
    policy: TinyGPT,
    tok: CharTokenizer,
    task: Task,
    cfg: PPOConfig,
    rng: random.Random,
    reward_fn: RewardFn | None = None,
    logger: Logger | None = None,
) -> TinyGPT:
    assert policy.value_head is not None, "PPO 는 value head 가 있는 정책이 필요합니다 (with_value_head=True)"
    reward_fn = reward_fn or task_reward_fn(task)
    ref = make_reference(policy)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    logger = logger or Logger()

    for step in range(1, cfg.steps + 1):
        # ================= 1) 롤아웃 수집 (gradient 없음) =================
        prompts = task.sample_prompts(cfg.batch_size, rng)
        roll = generate(policy, tok, prompts, task.max_new_tokens, temperature=cfg.temperature)
        score = reward_fn(roll)
        amask = roll.action_mask
        ids, attn = roll.input_ids, roll.attention_mask

        with torch.no_grad():
            policy.eval()
            old_logp, old_values = token_logprobs(policy, ids, attn, return_values=True)
            ref_logp = token_logprobs(ref, ids, attn)

            # ---- 토큰별 보상: KL 벌점 + 마지막 토큰에 점수 ----
            kl = (old_logp - ref_logp) * amask
            rewards = -cfg.kl_coef * kl
            last = last_token_index(amask)
            rewards[torch.arange(len(last)), last] += score

            old_values = old_values * amask
            adv, returns = compute_gae(rewards, old_values, amask, cfg.gamma, cfg.lam)
            if cfg.whiten_advantages:
                adv = masked_whiten(adv, amask)

        # ================= 2) 여러 epoch 미니배치 업데이트 =================
        policy.train()
        stats = {"pg_loss": 0.0, "vf_loss": 0.0, "clipfrac": 0.0, "approx_kl": 0.0}
        n_updates = 0
        B = ids.shape[0]
        for _ in range(cfg.ppo_epochs):
            perm = torch.randperm(B)
            for i in range(0, B, cfg.minibatch_size):
                mb = perm[i : i + cfg.minibatch_size]
                m = amask[mb]
                logp, values = token_logprobs(policy, ids[mb], attn[mb], return_values=True)

                # --- clipped surrogate objective ---
                ratio = torch.exp(logp - old_logp[mb])
                pg1 = -adv[mb] * ratio
                pg2 = -adv[mb] * torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range)
                pg_loss = masked_mean(torch.max(pg1, pg2), m)

                # --- clipped value loss ---
                v_clip = old_values[mb] + torch.clamp(
                    values - old_values[mb], -cfg.clip_range_value, cfg.clip_range_value
                )
                vf_loss = 0.5 * masked_mean(
                    torch.max((values - returns[mb]) ** 2, (v_clip - returns[mb]) ** 2), m
                )

                loss = pg_loss + cfg.vf_coef * vf_loss
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                opt.step()

                with torch.no_grad():
                    stats["pg_loss"] += pg_loss.item()
                    stats["vf_loss"] += vf_loss.item()
                    stats["clipfrac"] += masked_mean(((ratio - 1).abs() > cfg.clip_range).float(), m).item()
                    stats["approx_kl"] += masked_mean(0.5 * (logp - old_logp[mb]) ** 2, m).item()
                n_updates += 1

        if step % cfg.log_every == 0 or step == cfg.steps:
            ev, samples = evaluate(policy, tok, task, rng, cfg.eval_size)
            stats = {k: v / n_updates for k, v in stats.items()}
            logger.log(step, {"score": score.mean(), "kl": kl.sum(-1).mean(), **stats, **ev}, samples)
    return policy
