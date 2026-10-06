"""4단계: GRPO (Group Relative Policy Optimization) — DeepSeekMath / DeepSeek-R1.

PPO 의 critic(value 모델)은 정책만큼 큰 모델을 하나 더 학습해야 해서 비싸다.
GRPO 는 critic 을 없애고, **같은 프롬프트에서 G 개의 응답을 뽑아 그룹 안에서 비교**한다.

    A_i = (R_i - mean(R_1..R_G)) / (std(R_1..R_G) + ε)     (응답 i 의 모든 토큰에 같은 값)

손실 (토큰 평균):
    L = -1/G Σ_i 1/|y_i| Σ_t [ min(ρ_t A_i, clip(ρ_t, 1-ε, 1+ε) A_i) - β · KL_t ]
    KL_t = π_ref/π_θ - log(π_ref/π_θ) - 1      (k3 추정치: 항상 ≥ 0, 분산이 작음)

PPO 와의 차이: KL 을 보상에 섞지 않고 손실에 직접 더한다.
정답/오답처럼 **검증 가능한 보상(RLVR)** 과 특히 잘 맞는다.
그룹 전체가 맞거나 전체가 틀리면 A=0 → 학습 신호가 없다는 점에 주목하자.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch

from ..generation import generate, masked_mean, token_logprobs
from ..model import TinyGPT
from ..tasks import Task
from ..tokenizer import CharTokenizer
from ..utils import Logger
from .common import RewardFn, RLConfig, evaluate, make_reference, task_reward_fn


@dataclass
class GRPOConfig(RLConfig):
    group_size: int = 8  # G
    num_iterations: int = 1  # μ: 같은 롤아웃으로 몇 번 업데이트할지 (1 이면 ρ=1, on-policy)
    clip_range: float = 0.2
    kl_coef: float = 0.04
    scale_rewards: bool = True  # False 면 std 로 나누지 않음 (Dr. GRPO 제안)
    loss_type: str = "grpo"  # "grpo": 시퀀스별 토큰 평균 | "token": 배치 전체 토큰 평균 (DAPO)


def group_advantages(rewards: torch.Tensor, group_size: int, scale: bool = True, eps: float = 1e-4):
    g = rewards.view(-1, group_size)
    adv = g - g.mean(dim=1, keepdim=True)
    if scale:
        adv = adv / (g.std(dim=1, keepdim=True) + eps)
    return adv.view(-1)


def train_grpo(
    policy: TinyGPT,
    tok: CharTokenizer,
    task: Task,
    cfg: GRPOConfig,
    rng: random.Random,
    reward_fn: RewardFn | None = None,
    logger: Logger | None = None,
) -> TinyGPT:
    assert cfg.batch_size % cfg.group_size == 0, "batch_size 는 group_size 의 배수여야 합니다"
    reward_fn = reward_fn or task_reward_fn(task)
    ref = make_reference(policy)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    logger = logger or Logger()

    for step in range(1, cfg.steps + 1):
        # ---- 1) 프롬프트마다 G 개 응답 ----------------------------------------
        uniq = task.sample_prompts(cfg.batch_size // cfg.group_size, rng)
        prompts = [p for p in uniq for _ in range(cfg.group_size)]
        roll = generate(policy, tok, prompts, task.max_new_tokens, temperature=cfg.temperature)
        score = reward_fn(roll)
        amask = roll.action_mask
        ids, attn = roll.input_ids, roll.attention_mask

        # ---- 2) 그룹 상대 advantage (critic 없음!) ----------------------------
        adv = group_advantages(score, cfg.group_size, cfg.scale_rewards)[:, None]  # [B,1]
        with torch.no_grad():
            policy.eval()
            old_logp = token_logprobs(policy, ids, attn)
            ref_logp = token_logprobs(ref, ids, attn)

        # ---- 3) 업데이트 -------------------------------------------------------
        policy.train()
        for _ in range(cfg.num_iterations):
            logp = token_logprobs(policy, ids, attn)
            ratio = torch.exp(logp - old_logp)
            surr = torch.min(ratio * adv, torch.clamp(ratio, 1 - cfg.clip_range, 1 + cfg.clip_range) * adv)
            log_r = ref_logp - logp
            kl = torch.exp(log_r) - log_r - 1  # k3
            per_token = -(surr - cfg.kl_coef * kl)
            if cfg.loss_type == "grpo":
                loss = masked_mean(per_token, amask, dim=1).mean()
            else:
                loss = masked_mean(per_token, amask)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps:
            ev, samples = evaluate(policy, tok, task, rng, cfg.eval_size)
            g = score.view(-1, cfg.group_size)
            zero_signal = (g.std(dim=1) == 0).float().mean()  # 학습 신호가 없는 그룹 비율
            logger.log(
                step,
                {"score": score.mean(), "kl": masked_mean(kl, amask).item(), "loss": loss.item(),
                 "zero_adv_groups": zero_signal, **ev},
                samples,
            )
    return policy
