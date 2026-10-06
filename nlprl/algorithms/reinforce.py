"""1단계: REINFORCE (vanilla policy gradient) + baseline + KL 벌점.

목표:  max_θ  E_{x~D, y~π_θ(·|x)} [ R(x, y) ]  -  β · KL(π_θ || π_ref)

정책 경사 정리(policy gradient theorem):
    ∇_θ J = E[ (R - b) · ∇_θ log π_θ(y|x) ],   log π_θ(y|x) = Σ_t log π_θ(y_t | x, y_<t)

  - R - b : advantage. baseline b 를 빼도 기댓값은 그대로이지만 분산이 크게 줄어든다.
  - KL 은 보상에 섞어서 R' = R - β · Σ_t (log π_θ - log π_ref) 로 처리한다 (InstructGPT 방식).

코드 한 줄 요약:  loss = -(advantage * Σ_t log π_θ(y_t)).mean()
"""

from __future__ import annotations

import random

import torch

from ..generation import generate, token_logprobs
from ..model import TinyGPT
from ..tasks import Task
from ..tokenizer import CharTokenizer
from ..utils import Logger
from .common import RewardFn, RLConfig, evaluate, make_reference, task_reward_fn


def train_reinforce(
    policy: TinyGPT,
    tok: CharTokenizer,
    task: Task,
    cfg: RLConfig,
    rng: random.Random,
    reward_fn: RewardFn | None = None,
    baseline: str = "batch_mean",  # "none" | "batch_mean" | "ema"
    logger: Logger | None = None,
) -> TinyGPT:
    reward_fn = reward_fn or task_reward_fn(task)
    ref = make_reference(policy)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    logger = logger or Logger()
    ema_b = 0.0

    for step in range(1, cfg.steps + 1):
        # ---- 1) 롤아웃 -------------------------------------------------------
        prompts = task.sample_prompts(cfg.batch_size, rng)
        roll = generate(policy, tok, prompts, task.max_new_tokens, temperature=cfg.temperature)
        score = reward_fn(roll)  # [B]
        amask = roll.action_mask  # [B, L-1]

        # ---- 2) log π_θ (gradient 필요) 와 log π_ref ---------------------------
        policy.train()
        logp = token_logprobs(policy, roll.input_ids, roll.attention_mask)
        with torch.no_grad():
            ref_logp = token_logprobs(ref, roll.input_ids, roll.attention_mask)
            kl = ((logp - ref_logp) * amask).sum(-1)  # 시퀀스 단위 KL 추정치 (k1)
            total_reward = score - cfg.kl_coef * kl

            # ---- 3) baseline 과 advantage ------------------------------------
            if baseline == "batch_mean":
                b = total_reward.mean()
            elif baseline == "ema":
                ema_b = 0.9 * ema_b + 0.1 * total_reward.mean().item()
                b = ema_b
            else:
                b = 0.0
            adv = total_reward - b

        # ---- 4) 정책 경사 --------------------------------------------------
        seq_logp = (logp * amask).sum(-1)  # log π_θ(y|x)
        loss = -(adv * seq_logp).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
        opt.step()

        if step % cfg.log_every == 0 or step == cfg.steps:
            ev, samples = evaluate(policy, tok, task, rng, cfg.eval_size)
            logger.log(step, {"score": score.mean(), "kl": kl.mean(), "loss": loss.item(), **ev}, samples)
    return policy
