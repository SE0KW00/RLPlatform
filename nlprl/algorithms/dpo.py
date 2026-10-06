"""3단계: DPO (Direct Preference Optimization).

RLHF(보상 모델 + PPO)는 모델 4개(정책, 참조, 보상, critic)와 온라인 샘플링이 필요하다.
DPO 는 KL 제약 보상 최대화 문제의 최적해가

    π*(y|x) = π_ref(y|x) · exp(r(x,y)/β) / Z(x)    ⇔    r(x,y) = β log π*(y|x)/π_ref(y|x) + β log Z(x)

라는 사실을 이용해, 이 '암묵적 보상'을 Bradley-Terry 식에 바로 넣는다. Z(x) 는 소거된다.

    L_DPO = -log σ( β [ (log π_θ(y_w|x) - log π_ref(y_w|x)) - (log π_θ(y_l|x) - log π_ref(y_l|x)) ] )

→ 보상 모델도, 샘플링도, critic 도 없는 '지도학습 같은' 오프라인 알고리즘.
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from ..generation import token_logprobs
from ..model import TinyGPT
from ..preference import PreferencePair, encode_pairs
from ..tasks import Task
from ..tokenizer import CharTokenizer
from ..utils import Logger
from .common import evaluate, make_reference


@dataclass
class DPOConfig:
    epochs: int = 1
    batch_size: int = 64
    lr: float = 2e-5
    beta: float = 0.1
    log_every: int = 20
    eval_size: int = 256


def sequence_logprob(model, ids, mask, rmask):
    """log π(y|x) = 응답 토큰들의 log-prob 합."""
    lp = token_logprobs(model, ids, mask)
    return (lp * rmask[:, 1:].float()).sum(-1)


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta: float):
    logits = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    loss = -F.logsigmoid(logits).mean()
    chosen_reward = beta * (pi_c - ref_c).detach()
    rejected_reward = beta * (pi_r - ref_r).detach()
    return loss, chosen_reward, rejected_reward


def train_dpo(
    policy: TinyGPT,
    tok: CharTokenizer,
    task: Task,
    pairs: list[PreferencePair],
    cfg: DPOConfig,
    rng: random.Random,
    logger: Logger | None = None,
) -> TinyGPT:
    ref = make_reference(policy)
    opt = torch.optim.Adam(policy.parameters(), lr=cfg.lr)
    logger = logger or Logger()
    pairs = pairs[:]
    step = 0

    for _ in range(cfg.epochs):
        rng.shuffle(pairs)
        for i in range(0, len(pairs), cfg.batch_size):
            batch = pairs[i : i + cfg.batch_size]
            n = len(batch)
            ids, mask, rmask = encode_pairs(tok, batch)

            policy.train()
            pi = sequence_logprob(policy, ids, mask, rmask)
            with torch.no_grad():
                rf = sequence_logprob(ref, ids, mask, rmask)
            loss, rc, rr = dpo_loss(pi[:n], pi[n:], rf[:n], rf[n:], cfg.beta)

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
            opt.step()
            step += 1

            if step % cfg.log_every == 0:
                ev, samples = evaluate(policy, tok, task, rng, cfg.eval_size)
                logger.log(step, {
                    "loss": loss.item(),
                    "reward_acc": (rc > rr).float().mean(),  # 암묵적 보상으로 쌍을 맞히는 비율
                    "margin": (rc - rr).mean(),
                    "chosen_logp": pi[:n].mean().item(),
                    "rejected_logp": pi[n:].mean().item(),
                    **ev,
                }, samples)
    ev, samples = evaluate(policy, tok, task, rng, cfg.eval_size)
    logger.log(step, {"loss": loss.item(), **ev}, samples)
    return policy
