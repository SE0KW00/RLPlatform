"""여러 RL 알고리즘이 공유하는 설정과 보조 함수."""

from __future__ import annotations

import copy
import random
from dataclasses import dataclass
from typing import Callable

import torch

from ..generation import Rollout, generate
from ..model import RewardModel, TinyGPT
from ..tasks import Task
from ..tokenizer import CharTokenizer

# (rollout) -> 보상 텐서 [B]
RewardFn = Callable[[Rollout], torch.Tensor]


@dataclass
class RLConfig:
    steps: int = 200
    batch_size: int = 64
    lr: float = 3e-4
    kl_coef: float = 0.05  # β : 참조 정책에서 멀어지는 것에 대한 벌점 계수
    temperature: float = 1.0
    log_every: int = 20
    eval_size: int = 256


def task_reward_fn(task: Task) -> RewardFn:
    """규칙 기반(또는 정답 검증) 보상."""

    def fn(roll: Rollout) -> torch.Tensor:
        return torch.tensor(task.rewards(roll.prompts, roll.responses), dtype=torch.float32)

    return fn


def reward_model_fn(rm: RewardModel) -> RewardFn:
    """학습된 보상 모델 r_φ(x, y) 의 점수를 보상으로 사용 (RLHF)."""
    rm.eval()

    @torch.no_grad()
    def fn(roll: Rollout) -> torch.Tensor:
        return rm(roll.input_ids, roll.attention_mask)

    return fn


def make_reference(policy: TinyGPT) -> TinyGPT:
    """학습 시작 시점의 정책을 얼려(freeze) 참조 정책 π_ref 로 사용한다."""
    ref = copy.deepcopy(policy)
    ref.value_head = None
    ref.eval()
    for p in ref.parameters():
        p.requires_grad_(False)
    return ref


@torch.no_grad()
def evaluate(policy: TinyGPT, tok: CharTokenizer, task: Task, rng: random.Random, n: int = 256,
             temperature: float = 1.0) -> tuple[dict, list[str]]:
    prompts = task.sample_prompts(n, rng)
    roll = generate(policy, tok, prompts, task.max_new_tokens, temperature=temperature)
    samples = [p + r for p, r in zip(prompts[:3], roll.responses[:3])]
    return {f"eval/{k}": v for k, v in task.metrics(prompts, roll.responses).items()}, samples
