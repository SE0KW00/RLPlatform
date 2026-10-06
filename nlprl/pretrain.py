"""0단계: 사전학습 / SFT.

RL 은 '아무것도 모르는' 모델에서 시작하지 않는다. 먼저 다음 토큰 예측(교차 엔트로피)으로
과제의 언어를 배운 모델(= SFT 모델)을 만들고, 이것이
  - RL 의 초기 정책 π_θ  이자
  - KL 제약의 기준이 되는 참조 정책 π_ref
가 된다.
"""

from __future__ import annotations

import random

import torch
import torch.nn.functional as F

from .generation import generate, right_pad
from .model import GPTConfig, TinyGPT
from .tasks import Task
from .tokenizer import CharTokenizer
from .utils import Logger


def lm_loss(model: TinyGPT, ids: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    logits = model(ids, mask).logits[:, :-1]
    tgt = ids[:, 1:]
    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), tgt.reshape(-1), reduction="none")
    m = mask[:, 1:].reshape(-1).float()
    return (loss * m).sum() / m.sum()


def pretrain(
    task: Task,
    tok: CharTokenizer,
    rng: random.Random,
    steps: int = 1500,
    batch_size: int = 64,
    lr: float = 1e-3,
    cfg: GPTConfig | None = None,
    log_every: int = 250,
    logger: Logger | None = None,
) -> TinyGPT:
    cfg = cfg or GPTConfig(vocab_size=tok.vocab_size)
    model = TinyGPT(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    logger = logger or Logger()

    for step in range(1, steps + 1):
        texts = task.corpus(batch_size, rng)
        ids, mask = right_pad([tok.encode(t, add_bos=True, add_eos=True) for t in texts], tok.pad_id)
        loss = lm_loss(model, ids, mask)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()

        if step % log_every == 0 or step == steps:
            prompts = task.sample_prompts(256, rng)
            roll = generate(model, tok, prompts, task.max_new_tokens)
            m = task.metrics(prompts, roll.responses)
            samples = [p + r for p, r in zip(prompts[:3], roll.responses[:3])]
            logger.log(step, {"lm_loss": loss.item(), **m}, samples)
    return model
