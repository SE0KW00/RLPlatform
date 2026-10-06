"""선호(preference) 데이터 만들기 + 보상 모델 학습.

실제 RLHF 에서는 사람이 같은 프롬프트에 대한 두 응답 중 더 나은 것을 고른다.
여기서는 SFT 정책에서 응답을 여러 개 샘플링한 뒤, 과제의 규칙 보상으로 '사람 대신' 라벨링한다.
(label_noise 를 주면 사람 라벨러의 실수를 흉내낼 수 있다)

Bradley-Terry 모델:
    P(y_w ≻ y_l | x) = σ( r(x, y_w) - r(x, y_l) )
    보상 모델 손실   = -log σ( r_φ(x, y_w) - r_φ(x, y_l) )
"""

from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .generation import generate, right_pad
from .model import RewardModel, TinyGPT
from .tasks import Task
from .tokenizer import CharTokenizer
from .utils import Logger


@dataclass
class PreferencePair:
    prompt: str
    chosen: str
    rejected: str


@torch.no_grad()
def make_preference_pairs(
    policy: TinyGPT,
    tok: CharTokenizer,
    task: Task,
    rng: random.Random,
    n_pairs: int = 2000,
    samples_per_prompt: int = 4,
    label_noise: float = 0.0,
) -> list[PreferencePair]:
    pairs: list[PreferencePair] = []
    max_rounds = max(50, 4 * n_pairs // 64)
    for _ in range(max_rounds):
        if len(pairs) >= n_pairs:
            break
        uniq = task.sample_prompts(64, rng)
        prompts = [p for p in uniq for _ in range(samples_per_prompt)]
        roll = generate(policy, tok, prompts, task.max_new_tokens)
        rewards = task.rewards(prompts, roll.responses)
        for g in range(len(uniq)):
            idx = list(range(g * samples_per_prompt, (g + 1) * samples_per_prompt))
            best = max(idx, key=lambda i: rewards[i])
            worst = min(idx, key=lambda i: rewards[i])
            if rewards[best] == rewards[worst]:
                continue  # 우열을 가릴 수 없는 쌍은 버린다
            c, r = roll.responses[best], roll.responses[worst]
            if rng.random() < label_noise:
                c, r = r, c
            pairs.append(PreferencePair(uniq[g], c, r))
    if len(pairs) < n_pairs:
        raise RuntimeError(
            f"선호 쌍을 {n_pairs}개 만들지 못했습니다({len(pairs)}개). 정책의 응답 보상이 거의 같습니다 — SFT 를 더 학습하세요."
        )
    return pairs[:n_pairs]


def encode_pairs(tok: CharTokenizer, pairs: list[PreferencePair]):
    """chosen 과 rejected 를 [2N, L] 하나의 배치로 묶는다 (앞 N 개가 chosen).

    response_mask 는 응답 토큰(+EOS) 위치만 1 → DPO 의 log π(y|x) 계산에 사용.
    """
    seqs, resp = [], []
    for side in ("chosen", "rejected"):
        for p in pairs:
            pi = tok.encode(p.prompt, add_bos=True)
            ri = tok.encode(getattr(p, side), add_eos=True)
            seqs.append(pi + ri)
            resp.append([0] * len(pi) + [1] * len(ri))
    ids, mask = right_pad(seqs, tok.pad_id)
    rmask, _ = right_pad(resp, 0)
    return ids, mask, rmask


def train_reward_model(
    init_policy: TinyGPT,
    tok: CharTokenizer,
    pairs: list[PreferencePair],
    rng: random.Random,
    epochs: int = 3,
    batch_size: int = 64,
    lr: float = 3e-4,
    val_frac: float = 0.1,
    logger: Logger | None = None,
) -> RewardModel:
    """SFT 모델의 몸통을 그대로 가져와(초기화) 스칼라 head 를 붙여 학습한다."""
    rm = RewardModel(init_policy.cfg)
    rm.gpt.load_state_dict(init_policy.state_dict(), strict=False)
    opt = torch.optim.Adam(rm.parameters(), lr=lr)
    logger = logger or Logger()

    pairs = pairs[:]
    rng.shuffle(pairs)
    n_val = max(1, int(len(pairs) * val_frac))
    val, train = pairs[:n_val], pairs[n_val:]

    def batch_loss(batch):
        ids, mask, _ = encode_pairs(tok, batch)
        r = rm(ids, mask)
        rc, rr = r[: len(batch)], r[len(batch):]
        return -F.logsigmoid(rc - rr).mean(), (rc > rr).float().mean()

    step = 0
    for ep in range(epochs):
        rng.shuffle(train)
        rm.train()
        for i in range(0, len(train), batch_size):
            loss, acc = batch_loss(train[i : i + batch_size])
            opt.zero_grad()
            loss.backward()
            opt.step()
            step += 1
        rm.eval()
        with torch.no_grad():
            vloss, vacc = batch_loss(val)
        logger.log(step, {"epoch": ep + 1, "train_loss": loss.item(), "train_acc": acc.item(),
                          "val_loss": vloss.item(), "val_acc": vacc.item()})
    return rm
