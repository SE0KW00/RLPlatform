"""롤아웃(rollout) 생성과 토큰별 log-prob 계산.

RL 루프의 공통 뼈대:
    1) 정책으로 응답을 샘플링한다            → generate()
    2) 보상을 계산한다                        → task.reward() 또는 RewardModel
    3) 응답 토큰들의 log π(a_t|s_t) 를 구한다 → token_logprobs()
    4) 알고리즘별 손실로 정책을 업데이트한다

텐서 레이아웃 (길이 L 인 시퀀스):
    input_ids      = [pad pad BOS t h e _ m o v i e | _ w a s _ g o o d . EOS pad]
    attention_mask = [ 0   0   1  ...                                       1   0 ]
    response_mask  = [ 0   0   0  ...           0   | 1 ...                 1   0 ]
                     \______ 왼쪽 패딩 프롬프트 ______/ \_____ 생성된 응답 _____/

token_logprobs() 의 출력은 길이 L-1 이다: logprobs[:, i] = log π(ids[i+1] | ids[:i+1]).
그래서 '행동 마스크'는 response_mask[:, 1:] 이다.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .model import TinyGPT
from .tokenizer import CharTokenizer


@dataclass
class Rollout:
    prompts: list[str]
    responses: list[str]
    input_ids: torch.Tensor  # [B, L]
    attention_mask: torch.Tensor  # [B, L]
    response_mask: torch.Tensor  # [B, L]

    @property
    def action_mask(self) -> torch.Tensor:
        """token_logprobs() 출력과 정렬된 [B, L-1] 마스크."""
        return self.response_mask[:, 1:].float()


def left_pad(seqs: list[list[int]], pad_id: int, device=None):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, L - len(s):] = torch.tensor(s, dtype=torch.long)
        mask[i, L - len(s):] = 1
    return ids, mask


def right_pad(seqs: list[list[int]], pad_id: int, device=None):
    L = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), L), pad_id, dtype=torch.long, device=device)
    mask = torch.zeros((len(seqs), L), dtype=torch.long, device=device)
    for i, s in enumerate(seqs):
        ids[i, : len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, : len(s)] = 1
    return ids, mask


@torch.no_grad()
def generate(
    model: TinyGPT,
    tok: CharTokenizer,
    prompts: list[str],
    max_new_tokens: int,
    temperature: float = 1.0,
    greedy: bool = False,
) -> Rollout:
    """프롬프트 배치에서 응답을 샘플링한다. (교육용이라 KV-cache 없이 매번 전체를 다시 계산)"""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    prompt_ids = [tok.encode(p, add_bos=True) for p in prompts]
    ids, mask = left_pad(prompt_ids, tok.pad_id, device)
    B, P = ids.shape
    assert P + max_new_tokens <= model.cfg.block_size, "block_size 를 늘리거나 max_new_tokens 를 줄이세요"
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    resp_mask = torch.zeros_like(mask)

    for _ in range(max_new_tokens):
        logits = model(ids, mask).logits[:, -1, :]
        if greedy:
            nxt = logits.argmax(-1)
        else:
            probs = F.softmax(logits / temperature, dim=-1)
            nxt = torch.multinomial(probs, 1).squeeze(-1)
        alive = ~finished
        nxt = torch.where(alive, nxt, torch.full_like(nxt, tok.pad_id))
        ids = torch.cat([ids, nxt[:, None]], dim=1)
        mask = torch.cat([mask, alive.long()[:, None]], dim=1)
        resp_mask = torch.cat([resp_mask, alive.long()[:, None]], dim=1)
        finished |= nxt == tok.eos_id
        if finished.all():
            break

    responses = [tok.decode(row[P:].tolist()) for row in ids]
    model.train(was_training)
    return Rollout(prompts, responses, ids, mask, resp_mask)


def token_logprobs(model: TinyGPT, input_ids, attention_mask, return_values: bool = False):
    """각 위치에서 '실제로 다음에 나온 토큰'의 log-prob [B, L-1] (+ value [B, L-1]).

    values[:, i] 는 ids[i+1] 를 고르기 직전 상태(= ids[:i+1])의 가치 V(s) 이다.
    """
    out = model(input_ids, attention_mask)
    logp = F.log_softmax(out.logits[:, :-1].float(), dim=-1)
    lp = logp.gather(-1, input_ids[:, 1:, None]).squeeze(-1)
    if return_values:
        return lp, out.values[:, :-1]
    return lp


def entropy_from_logits(logits: torch.Tensor) -> torch.Tensor:
    logp = F.log_softmax(logits.float(), dim=-1)
    return -(logp.exp() * logp).sum(-1)


def masked_mean(x: torch.Tensor, mask: torch.Tensor, dim=None) -> torch.Tensor:
    if dim is None:
        return (x * mask).sum() / mask.sum().clamp(min=1)
    return (x * mask).sum(dim) / mask.sum(dim).clamp(min=1)


def masked_whiten(x: torch.Tensor, mask: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    mean = masked_mean(x, mask)
    var = masked_mean((x - mean) ** 2, mask)
    return (x - mean) * torch.rsqrt(var + eps) * mask
