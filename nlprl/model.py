"""처음부터 구현한 아주 작은 GPT (decoder-only Transformer).

RL 관점에서 이 모델은 **정책(policy) π(a_t | s_t)** 이다.
  - 상태 s_t  = 지금까지의 토큰 열 (프롬프트 + 이미 생성한 토큰)
  - 행동 a_t  = 다음 토큰
  - 정책 출력 = 마지막 위치의 logits → softmax → 다음 토큰 분포

PPO 에서는 상태 가치 V(s_t) 를 추정하는 **value head** 가 필요하고,
보상 모델(Reward Model)은 문장 전체에 스칼라 점수를 주는 **score head** 가 필요하다.
둘 다 같은 Transformer 몸통(trunk) 위에 Linear 하나를 얹어서 만든다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int
    block_size: int = 64
    n_layer: int = 2
    n_head: int = 4
    n_embd: int = 96
    dropout: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ModelOutput:
    logits: torch.Tensor  # [B, T, V]
    hidden: torch.Tensor  # [B, T, D]
    values: torch.Tensor | None = None  # [B, T]  (value head 가 있을 때만)


def position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """왼쪽 패딩(left padding)이 있어도 실제 토큰의 위치가 0부터 시작하도록 계산한다.

    예) mask = [0, 0, 1, 1, 1]  →  pos = [0, 0, 0, 1, 2]
    """
    return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)


def build_attn_mask(attention_mask: torch.Tensor) -> torch.Tensor:
    """causal mask 와 padding mask 를 합친 [B, 1, T, T] bool 마스크 (True = 볼 수 있음).

    패딩 위치의 query 는 볼 수 있는 key 가 하나도 없으면 softmax 가 NaN 이 되므로,
    대각선(자기 자신)은 항상 허용한다. 패딩 위치의 출력은 어차피 손실에서 마스킹된다.
    """
    B, T = attention_mask.shape
    device = attention_mask.device
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=device))
    key_ok = attention_mask.bool()[:, None, None, :]  # [B,1,1,T]
    mask = causal[None, None] & key_ok
    eye = torch.eye(T, dtype=torch.bool, device=device)[None, None]
    return mask | eye


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd)
        self.dropout = cfg.dropout

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        q, k, v = (t.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) for t in (q, k, v))
        y = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask, dropout_p=self.dropout if self.training else 0.0
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.proj(y)


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.ln1 = nn.LayerNorm(cfg.n_embd)
        self.attn = CausalSelfAttention(cfg)
        self.ln2 = nn.LayerNorm(cfg.n_embd)
        self.mlp = nn.Sequential(
            nn.Linear(cfg.n_embd, 4 * cfg.n_embd),
            nn.GELU(),
            nn.Linear(4 * cfg.n_embd, cfg.n_embd),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, attn_mask):
        x = x + self.attn(self.ln1(x), attn_mask)
        x = x + self.mlp(self.ln2(x))
        return x


class TinyGPT(nn.Module):
    """작은 GPT. `with_value_head=True` 이면 토큰마다 V(s_t) 도 함께 출력한다."""

    def __init__(self, cfg: GPTConfig, with_value_head: bool = False):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.block_size, cfg.n_embd)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.ln_f = nn.LayerNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.value_head = nn.Linear(cfg.n_embd, 1) if with_value_head else None
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def trunk(self, input_ids: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T = input_ids.shape
        assert T <= self.cfg.block_size, f"sequence length {T} > block_size {self.cfg.block_size}"
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        pos = position_ids_from_mask(attention_mask)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        mask = build_attn_mask(attention_mask)
        for blk in self.blocks:
            x = blk(x, mask)
        return self.ln_f(x)

    def forward(self, input_ids, attention_mask=None) -> ModelOutput:
        h = self.trunk(input_ids, attention_mask)
        values = self.value_head(h).squeeze(-1) if self.value_head is not None else None
        return ModelOutput(logits=self.lm_head(h), hidden=h, values=values)


def last_token_index(attention_mask: torch.Tensor) -> torch.Tensor:
    """각 행에서 마지막 실제 토큰(mask==1)의 인덱스. 왼쪽/오른쪽 패딩 모두 동작."""
    T = attention_mask.shape[1]
    return T - 1 - attention_mask.long().flip(-1).argmax(-1)


class RewardModel(nn.Module):
    """문장 전체에 스칼라 보상 r(x, y) 를 주는 모델 (마지막 토큰의 hidden → Linear)."""

    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.gpt = TinyGPT(cfg)
        self.score_head = nn.Linear(cfg.n_embd, 1)

    def forward(self, input_ids, attention_mask) -> torch.Tensor:
        h = self.gpt.trunk(input_ids, attention_mask)
        idx = last_token_index(attention_mask)
        h_last = h[torch.arange(h.shape[0], device=h.device), idx]
        return self.score_head(h_last).squeeze(-1)
