"""학습된 체크포인트를 '들여다보는' 함수들 — 웹 시각화의 데이터 소스.

모두 JSON 으로 직렬화 가능한 dict/list 를 반환한다.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch
import torch.nn.functional as F

from .. import pipeline
from ..algorithms.grpo import group_advantages
from ..generation import generate, token_logprobs
from ..preference import encode_pairs
from ..tasks import get_task
from ..tokenizer import CharTokenizer
from ..utils import load_policy, load_reward_model

TOK = CharTokenizer()
_cache: dict[tuple[str, float], object] = {}


def _cached(path: Path, loader):
    key = (str(path), path.stat().st_mtime)
    if key not in _cache:
        _cache[key] = loader(path)
    return _cache[key]


def policy(path) -> tuple:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"체크포인트가 없습니다: {path} (실험실에서 먼저 학습하세요)")
    return _cached(path, load_policy)


def reward_model(task: str):
    path = pipeline.rm_path(task)
    if not path.exists():
        raise FileNotFoundError(f"보상 모델이 없습니다: {path} (실험실에서 '보상 모델' 학습을 먼저 실행하세요)")
    return _cached(path, load_reward_model)


def tok_str(i: int) -> str:
    s = TOK.itos[int(i)]
    return {"<pad>": "∅", "<bos>": "BOS", "<eos>": "EOS"}.get(s, s)


def list_checkpoints() -> list[dict]:
    out = []
    root = pipeline.RUNS_ROOT
    if not root.exists():
        return out
    for p in sorted(root.glob("*/*/policy.pt")):
        out.append({"task": p.parent.parent.name, "name": p.parent.name, "path": str(p)})
    return out


def list_runs() -> list[dict]:
    root = pipeline.RUNS_ROOT
    out = []
    if root.exists():
        for h in sorted(root.glob("*/*/history.json")):
            out.append({"task": h.parent.parent.name, "name": h.parent.name,
                        "has_policy": (h.parent / "policy.pt").exists(),
                        "has_rm": (h.parent / "rm.pt").exists()})
    return out


@torch.no_grad()
def trace(ckpt: str, prompt: str, temperature: float = 1.0, greedy: bool = False, top_k: int = 8,
          seed: int = 0, ref_ckpt: str | None = None) -> dict:
    """프롬프트 하나에 대해 토큰을 한 개씩 생성하며 매 스텝의 정책 분포를 기록한다 (MDP 의 한 에피소드)."""
    model, task_name = policy(ckpt)
    task = get_task(task_name)
    ref_path = Path(ref_ckpt) if ref_ckpt else pipeline.sft_path(task_name)
    ref = policy(ref_path)[0] if ref_path.exists() else None
    torch.manual_seed(seed)
    ids = TOK.encode(prompt, add_bos=True)
    steps = []
    for _ in range(task.max_new_tokens):
        x = torch.tensor([ids])
        logits = model(x).logits[0, -1].float()
        probs = F.softmax(logits / temperature, dim=-1)
        nxt = int(probs.argmax()) if greedy else int(torch.multinomial(probs, 1))
        logp = F.log_softmax(logits, dim=-1)
        top = torch.topk(probs, top_k)
        step = {
            "state": TOK.decode(ids),
            "token": tok_str(nxt),
            "logp": float(logp[nxt]),
            "entropy": float(-(logp.exp() * logp).sum()),
            "top": [{"token": tok_str(i), "p": float(p)} for p, i in zip(top.values, top.indices)],
        }
        if ref is not None:
            rlogp = F.log_softmax(ref(x).logits[0, -1].float(), dim=-1)
            step["ref_logp"] = float(rlogp[nxt])
            step["kl_term"] = step["logp"] - step["ref_logp"]
            step["ref_top"] = {tok_str(i): float(rlogp[i].exp()) for i in top.indices}
        steps.append(step)
        ids.append(nxt)
        if nxt == TOK.eos_id:
            break
    response = TOK.decode(ids[len(TOK.encode(prompt, add_bos=True)):])
    return {"task": task_name, "prompt": prompt, "response": response, "reward": task.reward(prompt, response),
            "steps": steps, "has_ref": ref is not None}


@torch.no_grad()
def rollout(ckpt: str, prompts: list[str] | None = None, n: int = 8, group_size: int = 1,
            temperature: float = 1.0, seed: int = 0, scale_rewards: bool = True, with_tensors: bool = False) -> dict:
    """배치 롤아웃 + 보상 (+ GRPO 그룹 advantage, + 텐서 레이아웃)."""
    model, task_name = policy(ckpt)
    task = get_task(task_name)
    rng = random.Random(seed)
    torch.manual_seed(seed)
    base = prompts or task.sample_prompts(max(1, n // group_size), rng)
    full = [p for p in base for _ in range(group_size)]
    roll = generate(model, TOK, full, task.max_new_tokens, temperature=temperature)
    rewards = task.rewards(full, roll.responses)
    out = {
        "task": task_name,
        "samples": [{"prompt": p, "response": r, "reward": rw} for p, r, rw in zip(full, roll.responses, rewards)],
        "metrics": task.metrics(full, roll.responses),
    }
    if group_size > 1:
        adv = group_advantages(torch.tensor(rewards), group_size, scale=scale_rewards)
        for s, a in zip(out["samples"], adv.tolist()):
            s["advantage"] = a
    if with_tensors:
        rows = min(len(full), 6)
        lp = token_logprobs(model, roll.input_ids[:rows], roll.attention_mask[:rows])
        out["tensors"] = {
            "tokens": [[tok_str(i) for i in row] for row in roll.input_ids[:rows].tolist()],
            "input_ids": roll.input_ids[:rows].tolist(),
            "attention_mask": roll.attention_mask[:rows].tolist(),
            "response_mask": roll.response_mask[:rows].tolist(),
            "logprobs": [[round(v, 3) for v in row] for row in lp.tolist()],
        }
    return out


@torch.no_grad()
def rm_score(task: str, items: list[dict]) -> dict:
    """[{prompt, response}] → 규칙 보상 vs 학습된 보상 모델 점수."""
    task_obj = get_task(task)
    rm = reward_model(task)
    seqs = [TOK.encode(it["prompt"], add_bos=True) + TOK.encode(it["response"], add_eos=True) for it in items]
    from ..generation import right_pad
    ids, mask = right_pad(seqs, TOK.pad_id)
    scores = rm(ids, mask).tolist()
    return {"items": [
        {**it, "rule_reward": task_obj.reward(it["prompt"], it["response"]), "rm_score": s,
         "well_formed": getattr(task_obj, "is_well_formed", lambda r: None)(it["response"])}
        for it, s in zip(items, scores)
    ]}


@torch.no_grad()
def dpo_pairs(task: str, ckpt: str | None = None, n: int = 8, beta: float = 0.1) -> dict:
    """선호 쌍 + (정책, 참조) 하의 log-prob 과 DPO 암묵적 보상."""
    path = pipeline.pairs_path(task)
    if not path.exists():
        raise FileNotFoundError(f"선호 데이터가 없습니다: {path} (실험실에서 '보상 모델' 학습을 먼저 실행하세요)")
    pairs = pipeline.load_pairs(task)[:n]
    ref, _ = policy(pipeline.sft_path(task))
    pol, _ = policy(ckpt) if ckpt else (ref, task)
    ids, mask, rmask = encode_pairs(TOK, pairs)
    from ..algorithms.dpo import sequence_logprob
    pi = sequence_logprob(pol, ids, mask, rmask).tolist()
    rf = sequence_logprob(ref, ids, mask, rmask).tolist()
    out = []
    for i, p in enumerate(pairs):
        c, r = i, i + len(pairs)
        rc, rr = beta * (pi[c] - rf[c]), beta * (pi[r] - rf[r])
        out.append({"prompt": p.prompt, "chosen": p.chosen, "rejected": p.rejected,
                    "pi_chosen": pi[c], "ref_chosen": rf[c], "pi_rejected": pi[r], "ref_rejected": rf[r],
                    "reward_chosen": rc, "reward_rejected": rr,
                    "loss": float(-F.logsigmoid(torch.tensor(rc - rr)))})
    return {"pairs": out, "beta": beta}
