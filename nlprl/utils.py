"""시드 고정, 체크포인트, 로그 기록 등 공통 유틸리티."""

from __future__ import annotations

import json
import random
import time
from pathlib import Path

import torch

from .model import GPTConfig, RewardModel, TinyGPT


def set_seed(seed: int) -> random.Random:
    random.seed(seed)
    torch.manual_seed(seed)
    return random.Random(seed)


def save_checkpoint(path: str | Path, model: torch.nn.Module, task: str, extra: dict | None = None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    state = {k: v for k, v in model.state_dict().items() if not k.startswith("value_head.")}
    torch.save(
        {"config": model.cfg.to_dict(), "task": task, "kind": type(model).__name__,
         "state_dict": state, "extra": extra or {}},
        path,
    )


def load_policy(path: str | Path, with_value_head: bool = False) -> tuple[TinyGPT, str]:
    """체크포인트에서 정책을 불러온다. value head 는 새로 초기화된다 (PPO 용)."""
    ckpt = torch.load(path, map_location="cpu")
    model = TinyGPT(GPTConfig(**ckpt["config"]), with_value_head=with_value_head)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    return model, ckpt["task"]


def load_reward_model(path: str | Path) -> RewardModel:
    ckpt = torch.load(path, map_location="cpu")
    rm = RewardModel(GPTConfig(**ckpt["config"]))
    rm.load_state_dict(ckpt["state_dict"])
    return rm


class Logger:
    """콘솔 출력 + history.json 저장."""

    def __init__(self, out_dir: str | Path | None = None, quiet: bool = False, on_log=None):
        self.history: list[dict] = []
        self.out_dir = Path(out_dir) if out_dir else None
        self.quiet = quiet
        self.on_log = on_log  # 콜백(row, samples): 웹 앱이 실시간 스트리밍/중단에 사용
        self.t0 = time.time()

    def log(self, step: int, metrics: dict, samples: list[str] | None = None):
        row = {"step": step, "time": round(time.time() - self.t0, 1), **{k: float(v) for k, v in metrics.items()}}
        self.history.append(row)
        if not self.quiet:
            body = "  ".join(f"{k}={v:.3f}" for k, v in row.items() if k not in ("step", "time"))
            print(f"[step {step:4d} | {row['time']:6.1f}s] {body}")
            for s in samples or []:
                print(f"      └ {s!r}")
        self.save()
        if self.on_log:
            self.on_log(row, samples or [])

    def save(self):
        if self.out_dir:
            self.out_dir = Path(self.out_dir)
            self.out_dir.mkdir(parents=True, exist_ok=True)
            (self.out_dir / "history.json").write_text(json.dumps(self.history, indent=1))
