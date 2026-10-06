"""학습 단계(SFT → RL / RM / DPO)를 실행하는 함수들.

CLI(`cli.py`)와 웹 앱(`web/`)이 같은 함수를 호출하므로, 어디서 실행하든
결과는 `runs/<task>/<name>/` 에 같은 형식(policy.pt, history.json)으로 저장된다.
"""

from __future__ import annotations

import json
from pathlib import Path

from .algorithms import (DPOConfig, GRPOConfig, PPOConfig, RLConfig, evaluate, reward_model_fn,
                         train_dpo, train_grpo, train_ppo, train_reinforce)
from .preference import PreferencePair, make_preference_pairs, train_reward_model
from .pretrain import pretrain
from .tasks import get_task
from .tokenizer import CharTokenizer
from .utils import Logger, load_policy, load_reward_model, save_checkpoint, set_seed

RUNS_ROOT = Path("runs")

# 과제별로 잘 동작하는 기본값 (CPU 로 수 분 이내)
DEFAULTS = {
    "sentiment": {
        "pretrain_steps": 1500,
        "reinforce": dict(steps=150, lr=1e-4, kl_coef=0.3),
        "ppo": dict(steps=100, lr=1e-4, kl_coef=0.3),
        "grpo": dict(steps=150, lr=1e-4, kl_coef=0.3),
    },
    "arithmetic": {
        "pretrain_steps": 2500,
        "reinforce": dict(steps=150, lr=1e-4, kl_coef=0.0),
        "ppo": dict(steps=100, lr=1e-4, kl_coef=0.0),
        "grpo": dict(steps=200, lr=1e-4, kl_coef=0.0),
    },
}


def run_dir(task: str, name: str) -> Path:
    return RUNS_ROOT / task / name


def sft_path(task: str) -> Path:
    return run_dir(task, "sft") / "policy.pt"


def rm_path(task: str) -> Path:
    return run_dir(task, "rm") / "rm.pt"


def pairs_path(task: str) -> Path:
    return run_dir(task, "rm") / "pairs.jsonl"


def run_pretrain(task: str, seed: int = 0, steps: int | None = None, logger: Logger | None = None,
                 log_every: int = 250) -> Path:
    rng = set_seed(seed)
    out = run_dir(task, "sft")
    logger = logger or Logger(out)
    logger.out_dir = out
    steps = steps or DEFAULTS[task]["pretrain_steps"]
    model = pretrain(get_task(task), CharTokenizer(), rng, steps=steps, logger=logger, log_every=log_every)
    save_checkpoint(out / "policy.pt", model, task)
    return out / "policy.pt"


def run_train(
    task: str,
    algo: str,
    seed: int = 0,
    init: str | Path | None = None,
    reward: str = "task",
    rm: str | Path | None = None,
    name: str | None = None,
    baseline: str = "batch_mean",
    logger: Logger | None = None,
    **overrides,
) -> Path:
    """온라인 RL (reinforce / ppo / grpo). overrides 로 RLConfig 필드를 덮어쓴다."""
    rng = set_seed(seed)
    task_obj, tok = get_task(task), CharTokenizer()
    init = init or sft_path(task)
    policy, _ = load_policy(init, with_value_head=(algo == "ppo"))

    params = dict(DEFAULTS[task][algo])
    params.update({k: v for k, v in overrides.items() if v is not None})

    reward_fn = reward_model_fn(load_reward_model(rm or rm_path(task))) if reward == "rm" else None

    out = run_dir(task, name or (algo + ("_rm" if reward == "rm" else "")))
    logger = logger or Logger(out)
    logger.out_dir = out
    ev, samples = evaluate(policy, tok, task_obj, rng)
    logger.log(0, ev, samples)

    if algo == "reinforce":
        train_reinforce(policy, tok, task_obj, RLConfig(**params), rng, reward_fn, baseline=baseline, logger=logger)
    elif algo == "ppo":
        train_ppo(policy, tok, task_obj, PPOConfig(**params), rng, reward_fn, logger=logger)
    elif algo == "grpo":
        train_grpo(policy, tok, task_obj, GRPOConfig(**params), rng, reward_fn, logger=logger)
    else:
        raise ValueError(f"unknown algo: {algo}")
    save_checkpoint(out / "policy.pt", policy, task, {"algo": algo, **params})
    return out / "policy.pt"


def run_reward_model(task: str, seed: int = 0, init: str | Path | None = None, pairs: int = 2000,
                     label_noise: float = 0.0, epochs: int = 3, logger: Logger | None = None) -> Path:
    rng = set_seed(seed)
    task_obj, tok = get_task(task), CharTokenizer()
    policy, _ = load_policy(init or sft_path(task))
    pair_list = make_preference_pairs(policy, tok, task_obj, rng, pairs, label_noise=label_noise)
    out = run_dir(task, "rm")
    out.mkdir(parents=True, exist_ok=True)
    pairs_path(task).write_text("\n".join(json.dumps(p.__dict__) for p in pair_list))
    logger = logger or Logger(out)
    logger.out_dir = out
    model = train_reward_model(policy, tok, pair_list, rng, epochs=epochs, logger=logger)
    save_checkpoint(out / "rm.pt", model, task)
    return out / "rm.pt"


def load_pairs(task: str) -> list[PreferencePair]:
    return [PreferencePair(**json.loads(l)) for l in pairs_path(task).read_text().splitlines()]


def run_dpo(task: str, seed: int = 0, init: str | Path | None = None, pairs: int = 2000, regen_pairs: bool = False,
            label_noise: float = 0.0, epochs: int = 1, beta: float = 0.1, lr: float = 2e-5,
            name: str | None = None, logger: Logger | None = None, log_every: int = 20) -> Path:
    rng = set_seed(seed)
    task_obj, tok = get_task(task), CharTokenizer()
    policy, _ = load_policy(init or sft_path(task))
    if pairs_path(task).exists() and not regen_pairs:
        pair_list = load_pairs(task)
    else:
        pair_list = make_preference_pairs(policy, tok, task_obj, rng, pairs, label_noise=label_noise)
    out = run_dir(task, name or "dpo")
    logger = logger or Logger(out)
    logger.out_dir = out
    ev, samples = evaluate(policy, tok, task_obj, rng)
    logger.log(0, ev, samples)
    cfg = DPOConfig(epochs=epochs, beta=beta, lr=lr, log_every=log_every)
    train_dpo(policy, tok, task_obj, pair_list, cfg, rng, logger=logger)
    save_checkpoint(out / "policy.pt", policy, task, {"algo": "dpo", "beta": beta})
    return out / "policy.pt"
