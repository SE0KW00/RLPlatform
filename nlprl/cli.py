"""명령줄 인터페이스.

    python -m nlprl pretrain --task sentiment
    python -m nlprl train    --task sentiment --algo ppo
    python -m nlprl reward-model --task sentiment
    python -m nlprl train    --task sentiment --algo ppo --reward rm
    python -m nlprl dpo      --task sentiment
    python -m nlprl train    --task arithmetic --algo grpo
    python -m nlprl sample   --ckpt runs/sentiment/ppo/policy.pt
    python -m nlprl plot     runs/sentiment/*/history.json --metric eval/reward
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .algorithms import (DPOConfig, GRPOConfig, PPOConfig, RLConfig, evaluate, reward_model_fn,
                         train_dpo, train_grpo, train_ppo, train_reinforce)
from .generation import generate
from .preference import make_preference_pairs, train_reward_model
from .pretrain import pretrain
from .tasks import TASKS, get_task
from .tokenizer import CharTokenizer
from .utils import Logger, load_policy, load_reward_model, save_checkpoint, set_seed

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
    return Path("runs") / task / name


def sft_path(task: str) -> Path:
    return run_dir(task, "sft") / "policy.pt"


def cmd_pretrain(a):
    rng = set_seed(a.seed)
    task, tok = get_task(a.task), CharTokenizer()
    out = run_dir(a.task, "sft")
    steps = a.steps or DEFAULTS[a.task]["pretrain_steps"]
    model = pretrain(task, tok, rng, steps=steps, logger=Logger(out))
    save_checkpoint(out / "policy.pt", model, a.task)
    print(f"saved → {out / 'policy.pt'}")


def cmd_train(a):
    rng = set_seed(a.seed)
    task, tok = get_task(a.task), CharTokenizer()
    init = a.init or sft_path(a.task)
    policy, _ = load_policy(init, with_value_head=(a.algo == "ppo"))

    params = dict(DEFAULTS[a.task][a.algo])
    for k in ("steps", "lr", "kl_coef", "batch_size", "temperature"):
        if getattr(a, k) is not None:
            params[k] = getattr(a, k)

    reward_fn = None
    if a.reward == "rm":
        rm_path = a.rm or (run_dir(a.task, "rm") / "rm.pt")
        reward_fn = reward_model_fn(load_reward_model(rm_path))

    name = a.name or (a.algo + ("_rm" if a.reward == "rm" else ""))
    out = run_dir(a.task, name)
    logger = Logger(out)
    print(f"== {a.algo} on {a.task} | init={init} | reward={a.reward} | {params}")
    ev, samples = evaluate(policy, tok, task, rng)
    logger.log(0, ev, samples)

    if a.algo == "reinforce":
        train_reinforce(policy, tok, task, RLConfig(**params), rng, reward_fn, baseline=a.baseline, logger=logger)
    elif a.algo == "ppo":
        train_ppo(policy, tok, task, PPOConfig(**params), rng, reward_fn, logger=logger)
    elif a.algo == "grpo":
        if a.group_size:
            params["group_size"] = a.group_size
        train_grpo(policy, tok, task, GRPOConfig(**params), rng, reward_fn, logger=logger)
    save_checkpoint(out / "policy.pt", policy, a.task, {"algo": a.algo, **params})
    print(f"saved → {out / 'policy.pt'}")


def cmd_reward_model(a):
    rng = set_seed(a.seed)
    task, tok = get_task(a.task), CharTokenizer()
    policy, _ = load_policy(a.init or sft_path(a.task))
    print(f"선호 쌍 {a.pairs}개 생성 중 (label_noise={a.label_noise}) ...")
    pairs = make_preference_pairs(policy, tok, task, rng, a.pairs, label_noise=a.label_noise)
    out = run_dir(a.task, "rm")
    out.mkdir(parents=True, exist_ok=True)
    (out / "pairs.jsonl").write_text("\n".join(json.dumps(p.__dict__) for p in pairs))
    for p in pairs[:3]:
        print(f"  {p.prompt!r}: chosen={p.chosen!r}  rejected={p.rejected!r}")
    rm = train_reward_model(policy, tok, pairs, rng, epochs=a.epochs, logger=Logger(out))
    save_checkpoint(out / "rm.pt", rm, a.task)
    print(f"saved → {out / 'rm.pt'}")


def cmd_dpo(a):
    rng = set_seed(a.seed)
    task, tok = get_task(a.task), CharTokenizer()
    policy, _ = load_policy(a.init or sft_path(a.task))
    pairs_file = run_dir(a.task, "rm") / "pairs.jsonl"
    if pairs_file.exists() and not a.regen_pairs:
        from .preference import PreferencePair
        pairs = [PreferencePair(**json.loads(l)) for l in pairs_file.read_text().splitlines()]
        print(f"{pairs_file} 에서 선호 쌍 {len(pairs)}개를 불러왔습니다")
    else:
        pairs = make_preference_pairs(policy, tok, task, rng, a.pairs, label_noise=a.label_noise)
    out = run_dir(a.task, a.name or "dpo")
    logger = Logger(out)
    ev, samples = evaluate(policy, tok, task, rng)
    logger.log(0, ev, samples)
    cfg = DPOConfig(epochs=a.epochs, beta=a.beta, lr=a.lr)
    train_dpo(policy, tok, task, pairs, cfg, rng, logger=logger)
    save_checkpoint(out / "policy.pt", policy, a.task, {"algo": "dpo", "beta": a.beta})
    print(f"saved → {out / 'policy.pt'}")


def cmd_sample(a):
    set_seed(a.seed)
    import random
    tok = CharTokenizer()
    policy, task_name = load_policy(a.ckpt)
    task = get_task(task_name)
    prompts = a.prompt or task.sample_prompts(a.n, random.Random(a.seed))
    roll = generate(policy, tok, prompts, task.max_new_tokens, temperature=a.temperature, greedy=a.greedy)
    for p, r in zip(prompts, roll.responses):
        print(f"[reward={task.reward(p, r):+.1f}] {p}{r}")
    print(task.metrics(prompts, roll.responses))


def cmd_plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4))
    for f in a.files:
        hist = json.loads(Path(f).read_text())
        xs = [h["step"] for h in hist if a.metric in h]
        ys = [h[a.metric] for h in hist if a.metric in h]
        if not xs:
            continue  # 해당 지표가 없는 실행 (예: rm, sft) 은 건너뜀
        ax.plot(xs, ys, marker="o", ms=3, label=Path(f).parent.name)
    ax.set_xlabel("step")
    ax.set_ylabel(a.metric)
    ax.grid(alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(a.out, dpi=120)
    print(f"saved → {a.out}")


def main(argv=None):
    p = argparse.ArgumentParser(prog="nlprl", description="NLP 강화학습 학습 플랫폼")
    sub = p.add_subparsers(dest="cmd", required=True)

    def common(sp):
        sp.add_argument("--task", choices=list(TASKS), default="sentiment")
        sp.add_argument("--seed", type=int, default=0)

    sp = sub.add_parser("pretrain", help="0단계: SFT 모델 만들기")
    common(sp)
    sp.add_argument("--steps", type=int)
    sp.set_defaults(fn=cmd_pretrain)

    sp = sub.add_parser("train", help="온라인 RL (reinforce / ppo / grpo)")
    common(sp)
    sp.add_argument("--algo", choices=["reinforce", "ppo", "grpo"], required=True)
    sp.add_argument("--init", help="초기 정책 체크포인트 (기본: runs/<task>/sft/policy.pt)")
    sp.add_argument("--reward", choices=["task", "rm"], default="task", help="규칙 보상 / 학습된 보상 모델")
    sp.add_argument("--rm", help="보상 모델 체크포인트 (기본: runs/<task>/rm/rm.pt)")
    sp.add_argument("--steps", type=int)
    sp.add_argument("--lr", type=float)
    sp.add_argument("--kl-coef", dest="kl_coef", type=float)
    sp.add_argument("--batch-size", dest="batch_size", type=int)
    sp.add_argument("--temperature", type=float)
    sp.add_argument("--baseline", choices=["none", "batch_mean", "ema"], default="batch_mean")
    sp.add_argument("--group-size", dest="group_size", type=int)
    sp.add_argument("--name", help="실행 이름 (기본: 알고리즘 이름)")
    sp.set_defaults(fn=cmd_train)

    sp = sub.add_parser("reward-model", help="선호 데이터 생성 + 보상 모델 학습")
    common(sp)
    sp.add_argument("--init")
    sp.add_argument("--pairs", type=int, default=2000)
    sp.add_argument("--label-noise", dest="label_noise", type=float, default=0.0)
    sp.add_argument("--epochs", type=int, default=3)
    sp.set_defaults(fn=cmd_reward_model)

    sp = sub.add_parser("dpo", help="DPO (오프라인 선호 최적화)")
    common(sp)
    sp.add_argument("--init")
    sp.add_argument("--pairs", type=int, default=2000)
    sp.add_argument("--regen-pairs", dest="regen_pairs", action="store_true")
    sp.add_argument("--label-noise", dest="label_noise", type=float, default=0.0)
    sp.add_argument("--epochs", type=int, default=1)
    sp.add_argument("--beta", type=float, default=0.1)
    sp.add_argument("--lr", type=float, default=2e-5)
    sp.add_argument("--name")
    sp.set_defaults(fn=cmd_dpo)

    sp = sub.add_parser("sample", help="체크포인트에서 샘플 생성")
    sp.add_argument("--ckpt", required=True)
    sp.add_argument("--n", type=int, default=10)
    sp.add_argument("--prompt", action="append", help="직접 프롬프트 지정 (여러 번 가능)")
    sp.add_argument("--temperature", type=float, default=1.0)
    sp.add_argument("--greedy", action="store_true")
    sp.add_argument("--seed", type=int, default=0)
    sp.set_defaults(fn=cmd_sample)

    sp = sub.add_parser("plot", help="history.json 들을 한 그래프로 비교")
    sp.add_argument("files", nargs="+")
    sp.add_argument("--metric", default="eval/reward")
    sp.add_argument("--out", default="plot.png")
    sp.set_defaults(fn=cmd_plot)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
