"""명령줄 인터페이스.

    python -m nlprl pretrain --task sentiment
    python -m nlprl train    --task sentiment --algo ppo
    python -m nlprl reward-model --task sentiment
    python -m nlprl train    --task sentiment --algo ppo --reward rm
    python -m nlprl dpo      --task sentiment
    python -m nlprl train    --task arithmetic --algo grpo
    python -m nlprl sample   --ckpt runs/sentiment/ppo/policy.pt
    python -m nlprl plot     runs/sentiment/*/history.json --metric eval/reward
    python -m nlprl serve    # 웹 학습 플랫폼 → http://127.0.0.1:8000
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .generation import generate
from .pipeline import run_dpo, run_pretrain, run_reward_model, run_train
from .tasks import TASKS, get_task
from .tokenizer import CharTokenizer
from .utils import load_policy, set_seed


def cmd_pretrain(a):
    path = run_pretrain(a.task, a.seed, a.steps)
    print(f"saved → {path}")


def cmd_train(a):
    params = {k: getattr(a, k) for k in ("steps", "lr", "kl_coef", "batch_size", "temperature", "group_size")}
    if a.algo != "grpo":
        params.pop("group_size")
    print(f"== {a.algo} on {a.task} | reward={a.reward} | overrides={ {k: v for k, v in params.items() if v is not None} }")
    path = run_train(a.task, a.algo, a.seed, init=a.init, reward=a.reward, rm=a.rm, name=a.name,
                     baseline=a.baseline, **params)
    print(f"saved → {path}")


def cmd_reward_model(a):
    print(f"선호 쌍 {a.pairs}개 생성 + 보상 모델 학습 (label_noise={a.label_noise}) ...")
    path = run_reward_model(a.task, a.seed, a.init, a.pairs, a.label_noise, a.epochs)
    print(f"saved → {path}")


def cmd_dpo(a):
    path = run_dpo(a.task, a.seed, a.init, a.pairs, a.regen_pairs, a.label_noise, a.epochs, a.beta, a.lr, a.name)
    print(f"saved → {path}")


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


def cmd_serve(a):
    from .web.server import serve
    serve(a.host, a.port, open_browser=a.open)


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

    sp = sub.add_parser("serve", help="웹 학습 플랫폼 실행 (레슨·시각화·실험실)")
    sp.add_argument("--host", default="127.0.0.1")
    sp.add_argument("--port", type=int, default=8000)
    sp.add_argument("--open", action="store_true", help="브라우저 자동 열기")
    sp.set_defaults(fn=cmd_serve)

    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
