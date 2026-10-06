"""브라우저에서 시작한 학습을 백그라운드 스레드로 실행하고, 진행 상황을 이벤트로 쌓는다.

CPU 를 나눠 쓰면 모두 느려지므로 한 번에 하나의 작업만 실행하고 나머지는 대기열에 둔다.
"""

from __future__ import annotations

import itertools
import queue
import threading
import time
import traceback
from dataclasses import dataclass, field

from .. import pipeline
from ..utils import Logger


class JobStopped(Exception):
    pass


@dataclass
class Job:
    id: int
    kind: str  # pretrain | train | reward_model | dpo
    params: dict
    status: str = "queued"  # queued | running | done | error | stopped
    events: list = field(default_factory=list)
    error: str | None = None
    result: str | None = None
    created: float = field(default_factory=time.time)
    stop_flag: threading.Event = field(default_factory=threading.Event)
    cond: threading.Condition = field(default_factory=threading.Condition)

    def emit(self, kind: str, data: dict):
        with self.cond:
            self.events.append({"seq": len(self.events), "type": kind, **data})
            self.cond.notify_all()

    def summary(self) -> dict:
        return {"id": self.id, "kind": self.kind, "params": self.params, "status": self.status,
                "error": self.error, "result": self.result, "created": self.created,
                "n_events": len(self.events), "title": self.title}

    @property
    def title(self) -> str:
        p = self.params
        if self.kind == "train":
            return f"{p.get('task')} · {p.get('algo')}" + (" (RM 보상)" if p.get("reward") == "rm" else "")
        return f"{p.get('task')} · {self.kind}"


# 웹에서 실행할 때는 그래프가 촘촘하도록 더 자주 기록한다
LIVE_LOG = {"pretrain": 100, "train": 5, "dpo": 5}


class JobManager:
    def __init__(self):
        self.jobs: dict[int, Job] = {}
        self._ids = itertools.count(1)
        self._queue: queue.Queue[Job] = queue.Queue()
        self._lock = threading.Lock()
        threading.Thread(target=self._worker, daemon=True).start()

    def submit(self, kind: str, params: dict) -> Job:
        if kind not in ("pretrain", "train", "reward_model", "dpo"):
            raise ValueError(f"unknown job kind: {kind}")
        with self._lock:
            job = Job(next(self._ids), kind, params)
            self.jobs[job.id] = job
        job.emit("status", {"status": "queued"})
        self._queue.put(job)
        return job

    def stop(self, job_id: int):
        job = self.jobs[job_id]
        job.stop_flag.set()
        if job.status == "queued":
            job.status = "stopped"
            job.emit("status", {"status": "stopped"})

    def _worker(self):
        while True:
            job = self._queue.get()
            if job.status == "stopped":
                continue
            self._run(job)

    def _run(self, job: Job):
        job.status = "running"
        job.emit("status", {"status": "running"})

        def on_log(row, samples):
            job.emit("log", {"row": row, "samples": samples})
            if job.stop_flag.is_set():
                raise JobStopped()

        logger = Logger(quiet=True, on_log=on_log)
        p = dict(job.params)
        try:
            if job.kind == "pretrain":
                path = pipeline.run_pretrain(p["task"], int(p.get("seed", 0)), _int(p.get("steps")), logger,
                                             log_every=LIVE_LOG["pretrain"])
            elif job.kind == "train":
                over = {k: _num(p.get(k)) for k in ("steps", "lr", "kl_coef", "batch_size", "temperature")}
                if p["algo"] == "grpo":
                    over["group_size"] = _int(p.get("group_size"))
                over = {k: (int(v) if k in ("steps", "batch_size", "group_size") and v is not None else v)
                        for k, v in over.items()}
                path = pipeline.run_train(p["task"], p["algo"], int(p.get("seed", 0)), reward=p.get("reward", "task"),
                                          name=p.get("name") or None, baseline=p.get("baseline", "batch_mean"),
                                          logger=logger, log_every=LIVE_LOG["train"], eval_size=128, **over)
            elif job.kind == "reward_model":
                path = pipeline.run_reward_model(p["task"], int(p.get("seed", 0)), pairs=_int(p.get("pairs")) or 2000,
                                                 label_noise=_num(p.get("label_noise")) or 0.0, logger=logger)
            else:
                path = pipeline.run_dpo(p["task"], int(p.get("seed", 0)), beta=_num(p.get("beta")) or 0.1,
                                        lr=_num(p.get("lr")) or 2e-5, epochs=_int(p.get("epochs")) or 1,
                                        name=p.get("name") or None, logger=logger, log_every=LIVE_LOG["dpo"])
            job.result = str(path)
            job.status = "done"
        except JobStopped:
            job.status = "stopped"
        except Exception as e:  # noqa: BLE001 — 오류를 브라우저에 그대로 보여 준다
            job.status = "error"
            job.error = f"{type(e).__name__}: {e}"
            traceback.print_exc()
        job.emit("status", {"status": job.status, "error": job.error, "result": job.result})


def _num(v):
    if v in (None, ""):
        return None
    return float(v)


def _int(v):
    v = _num(v)
    return None if v is None else int(v)
