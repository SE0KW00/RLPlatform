"""웹 앱 테스트: 레슨 앵커, 정적 파일, API, 학습 작업 스트리밍."""

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer

import pytest

from nlprl import pipeline
from nlprl.web import lessons
from nlprl.web.jobs import JobManager
from nlprl.web.server import make_handler


def test_all_walkthrough_anchors_resolve():
    for l in lessons.LESSONS:
        for st in lessons.lesson(l["id"])["walkthrough"]:
            assert 1 <= st["line_start"] <= st["line_end"], st
            assert st["line_end"] - st["line_start"] < 60, f"구간이 너무 깁니다: {st['title']}"
        assert (lessons.ROOT / l["doc"]).exists()


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    root = tmp_path_factory.mktemp("runs")
    old = pipeline.RUNS_ROOT
    pipeline.RUNS_ROOT = root
    pipeline.run_pretrain("arithmetic", steps=30, log_every=1000)
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(JobManager()))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()
    pipeline.RUNS_ROOT = old


def get(url):
    with urllib.request.urlopen(url, timeout=60) as r:
        body = r.read().decode()
        return json.loads(body) if "json" in r.headers["Content-Type"] else body


def post(url, data):
    req = urllib.request.Request(url, json.dumps(data).encode(), {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read())


def test_static_and_docs(server):
    assert "<title>" in get(server + "/")
    assert "export function h" in get(server + "/js/lib.js")
    assert get(server + "/api/doc?path=docs/02_reinforce.md").startswith("# 02.")
    assert "def compute_gae" in get(server + "/api/source?path=nlprl/algorithms/ppo.py")
    for bad in ("../README.md", "nlprl/../runs/x.py", "/etc/passwd"):
        with pytest.raises(urllib.error.HTTPError) as e:
            get(server + f"/api/source?path={bad}")
        assert e.value.code == 403


def test_probe_endpoints(server):
    st = get(server + "/api/status")
    ckpt = next(c["path"] for c in st["checkpoints"] if c["name"] == "sft")
    tr = post(server + "/api/trace", {"ckpt": ckpt, "prompt": "3+4="})
    assert tr["steps"] and "top" in tr["steps"][0] and tr["has_ref"]
    ro = post(server + "/api/rollout", {"ckpt": ckpt, "prompts": ["3+4="], "group_size": 4, "with_tensors": True})
    assert len(ro["samples"]) == 4 and all("advantage" in s for s in ro["samples"])
    t = ro["tensors"]
    assert len(t["logprobs"][0]) == len(t["tokens"][0]) - 1
    with pytest.raises(urllib.error.HTTPError) as e:
        post(server + "/api/rm_score", {"task": "arithmetic", "items": []})
    assert e.value.code == 404  # 보상 모델 없음 → 안내 메시지


def test_job_streams_events(server):
    job = post(server + "/api/jobs", {"kind": "train", "params": {"task": "arithmetic", "algo": "grpo", "steps": "5", "name": "t"}})
    events = []
    with urllib.request.urlopen(f"{server}/api/jobs/{job['id']}/events", timeout=120) as r:
        for line in r:
            if line.startswith(b"data: "):
                events.append(json.loads(line[6:]))
    assert events[-1]["status"] == "done", events[-1]
    assert any(e["type"] == "log" for e in events)
    assert (pipeline.run_dir("arithmetic", "t") / "policy.pt").exists()
