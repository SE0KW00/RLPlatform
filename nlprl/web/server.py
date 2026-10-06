"""NLP RL 학습 웹 앱 서버 (표준 라이브러리만 사용).

    python -m nlprl serve            # http://127.0.0.1:8000

API
  GET  /api/lessons                 레슨 목록
  GET  /api/lessons/<id>            레슨 상세 (워크스루 줄 번호 포함)
  GET  /api/doc?path=docs/...md     문서 원문 (markdown)
  GET  /api/source?path=nlprl/...   소스 코드 원문
  GET  /api/status                  체크포인트·실행 기록·작업 목록
  GET  /api/runs/<task>/<name>      history.json
  POST /api/trace                   토큰별 생성 추적
  POST /api/rollout                 배치 롤아웃 (+그룹 advantage, 텐서 레이아웃)
  POST /api/rm_score                규칙 보상 vs 보상 모델 점수
  POST /api/dpo_pairs               선호 쌍과 DPO 암묵적 보상
  POST /api/jobs                    학습 작업 시작 {kind, params}
  POST /api/jobs/<id>/stop          작업 중단
  GET  /api/jobs/<id>/events        작업 이벤트 스트림 (Server-Sent Events)
"""

from __future__ import annotations

import json
import mimetypes
import re
import threading
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from .. import pipeline
from . import lessons, probe
from .jobs import JobManager

STATIC = Path(__file__).resolve().parent / "static"
ROOT = lessons.ROOT
READABLE = {"nlprl": ".py", "docs": ".md"}  # 브라우저에 공개하는 디렉터리와 확장자

# torch 연산은 스레드 안전하지만, 시각화 요청이 학습과 겹치면 느려지므로 하나씩 처리한다
_probe_lock = threading.Lock()


def safe_path(rel: str) -> Path:
    p = (ROOT / rel).resolve()
    top = p.relative_to(ROOT).parts[0] if p.is_relative_to(ROOT) else None
    if top not in READABLE or p.suffix != READABLE[top] or not p.is_file():
        raise PermissionError(f"읽을 수 없는 경로입니다: {rel}")
    return p


def status(jobs: JobManager) -> dict:
    return {
        "checkpoints": probe.list_checkpoints(),
        "runs": probe.list_runs(),
        "reward_models": [t for t in ("sentiment", "arithmetic") if pipeline.rm_path(t).exists()],
        "pairs": [t for t in ("sentiment", "arithmetic") if pipeline.pairs_path(t).exists()],
        "jobs": [j.summary() for j in sorted(jobs.jobs.values(), key=lambda j: -j.id)],
        "defaults": pipeline.DEFAULTS,
    }


def make_handler(jobs: JobManager):
    class Handler(BaseHTTPRequestHandler):
        server_version = "nlprl"

        def log_message(self, fmt, *args):  # 요청 로그는 조용히
            pass

        # ---------------- 응답 헬퍼 ----------------
        def send_json(self, data, code=200):
            body = json.dumps(data, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def send_text(self, text: str, ctype="text/plain"):
            body = text.encode()
            self.send_response(200)
            self.send_header("Content-Type", f"{ctype}; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def body_json(self) -> dict:
            n = int(self.headers.get("Content-Length") or 0)
            return json.loads(self.rfile.read(n) or b"{}")

        def fail(self, e: Exception):
            code = {FileNotFoundError: 404, PermissionError: 403, KeyError: 404, StopIteration: 404,
                    ValueError: 400}.get(type(e), 500)
            if code == 500:
                traceback.print_exc()
            self.send_json({"error": f"{e}"}, code)

        # ---------------- 라우팅 ----------------
        def do_GET(self):
            url = urlparse(self.path)
            q = {k: v[0] for k, v in parse_qs(url.query).items()}
            path = url.path
            try:
                if path == "/api/lessons":
                    return self.send_json(lessons.lessons_index())
                if m := re.fullmatch(r"/api/lessons/([\w-]+)", path):
                    return self.send_json(lessons.lesson(m[1]))
                if path in ("/api/doc", "/api/source"):
                    return self.send_text(safe_path(q["path"]).read_text(encoding="utf-8"))
                if path == "/api/status":
                    return self.send_json(status(jobs))
                if m := re.fullmatch(r"/api/runs/(\w+)/([\w.-]+)", path):
                    f = pipeline.run_dir(m[1], m[2]) / "history.json"
                    return self.send_json(json.loads(f.read_text()))
                if m := re.fullmatch(r"/api/jobs/(\d+)/events", path):
                    return self.stream_events(jobs.jobs[int(m[1])], int(q.get("from", 0)))
                return self.serve_static(path)
            except Exception as e:  # noqa: BLE001
                self.fail(e)

        def do_POST(self):
            path = urlparse(self.path).path
            try:
                b = self.body_json()
                if path in ("/api/trace", "/api/rollout", "/api/rm_score", "/api/dpo_pairs"):
                    fn = {"/api/trace": probe.trace, "/api/rollout": probe.rollout,
                          "/api/rm_score": probe.rm_score, "/api/dpo_pairs": probe.dpo_pairs}[path]
                    with _probe_lock:
                        return self.send_json(fn(**b))
                if path == "/api/jobs":
                    return self.send_json(jobs.submit(b["kind"], b.get("params", {})).summary())
                if m := re.fullmatch(r"/api/jobs/(\d+)/stop", path):
                    jobs.stop(int(m[1]))
                    return self.send_json({"ok": True})
                self.send_json({"error": "not found"}, 404)
            except TypeError as e:  # 잘못된 인자
                self.send_json({"error": f"{e}"}, 400)
            except Exception as e:  # noqa: BLE001
                self.fail(e)

        def serve_static(self, path: str):
            if path == "/" or not Path(path).suffix:
                path = "/index.html"  # SPA: 해시 라우팅이지만 새로고침에도 안전하게
            f = (STATIC / path.lstrip("/")).resolve()
            if not f.is_relative_to(STATIC) or not f.is_file():
                return self.send_json({"error": "not found"}, 404)
            ctype = mimetypes.guess_type(f.name)[0] or "application/octet-stream"
            if f.suffix == ".js":
                ctype = "text/javascript"
            body = f.read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", ctype + ("; charset=utf-8" if ctype.startswith("text") else ""))
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            self.wfile.write(body)

        def stream_events(self, job, start: int):
            """Server-Sent Events: 작업이 끝날 때까지 새 이벤트를 밀어 준다."""
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            i = start
            while True:
                with job.cond:
                    if i >= len(job.events):
                        job.cond.wait(timeout=15)
                    new = job.events[i:]
                try:
                    if not new:
                        self.wfile.write(b": keep-alive\n\n")
                    for ev in new:
                        self.wfile.write(f"data: {json.dumps(ev, ensure_ascii=False)}\n\n".encode())
                    self.wfile.flush()
                except (BrokenPipeError, ConnectionResetError):
                    return
                i += len(new)
                if job.status in ("done", "error", "stopped") and i >= len(job.events):
                    return

    return Handler


def serve(host: str = "127.0.0.1", port: int = 8000, open_browser: bool = False):
    jobs = JobManager()
    httpd = ThreadingHTTPServer((host, port), make_handler(jobs))
    httpd.daemon_threads = True
    url = f"http://{host}:{httpd.server_address[1]}"
    print(f"NLP RL 학습 플랫폼 → {url}   (종료: Ctrl+C)")
    if open_browser:
        import webbrowser
        webbrowser.open(url)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return httpd
