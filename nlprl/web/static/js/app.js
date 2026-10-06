// 앱 진입점: 해시 라우터, 사이드바, 홈·레슨·소스 페이지.

import { h, $, api, errorBox, loading, renderMarkdown, highlightPython } from "./lib.js";
import { WIDGETS } from "./widgets.js";
import { diagram, DIAGRAMS } from "./diagrams.js";
import { labPage, leaveLab, comparePage } from "./lab.js";

const ctx = { status: null, lessons: [] };
const main = $("#main");
const nav = $("#nav");
const sourceCache = new Map();

// ---------------------------------------------------------------------------
// 링크 변환: 문서 안의 상대 링크를 앱 경로로
// ---------------------------------------------------------------------------
function docLink(url) {
  if (/^https?:/.test(url) || url.startsWith("#")) return url;
  const file = url.split("/").pop();
  const lesson = ctx.lessons.find((l) => l.doc?.endsWith(file));
  if (lesson) return `#/lesson/${lesson.id}/doc`;
  const m = url.match(/(nlprl\/.*\.py)$/) || url.match(/^\.\.\/(nlprl\/.*\.py)$/);
  if (m) return `#/source?path=${encodeURIComponent(m[1])}`;
  return url;
}

// ---------------------------------------------------------------------------
// 사이드바
// ---------------------------------------------------------------------------
function drawNav(active) {
  const item = (href, num, title, key) => h("a", { href, class: `nav-item${active === key ? " active" : ""}`, "aria-current": active === key ? "page" : null },
    num ? h("span", { class: "nav-num" }, num) : null, h("span", {}, title));
  nav.replaceChildren(
    h("div", { class: "nav-section" }, "시작"),
    item("#/", "", "홈", "home"),
    h("div", { class: "nav-section" }, "레슨"),
    ...ctx.lessons.map((l) => item(`#/lesson/${l.id}`, l.num, l.title, `lesson:${l.id}`)),
    h("div", { class: "nav-section" }, "실습"),
    item("#/lab", "▶", "실험실", "lab"),
    item("#/compare", "≋", "실행 비교", "compare"),
    item("#/source", "{}", "소스 탐색기", "source"),
  );
}

// ---------------------------------------------------------------------------
// 홈
// ---------------------------------------------------------------------------
function homePage() {
  const cards = ctx.lessons.map((l) => h("a", { class: "lesson-card", href: `#/lesson/${l.id}` },
    h("span", { class: "lesson-num" }, l.num), h("strong", {}, l.title), h("span", { class: "muted" }, l.subtitle)));
  main.replaceChildren(
    h("div", { class: "hero" },
      h("h1", {}, "NLP 강화학습을 코드와 그림으로"),
      h("p", { class: "lead" }, "REINFORCE · PPO-RLHF · 보상 모델 · DPO · GRPO 를 작은 GPT 로 직접 돌려 보며 배웁니다. 각 레슨은 ",
        h("strong", {}, "동작 구조 시각화"), ", ", h("strong", {}, "실제 소스 코드 워크스루"), ", ", h("strong", {}, "개념 문서"), " 로 구성됩니다."),
      h("div", { class: "row" }, h("a", { class: "btn primary", href: "#/lesson/overview" }, "레슨 00 부터 시작 →"), h("a", { class: "btn", href: "#/lab" }, "실험실 열기"))),
    WIDGETS.status(ctx),
    h("section", { class: "card" }, h("h3", {}, "전체 파이프라인"), diagram(DIAGRAMS.pipeline)),
    h("h2", { class: "section-title" }, "레슨"),
    h("div", { class: "lesson-grid" }, cards),
  );
}

// ---------------------------------------------------------------------------
// 레슨
// ---------------------------------------------------------------------------
const TABS = [["visual", "구조 & 시각화"], ["code", "코드 워크스루"], ["doc", "개념 문서"]];

async function lessonPage(id, tab = "visual", stepArg) {
  const lesson = await api(`/api/lessons/${id}`);
  const available = TABS.filter(([k]) => (k === "visual" ? lesson.widgets.length : k === "code" ? lesson.walkthrough.length : true));
  if (!available.some(([k]) => k === tab)) tab = available[0][0];
  const idx = ctx.lessons.findIndex((l) => l.id === id);
  const prev = ctx.lessons[idx - 1], next = ctx.lessons[idx + 1];
  const body = h("div", { class: "lesson-body" });
  main.replaceChildren(
    h("div", { class: "page-head" },
      h("div", { class: "eyebrow" }, `레슨 ${lesson.num}`),
      h("h1", {}, lesson.title),
      h("p", { class: "lead" }, lesson.subtitle)),
    h("nav", { class: "tabs", role: "tablist" }, available.map(([k, t]) =>
      h("a", { href: `#/lesson/${id}/${k}`, class: k === tab ? "active" : "", role: "tab", "aria-selected": k === tab }, t))),
    body,
    h("div", { class: "pager" },
      prev ? h("a", { class: "btn ghost", href: `#/lesson/${prev.id}` }, `← ${prev.num} ${prev.title}`) : h("span"),
      next ? h("a", { class: "btn ghost", href: `#/lesson/${next.id}` }, `${next.num} ${next.title} →`) : h("span")),
  );
  if (tab === "visual") {
    for (const w of lesson.widgets) {
      try { body.append(WIDGETS[w](ctx)); } catch (e) { body.append(errorBox(e)); console.error(e); }
    }
  } else if (tab === "code") {
    body.append(await walkthrough(lesson, +stepArg || 0));
  } else {
    const md = await api(`/api/doc?path=${encodeURIComponent(lesson.doc)}`);
    body.append(h("article", { class: "card prose", html: renderMarkdown(md, docLink) }));
  }
}

async function getSource(path) {
  if (!sourceCache.has(path)) sourceCache.set(path, api(`/api/source?path=${encodeURIComponent(path)}`));
  return sourceCache.get(path);
}

function codeView(path, src, { from, to } = {}) {
  const lines = highlightPython(src);
  const pre = h("div", { class: "code-view", role: "region", "aria-label": path });
  lines.forEach((html, i) => {
    const n = i + 1;
    const hl = from && n >= from && n <= to;
    pre.append(h("div", { class: `code-line${hl ? " hl" : ""}${hl && n === from ? " hl-start" : ""}`, id: `L${n}` },
      h("span", { class: "ln" }, n), h("span", { class: "lc", html: html || " " })));
  });
  return pre;
}

async function walkthrough(lesson, start) {
  const steps = lesson.walkthrough;
  const wrap = h("div", { class: "walkthrough" });
  const stepList = h("ol", { class: "step-list" });
  const explain = h("div", { class: "step-explain" });
  const codeBox = h("div", { class: "code-pane" });
  let cur = -1;

  async function show(i) {
    i = Math.max(0, Math.min(steps.length - 1, i));
    const st = steps[i];
    const sameFile = cur >= 0 && steps[cur].file === st.file;
    cur = i;
    history.replaceState(null, "", `#/lesson/${lesson.id}/code/${i}`);
    stepList.querySelectorAll("li").forEach((li, k) => li.classList.toggle("active", k === i));
    explain.replaceChildren(
      h("div", { class: "step-meta" }, h("span", { class: "step-count" }, `${i + 1} / ${steps.length}`),
        h("a", { href: `#/source?path=${encodeURIComponent(st.file)}&from=${st.line_start}&to=${st.line_end}`, class: "file-link" }, `${st.file}:${st.line_start}–${st.line_end}`)),
      h("h3", {}, st.title),
      h("div", { class: "prose", html: renderMarkdown(st.body, docLink) }),
      h("div", { class: "row" },
        h("button", { class: "btn", disabled: i === 0, onclick: () => show(i - 1) }, "← 이전"),
        h("button", { class: "btn primary", disabled: i === steps.length - 1, onclick: () => show(i + 1) }, "다음 →")),
      h("p", { class: "hint" }, "키보드 ← → 로도 이동할 수 있습니다."));
    let view = codeBox.querySelector(".code-view");
    if (!sameFile || !view) {
      codeBox.replaceChildren(h("div", { class: "code-file" }, st.file), loading("코드 불러오는 중…"));
      const src = await getSource(st.file);
      view = codeView(st.file, src, { from: st.line_start, to: st.line_end });
      codeBox.replaceChildren(h("div", { class: "code-file" }, st.file), view);
    } else {
      view.querySelectorAll(".code-line").forEach((el, k) => {
        const n = k + 1;
        el.classList.toggle("hl", n >= st.line_start && n <= st.line_end);
        el.classList.toggle("hl-start", n === st.line_start);
      });
    }
    const target = view.querySelector(`#L${Math.max(1, st.line_start - 3)}`);
    if (target) view.scrollTo({ top: target.offsetTop - view.offsetTop, behavior: sameFile ? "smooth" : "auto" });
  }

  steps.forEach((st, i) => stepList.append(h("li", { onclick: () => show(i) }, h("span", { class: "step-title" }, st.title), h("span", { class: "step-file" }, st.file.replace("nlprl/", "")))));
  const onKey = (e) => {
    if (!wrap.isConnected) return document.removeEventListener("keydown", onKey);
    if (["INPUT", "SELECT", "TEXTAREA"].includes(document.activeElement?.tagName)) return;
    if (e.key === "ArrowRight") show(cur + 1);
    if (e.key === "ArrowLeft") show(cur - 1);
  };
  document.addEventListener("keydown", onKey);
  wrap.append(h("aside", { class: "walk-side" }, h("div", { class: "card" }, stepList), h("div", { class: "card" }, explain)), h("div", { class: "card code-card" }, codeBox));
  setTimeout(() => show(start));
  return wrap;
}

// ---------------------------------------------------------------------------
// 소스 탐색기
// ---------------------------------------------------------------------------
const SOURCE_FILES = [
  "nlprl/tokenizer.py", "nlprl/model.py", "nlprl/tasks.py", "nlprl/generation.py", "nlprl/pretrain.py",
  "nlprl/preference.py", "nlprl/algorithms/common.py", "nlprl/algorithms/reinforce.py", "nlprl/algorithms/ppo.py",
  "nlprl/algorithms/dpo.py", "nlprl/algorithms/grpo.py", "nlprl/pipeline.py", "nlprl/utils.py", "nlprl/cli.py",
];

async function sourcePage(params) {
  const path = params.get("path") || "nlprl/algorithms/ppo.py";
  const from = +params.get("from") || null, to = +params.get("to") || from;
  const list = h("nav", { class: "file-list" }, SOURCE_FILES.map((f) => h("a", { href: `#/source?path=${encodeURIComponent(f)}`, class: f === path ? "active" : "" }, f.replace("nlprl/", ""))));
  const box = h("div", { class: "card code-card" }, loading());
  main.replaceChildren(h("div", { class: "page-head" }, h("h1", {}, "소스 탐색기"), h("p", { class: "lead" }, "플랫폼의 모든 코드를 브라우저에서 읽을 수 있습니다. 각 레슨의 ‘코드 워크스루’는 이 파일들의 구간을 짚어 가며 설명합니다.")),
    h("div", { class: "source-layout" }, h("div", { class: "card" }, list), box));
  const src = await getSource(path);
  const view = codeView(path, src, { from, to });
  box.replaceChildren(h("div", { class: "code-file" }, path), view);
  if (from) setTimeout(() => { const t = view.querySelector(`#L${Math.max(1, from - 3)}`); if (t) view.scrollTop = t.offsetTop - view.offsetTop; });
}

// ---------------------------------------------------------------------------
// 라우터
// ---------------------------------------------------------------------------
async function route() {
  const hash = location.hash.slice(1) || "/";
  const [path, query] = hash.split("?");
  const parts = path.split("/").filter(Boolean);
  const params = new URLSearchParams(query || "");
  leaveLab();
  try {
    ctx.status = await api("/api/status");
    if (!parts.length) { drawNav("home"); homePage(); }
    else if (parts[0] === "lesson") { drawNav(`lesson:${parts[1]}`); await lessonPage(parts[1], parts[2], parts[3]); }
    else if (parts[0] === "lab") { drawNav("lab"); await labPage(ctx, main); }
    else if (parts[0] === "compare") { drawNav("compare"); await comparePage(ctx, main); }
    else if (parts[0] === "source") { drawNav("source"); await sourcePage(params); }
    else { drawNav(""); main.replaceChildren(h("div", { class: "notice" }, "페이지를 찾을 수 없습니다.")); }
  } catch (e) {
    main.replaceChildren(errorBox(e));
    console.error(e);
  }
  if (!/^\/lesson\/[\w-]+\/code/.test(path)) main.scrollTo?.(0, 0);
  document.body.classList.remove("nav-open");
}

// 테마 전환 (시스템 설정 → 라이트 → 다크)
const svgIcon = (body) => `<svg viewBox="0 0 24 24" width="18" height="18" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round" aria-hidden="true">${body}</svg>`;
const THEME_ICONS = {
  auto: svgIcon('<circle cx="12" cy="12" r="9"/><path d="M12 3v18a9 9 0 0 0 0-18z" fill="currentColor"/>'),
  light: svgIcon('<circle cx="12" cy="12" r="4"/><path d="M12 2v2M12 20v2M4.9 4.9l1.4 1.4M17.7 17.7l1.4 1.4M2 12h2M20 12h2M4.9 19.1l1.4-1.4M17.7 6.3l1.4-1.4"/>'),
  dark: svgIcon('<path d="M21 12.8A9 9 0 1 1 11.2 3a7 7 0 0 0 9.8 9.8z"/>'),
};
function setupTheme() {
  const btn = $("#theme");
  const apply = (t) => {
    if (t) document.documentElement.dataset.theme = t; else delete document.documentElement.dataset.theme;
    btn.innerHTML = THEME_ICONS[t || "auto"];
    btn.title = `테마: ${t === "dark" ? "다크" : t === "light" ? "라이트" : "시스템"}`;
  };
  let t = null;
  try { t = localStorage.getItem("theme"); } catch { /* 저장소 사용 불가 */ }
  apply(t);
  btn.addEventListener("click", () => {
    t = t === null ? "light" : t === "light" ? "dark" : null;
    try { t ? localStorage.setItem("theme", t) : localStorage.removeItem("theme"); } catch { /* 무시 */ }
    apply(t);
  });
  $("#menu").addEventListener("click", () => document.body.classList.toggle("nav-open"));
}

(async function init() {
  setupTheme();
  try {
    ctx.lessons = await api("/api/lessons");
  } catch (e) {
    main.replaceChildren(errorBox(e));
    return;
  }
  window.addEventListener("hashchange", route);
  route();
})();

