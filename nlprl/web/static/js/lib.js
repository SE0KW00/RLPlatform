// 공통 유틸: DOM 헬퍼, API, 마크다운 렌더러, 수식 포맷터, 파이썬 문법 강조.
// 외부 라이브러리 없이 동작하도록 직접 구현했다 (오프라인 환경에서도 사용 가능).

import { texToMathML } from "./math.js";

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v == null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "html") el.innerHTML = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k === "style" && typeof v === "object") {
      for (const [sk, sv] of Object.entries(v)) {
        if (sk.startsWith("--")) el.style.setProperty(sk, sv); // CSS 사용자 정의 속성은 setProperty 로만 설정됨
        else el.style[sk] = sv;
      }
    }
    else el.setAttribute(k, v === true ? "" : v);
  }
  for (const c of children.flat(Infinity)) {
    if (c == null || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
  return el;
}

export const $ = (sel, root = document) => root.querySelector(sel);

export function esc(s) {
  return String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

export async function api(path, body) {
  const opts = body === undefined ? {} : { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
  const res = await fetch(path, opts);
  const ct = res.headers.get("Content-Type") || "";
  const data = ct.includes("json") ? await res.json() : await res.text();
  if (!res.ok) throw new Error(data.error || data || res.statusText);
  return data;
}

export const fmt = (v, d = 3) => (v == null || Number.isNaN(v) ? "–" : Math.abs(v) >= 100 ? v.toFixed(0) : (+v).toFixed(d));

export function errorBox(e) {
  return h("div", { class: "notice warn" }, h("strong", {}, "불러올 수 없습니다. "), String(e.message || e));
}

export function loading(text = "계산 중…") {
  return h("div", { class: "loading" }, h("span", { class: "spinner" }), text);
}

// ---------------------------------------------------------------------------
// 마크다운 (문서에 쓰이는 문법: 제목, 문단, 목록, 표, 코드 블록, 인용, $$수식$$, 인라인 서식)
// ---------------------------------------------------------------------------
export function inlineMd(text, linkFn = (u) => u) {
  const codes = [];
  let s = text.replace(/`([^`]+)`/g, (_, c) => `\u0000${codes.push(`<code>${esc(c)}</code>`) - 1}\u0000`);
  const maths = [];
  s = s.replace(/\$\$([^$]+)\$\$|\$([^$\n]+)\$/g, (_, a, b) => `\u0001${maths.push(texToMathML(a || b, false)) - 1}\u0001`);
  s = esc(s)
    .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*)\*/g, "$1<em>$2</em>")
    .replace(/\[([^\]]+)\]\(([^)]+)\)/g, (_, t, u) => {
      const href = linkFn(u.replace(/&amp;/g, "&"));
      const ext = /^https?:/.test(href);
      return `<a href="${esc(href)}"${ext ? ' target="_blank" rel="noopener"' : ""}>${t}</a>`;
    });
  return s
    .replace(/\u0001(\d+)\u0001/g, (_, i) => maths[i])
    .replace(/\u0000(\d+)\u0000/g, (_, i) => codes[i]);
}

export function renderMarkdown(md, linkFn) {
  const lines = md.replace(/\r/g, "").split("\n");
  const out = [];
  let i = 0;
  const inl = (t) => inlineMd(t, linkFn);
  while (i < lines.length) {
    const line = lines[i];
    if (/^```/.test(line)) {
      const lang = line.slice(3).trim();
      const buf = [];
      for (i++; i < lines.length && !/^```/.test(lines[i]); i++) buf.push(lines[i]);
      i++;
      const code = buf.join("\n");
      out.push(`<pre class="code-block"><code>${lang === "python" ? highlightPython(code).join("\n") : esc(code)}</code></pre>`);
    } else if (/^\$\$/.test(line.trim())) {
      let tex = line.trim().slice(2);
      if (tex.endsWith("$$") && tex.length > 2) tex = tex.slice(0, -2);
      else {
        const buf = [tex];
        for (i++; i < lines.length && !lines[i].trim().endsWith("$$"); i++) buf.push(lines[i]);
        if (i < lines.length) buf.push(lines[i].trim().slice(0, -2));
        tex = buf.join(" ");
      }
      i++;
      out.push(`<div class="math-block">${texToMathML(tex, true)}</div>`);
    } else if (/^#{1,6} /.test(line)) {
      const n = line.match(/^#+/)[0].length;
      out.push(`<h${n}>${inl(line.slice(n + 1))}</h${n}>`);
      i++;
    } else if (/^\|/.test(line)) {
      const rows = [];
      for (; i < lines.length && /^\|/.test(lines[i]); i++) rows.push(lines[i]);
      const cells = (r) => r.replace(/^\||\|$/g, "").split(/(?<!\\)\|/).map((c) => c.trim());
      const head = cells(rows[0]);
      const body = rows.slice(/^\|[\s:|-]+\|$/.test(rows[1] || "") ? 2 : 1).map(cells);
      out.push(`<div class="table-wrap"><table><thead><tr>${head.map((c) => `<th>${inl(c)}</th>`).join("")}</tr></thead><tbody>${body
        .map((r) => `<tr>${r.map((c) => `<td>${inl(c)}</td>`).join("")}</tr>`)
        .join("")}</tbody></table></div>`);
    } else if (/^\s*([-*]|\d+\.) /.test(line)) {
      const ordered = /^\s*\d+\./.test(line);
      const items = [];
      for (; i < lines.length && (/^\s*([-*]|\d+\.) /.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length)); i++) {
        if (/^\s*([-*]|\d+\.) /.test(lines[i])) items.push({ depth: lines[i].match(/^\s*/)[0].length, text: lines[i].replace(/^\s*([-*]|\d+\.) /, "") });
        else items[items.length - 1].text += " " + lines[i].trim();
      }
      const tag = ordered ? "ol" : "ul";
      out.push(`<${tag}>${items.map((it) => `<li${it.depth ? ' class="nested"' : ""}>${inl(it.text)}</li>`).join("")}</${tag}>`);
    } else if (/^> /.test(line)) {
      const buf = [];
      for (; i < lines.length && /^> ?/.test(lines[i]); i++) buf.push(lines[i].replace(/^> ?/, ""));
      out.push(`<blockquote>${inl(buf.join(" "))}</blockquote>`);
    } else if (line.trim() === "") {
      i++;
    } else {
      const buf = [];
      for (; i < lines.length && lines[i].trim() && !/^(#{1,6} |```|\||>|\s*([-*]|\d+\.) |\$\$)/.test(lines[i]); i++) buf.push(lines[i]);
      out.push(`<p>${inl(buf.join(" "))}</p>`);
    }
  }
  return out.join("\n");
}

// ---------------------------------------------------------------------------
// 파이썬 문법 강조 → 줄 단위 HTML 배열
// ---------------------------------------------------------------------------
const KW = new Set("and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield None True False".split(" "));
const BUILTIN = new Set("print len range enumerate zip sum min max abs float int str list dict set tuple isinstance type super self cls any all sorted map filter getattr setattr open".split(" "));
const TOKEN_RE = /("""[\s\S]*?"""|'''[\s\S]*?'''|[rbfu]{0,2}"(?:\\.|[^"\\\n])*"|[rbfu]{0,2}'(?:\\.|[^'\\\n])*'|#[^\n]*|@[\w.]+|\b\d[\d_]*(?:\.\d+)?(?:e[-+]?\d+)?\b|\b[A-Za-z_]\w*\b|\n|\s+|.)/g;

export function highlightPython(src) {
  const lines = [[]];
  let prev = "";
  for (const m of src.matchAll(TOKEN_RE)) {
    const t = m[0];
    let cls = null;
    if (t === "\n") { lines.push([]); continue; }
    if (/^([rbfu]{0,2})("""|'''|"|')/.test(t)) cls = t.startsWith('"""') || t.startsWith("'''") ? "str doc" : "str";
    else if (t[0] === "#") cls = "com";
    else if (t[0] === "@") cls = "dec";
    else if (/^\d/.test(t)) cls = "num";
    else if (KW.has(t)) cls = "kw";
    else if (BUILTIN.has(t)) cls = "bi";
    else if (prev === "def" || prev === "class") cls = "fn";
    if (!/^\s+$/.test(t)) prev = t;
    // 여러 줄 토큰(독스트링)은 줄마다 span 을 닫고 다시 연다
    const parts = t.split("\n");
    parts.forEach((p, k) => {
      if (k > 0) lines.push([]);
      if (p) lines[lines.length - 1].push(cls ? `<span class="${cls}">${esc(p)}</span>` : esc(p));
    });
  }
  return lines.map((l) => l.join(""));
}

// 슬라이더 + 숫자 표시 컨트롤
export function slider({ label, min, max, step, value, onInput, format = (v) => v }) {
  const out = h("output", {}, format(value));
  const input = h("input", { type: "range", min, max, step, value, "aria-label": label });
  input.addEventListener("input", () => { out.textContent = format(+input.value); onInput(+input.value); });
  return h("label", { class: "control" }, h("span", { class: "control-label" }, label), input, out);
}

export function select({ label, options, value, onChange }) {
  const sel = h("select", { "aria-label": label }, options.map((o) => {
    const [v, t] = Array.isArray(o) ? o : [o, o];
    return h("option", { value: v, selected: v === value }, t);
  }));
  sel.addEventListener("change", () => onChange(sel.value));
  return h("label", { class: "control" }, h("span", { class: "control-label" }, label), sel);
}

export function toggle({ label, value, onChange }) {
  const cb = h("input", { type: "checkbox", checked: value });
  cb.addEventListener("change", () => onChange(cb.checked));
  return h("label", { class: "control toggle" }, cb, h("span", {}, label));
}
