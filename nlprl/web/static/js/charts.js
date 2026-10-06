// 가벼운 SVG 차트: 선 그래프(실시간 갱신, 십자선 툴팁)와 막대 그래프(부호 있는 값 지원).
// 색은 CSS 변수(--s1..--s8, 검증된 범주형 팔레트)를 순서대로 쓴다.

import { h, fmt } from "./lib.js";

const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}, ...kids) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
};

export const seriesColor = (i) => `var(--s${(i % 8) + 1})`;

function niceTicks(lo, hi, n = 5) {
  if (lo === hi) { lo -= 1; hi += 1; }
  const span = hi - lo;
  const step0 = span / n;
  const mag = 10 ** Math.floor(Math.log10(step0));
  const step = [1, 2, 2.5, 5, 10].map((m) => m * mag).find((st) => span / st <= n) || 10 * mag;
  const start = Math.floor(lo / step) * step;
  const ticks = [];
  for (let v = start; v <= hi + step * 0.5; v += step) ticks.push(+v.toFixed(10));
  return ticks;
}

function tooltip(container) {
  const tip = h("div", { class: "chart-tip", role: "status" });
  container.append(tip);
  return {
    show(html, x, y) {
      tip.innerHTML = html;
      tip.style.display = "block";
      const cw = container.clientWidth;
      const tw = tip.offsetWidth;
      tip.style.left = `${Math.min(Math.max(x + 12, 0), cw - tw - 4)}px`;
      tip.style.top = `${Math.max(y - 10, 0)}px`;
    },
    hide() { tip.style.display = "none"; },
  };
}

function responsive(root, draw) {
  let last = 0;
  const ro = new ResizeObserver(() => {
    const w = root.clientWidth;
    if (w && Math.abs(w - last) > 4) { last = w; draw(w); }
  });
  ro.observe(root);
  return (force) => draw(root.clientWidth || last || 600, force);
}

/**
 * 선 그래프.
 * opts: { series: [{name, points: [[x,y],...], dashed?}], xLabel, yLabel, height, yMin, yMax, refLines: [{y,label}],
 *         vLines: [{x,label}], xFormat, yFormat, legend }
 * 반환된 element.update(series) 로 갱신.
 */
export function lineChart(opts) {
  const root = h("div", { class: "chart" });
  const tip = tooltip(root);
  let series = opts.series || [];
  const H = opts.height || 220;
  const M = { l: 48, r: 16, t: opts.yLabel ? 26 : 12, b: 34 };
  const xf = opts.xFormat || ((v) => (Number.isInteger(v) ? String(v) : fmt(v, 1)));
  const yf = opts.yFormat || ((v) => fmt(v, 2));
  let svg;

  function draw(W) {
    svg?.remove();
    const all = series.flatMap((sr) => sr.points);
    const xs = all.map((p) => p[0]);
    const ys = all.map((p) => p[1]).concat((opts.refLines || []).map((r) => r.y));
    let x0 = opts.xMin ?? (xs.length ? Math.min(...xs) : 0);
    let x1 = opts.xMax ?? (xs.length ? Math.max(...xs) : 1);
    if (x0 === x1) x1 = x0 + 1;
    let y0 = opts.yMin ?? (ys.length ? Math.min(...ys) : 0);
    let y1 = opts.yMax ?? (ys.length ? Math.max(...ys) : 1);
    if (opts.yMin == null && opts.yMax == null) { const pad = (y1 - y0 || 1) * 0.08; y0 -= pad; y1 += pad; }
    const yt = niceTicks(y0, y1, 4).filter((t) => t >= y0 - 1e-9 && t <= y1 + 1e-9);
    let xt = niceTicks(x0, x1, Math.max(2, Math.floor(W / 90))).filter((t) => t >= x0 - 1e-9 && t <= x1 + 1e-9);
    if (opts.integerX !== false && Number.isInteger(x0) && Number.isInteger(x1) && x1 - x0 <= 12) xt = xt.filter(Number.isInteger);
    const X = (v) => M.l + ((v - x0) / (x1 - x0)) * (W - M.l - M.r);
    const Y = (v) => H - M.b - ((v - y0) / (y1 - y0 || 1)) * (H - M.t - M.b);

    svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.ariaLabel || opts.yLabel || "chart" });
    const g = s("g");
    for (const t of yt) {
      g.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), class: "grid" }));
      g.append(s("text", { x: M.l - 6, y: Y(t) + 4, class: "tick", "text-anchor": "end" }, yf(t)));
    }
    for (const t of xt) g.append(s("text", { x: X(t), y: H - M.b + 16, class: "tick", "text-anchor": "middle" }, xf(t)));
    g.append(s("line", { x1: M.l, x2: W - M.r, y1: H - M.b, y2: H - M.b, class: "axis" }));
    if (opts.xLabel) g.append(s("text", { x: W - M.r, y: H - 4, class: "axis-label", "text-anchor": "end" }, opts.xLabel));
    if (opts.yLabel) g.append(s("text", { x: 4, y: 11, class: "axis-label" }, opts.yLabel));
    for (const r of opts.refLines || []) {
      g.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(r.y), y2: Y(r.y), class: "ref" }));
      if (r.label) g.append(s("text", { x: W - M.r - 4, y: Y(r.y) - 4, class: "ref-label", "text-anchor": "end" }, r.label));
    }
    for (const r of opts.vLines || []) {
      g.append(s("line", { x1: X(r.x), x2: X(r.x), y1: M.t, y2: H - M.b, class: "ref" }));
      if (r.label) g.append(s("text", { x: X(r.x) + 4, y: M.t + 10, class: "ref-label" }, r.label));
    }
    for (const b of opts.bands || []) {
      g.append(s("rect", { x: X(b.x0), width: Math.max(0, X(b.x1) - X(b.x0)), y: M.t, height: H - M.t - M.b, class: "band" }));
      if (b.label) g.append(s("text", { x: (X(b.x0) + X(b.x1)) / 2, y: M.t + 12, class: "ref-label", "text-anchor": "middle" }, b.label));
    }
    series.forEach((sr, i) => {
      const color = sr.color || seriesColor(sr.colorIndex ?? i);
      const pts = sr.points.filter((p) => Number.isFinite(p[1]));
      if (!pts.length) return;
      const d = pts.map((p, k) => `${k ? "L" : "M"}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join("");
      g.append(s("path", { d, fill: "none", stroke: color, "stroke-width": 2, "stroke-linejoin": "round", "stroke-linecap": "round", "stroke-dasharray": sr.dashed ? "5 4" : null }));
      if (pts.length <= 40 && !opts.noDots) for (const p of pts) g.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 3, fill: color, class: "dot" }));
      else { const p = pts[pts.length - 1]; g.append(s("circle", { cx: X(p[0]), cy: Y(p[1]), r: 4, fill: color, class: "dot" })); }
    });
    // 십자선 + 툴팁
    const cross = s("line", { y1: M.t, y2: H - M.b, class: "crosshair", visibility: "hidden" });
    g.append(cross);
    const hit = s("rect", { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: "transparent" });
    hit.addEventListener("pointermove", (e) => {
      const rect = svg.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const xv = x0 + ((px - M.l) / (W - M.l - M.r)) * (x1 - x0);
      let best = null;
      for (const sr of series) for (const p of sr.points) if (!best || Math.abs(p[0] - xv) < Math.abs(best - xv)) best = p[0];
      if (best == null) return;
      cross.setAttribute("x1", X(best)); cross.setAttribute("x2", X(best)); cross.setAttribute("visibility", "visible");
      const rows = series.map((sr, i) => {
        const p = sr.points.find((q) => q[0] === best);
        return p ? `<div><i style="background:${sr.color || seriesColor(sr.colorIndex ?? i)}"></i>${sr.name}<b>${yf(p[1])}</b></div>` : "";
      }).join("");
      tip.show(`<div class="tip-title">${opts.xLabel || "x"} ${xf(best)}</div>${rows}`, X(best), e.clientY - rect.top);
    });
    hit.addEventListener("pointerleave", () => { cross.setAttribute("visibility", "hidden"); tip.hide(); });
    g.append(hit);
    svg.append(g);
    root.prepend(svg);
  }

  const redraw = responsive(root, draw);
  const legend = h("div", { class: "legend" });
  root.append(legend);
  function drawLegend() {
    legend.replaceChildren(...(series.length >= 2 || opts.legend ? series.map((sr, i) =>
      h("span", { class: "legend-item" }, h("i", { class: sr.dashed ? "dashed" : "", style: { background: sr.color || seriesColor(sr.colorIndex ?? i) } }), sr.name)) : []));
  }
  drawLegend();
  root.update = (next) => { series = next; drawLegend(); redraw(); };
  return root;
}

/**
 * 막대 그래프 (0 기준선, 음수 지원).
 * opts: { bars: [{label, value, color?, tip?}], height, yLabel, yMin, yMax, valueFormat }
 */
export function barChart(opts) {
  const root = h("div", { class: "chart" });
  const tip = tooltip(root);
  const H = opts.height || 200;
  const M = { l: 44, r: 10, t: opts.yLabel ? 26 : 16, b: opts.rotateLabels ? 60 : 30 };
  const vf = opts.valueFormat || ((v) => fmt(v, 2));
  let bars = opts.bars;
  let svg;
  function draw(W) {
    svg?.remove();
    const vals = bars.map((b) => b.value);
    let y0 = Math.min(opts.yMin ?? 0, ...vals, 0);
    let y1 = Math.max(opts.yMax ?? 0, ...vals, 0) || 1;
    if (opts.showValues) { // 값 라벨이 들어갈 여유 공간
      const span = y1 - y0 || 1;
      if (y0 < 0) y0 -= span * 0.16;
      if (y1 > 0 && opts.yMax == null) y1 += span * 0.12;
    }
    const Y = (v) => H - M.b - ((v - y0) / (y1 - y0 || 1)) * (H - M.t - M.b);
    const n = bars.length;
    const slot = (W - M.l - M.r) / Math.max(n, 1);
    const bw = Math.max(4, Math.min(42, slot - 4));
    svg = s("svg", { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": opts.yLabel || "bar chart" });
    for (const t of niceTicks(y0, y1, 4).filter((t) => t >= y0 - 1e-9 && t <= y1 + 1e-9)) {
      svg.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(t), y2: Y(t), class: "grid" }));
      svg.append(s("text", { x: M.l - 6, y: Y(t) + 4, class: "tick", "text-anchor": "end" }, vf(t)));
    }
    if (opts.yLabel) svg.append(s("text", { x: 4, y: 10, class: "axis-label" }, opts.yLabel));
    bars.forEach((b, i) => {
      const cx = M.l + slot * (i + 0.5);
      const top = Y(Math.max(b.value, 0));
      const bot = Y(Math.min(b.value, 0));
      const hgt = Math.max(bot - top, b.value === 0 ? 0 : 1);
      // 데이터 끝(0 에서 먼 쪽)만 둥글게: 둥근 사각형 + 기준선 쪽 덮개
      const color = b.color || seriesColor(0);
      const r = Math.min(4, hgt / 2, bw / 2);
      svg.append(s("rect", { x: cx - bw / 2, y: top, width: bw, height: hgt, rx: r, fill: color }));
      if (hgt > r) svg.append(s("rect", { x: cx - bw / 2, y: b.value >= 0 ? bot - r : top, width: bw, height: r, fill: color }));
      if (opts.showValues) svg.append(s("text", { x: cx, y: b.value >= 0 ? top - 4 : bot + 12, class: "bar-value", "text-anchor": "middle" }, vf(b.value)));
      const lx = cx, ly = H - M.b + 14;
      svg.append(s("text", { x: lx, y: ly, class: "tick", "text-anchor": opts.rotateLabels ? "end" : "middle", transform: opts.rotateLabels ? `rotate(-35 ${lx} ${ly})` : null }, b.label));
      const hit = s("rect", { x: cx - slot / 2, y: M.t, width: slot, height: H - M.t - M.b, fill: "transparent" });
      hit.addEventListener("pointermove", (e) => {
        const rect = svg.getBoundingClientRect();
        tip.show(`<div class="tip-title">${b.tipTitle || b.label}</div><div>${b.tip || ""}<b>${vf(b.value)}</b></div>`, cx, e.clientY - rect.top);
      });
      hit.addEventListener("pointerleave", () => tip.hide());
      svg.append(hit);
    });
    svg.append(s("line", { x1: M.l, x2: W - M.r, y1: Y(0), y2: Y(0), class: "axis" }));
    root.prepend(svg);
  }
  const redraw = responsive(root, draw);
  root.update = (next) => { bars = next; redraw(); };
  return root;
}
