// 레슨 시각화 위젯. 각 위젯은 (ctx) => HTMLElement.
//  - 실제 모델 위젯: 서버 API 로 체크포인트를 실행해 결과를 그린다.
//  - 시뮬레이터 위젯: 브라우저 안에서 수식을 직접 계산한다 (서버 불필요).

import { h, api, fmt, errorBox, loading, slider, select, toggle, inlineMd, esc } from "./lib.js";
import { lineChart, barChart, seriesColor } from "./charts.js";
import { diagram, DIAGRAMS } from "./diagrams.js";

// ---------------------------------------------------------------------------
// 공통
// ---------------------------------------------------------------------------
function card(title, desc, ...body) {
  return h("section", { class: "card widget" },
    h("header", { class: "card-head" }, h("h3", {}, title), desc ? h("p", { class: "card-desc", html: inlineMd(desc) }) : null),
    ...body);
}

const mulberry32 = (a) => () => {
  a |= 0; a = (a + 0x6d2b79f5) | 0;
  let t = Math.imul(a ^ (a >>> 15), 1 | a);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
};
const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const range = (a, b, n) => Array.from({ length: n }, (_, i) => a + ((b - a) * i) / (n - 1));

/** 체크포인트 선택기. 없으면 실험실로 안내한다. */
function ckptPicker(ctx, { task, prefer = [], label = "체크포인트", onChange }) {
  const list = ctx.status.checkpoints.filter((c) => !task || c.task === task);
  if (!list.length) {
    return { el: noCkpt(task), value: null };
  }
  // prefer 항목은 "이름" 또는 "과제/이름"
  const byName = (n) => list.find((c) => (n.includes("/") ? `${c.task}/${c.name}` === n : c.name === n));
  const initial = (prefer.map(byName).find(Boolean) || list[0]).path;
  const picker = { value: initial };
  picker.el = select({
    label, value: initial,
    options: list.map((c) => [c.path, task ? c.name : `${c.task} / ${c.name}`]),
    onChange: (v) => { picker.value = v; onChange?.(v); },
  });
  return picker;
}

function noCkpt(task) {
  return h("div", { class: "notice" },
    `${task ? `'${task}' 과제의 ` : ""}학습된 체크포인트가 없습니다. `,
    h("a", { href: "#/lab" }, "실험실"), "에서 SFT 를 먼저 학습하거나, 홈의 ", h("a", { href: "#/" }, "‘필수 체크포인트 한 번에 만들기’"), " 를 실행하세요.");
}

const taskOf = (ctx, path) => ctx.status.checkpoints.find((c) => c.path === path)?.task;
const defaultPrompt = (task) => (task === "arithmetic" ? "7+12=" : "the movie");

function run(button, out, fn) {
  button.addEventListener("click", async () => {
    button.disabled = true;
    out.replaceChildren(loading());
    try { await fn(); } catch (e) { out.replaceChildren(errorBox(e)); }
    button.disabled = false;
  });
}

// ---------------------------------------------------------------------------
// 다이어그램 위젯
// ---------------------------------------------------------------------------
const diagramWidget = (key, title, desc) => () => card(title, desc, diagram(DIAGRAMS[key]));

// ---------------------------------------------------------------------------
// 00 상태
// ---------------------------------------------------------------------------
const ESSENTIALS = [
  { key: "sentiment/sft", label: "감성 SFT", job: { kind: "pretrain", params: { task: "sentiment" } } },
  { key: "arithmetic/sft", label: "덧셈 SFT", job: { kind: "pretrain", params: { task: "arithmetic" } } },
  { key: "sentiment/rm", label: "감성 보상 모델 + 선호 쌍", job: { kind: "reward_model", params: { task: "sentiment" } } },
  { key: "sentiment/ppo", label: "감성 PPO (β=0.3)", job: { kind: "train", params: { task: "sentiment", algo: "ppo" } } },
  { key: "sentiment/ppo_hack", label: "감성 PPO (β=0.05, 해킹)", job: { kind: "train", params: { task: "sentiment", algo: "ppo", kl_coef: 0.05, name: "ppo_hack" } } },
  { key: "sentiment/dpo", label: "감성 DPO", job: { kind: "dpo", params: { task: "sentiment" } } },
  { key: "arithmetic/grpo", label: "덧셈 GRPO", job: { kind: "train", params: { task: "arithmetic", algo: "grpo" } } },
];

function statusWidget(ctx) {
  const have = new Set([...ctx.status.checkpoints.map((c) => `${c.task}/${c.name}`), ...ctx.status.reward_models.map((t) => `${t}/rm`)]);
  const missing = ESSENTIALS.filter((e) => !have.has(e.key));
  const btn = h("button", { class: "btn primary", disabled: !missing.length }, missing.length ? `필수 체크포인트 한 번에 만들기 (${missing.length}개, CPU 약 ${missing.length * 1.2 | 0}~${missing.length * 2}분)` : "모두 준비됨 ✓");
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    for (const m of missing) await api("/api/jobs", m.job);
    location.hash = "#/lab";
  });
  return card("준비 상태", "시각화 위젯 중 일부는 **학습된 체크포인트**를 직접 실행합니다. 아래 항목이 준비되면 모든 위젯을 쓸 수 있습니다. 작업은 순서대로 대기열에서 실행됩니다.",
    h("ul", { class: "checklist" }, ESSENTIALS.map((e) =>
      h("li", { class: have.has(e.key) ? "ok" : "" }, h("span", { class: "check" }, have.has(e.key) ? "✓" : "○"), e.label, h("code", {}, `runs/${e.key}`)))),
    h("div", { class: "row" }, btn, h("a", { class: "btn ghost", href: "#/lab" }, "실험실 열기 →")));
}

// ---------------------------------------------------------------------------
// 01 토큰 생성 추적
// ---------------------------------------------------------------------------
function traceWidget(ctx) {
  const out = h("div", { class: "widget-out" });
  const promptIn = h("input", { type: "text", class: "text-input", "aria-label": "프롬프트" });
  const picker = ckptPicker(ctx, { prefer: ["sentiment/ppo", "sentiment/sft", "ppo", "sft"], onChange: (v) => { promptIn.value = defaultPrompt(taskOf(ctx, v)); } });
  if (!picker.value) return card("토큰 생성 추적", null, picker.el);
  promptIn.value = defaultPrompt(taskOf(ctx, picker.value));
  let temperature = 1, greedy = false, seed = 0;
  const go = h("button", { class: "btn primary" }, "생성");
  const again = h("button", { class: "btn" }, "다른 샘플");
  again.addEventListener("click", () => { seed += 1; go.click(); });
  run(go, out, async () => {
    const r = await api("/api/trace", { ckpt: picker.value, prompt: promptIn.value, temperature, greedy, seed });
    out.replaceChildren(renderTrace(r));
  });
  return card("토큰 생성 추적 — 한 에피소드 들여다보기",
    "실제 체크포인트로 응답을 **한 토큰씩** 생성합니다. 각 토큰(행동)을 누르면 그 순간의 정책 분포 π(·|s_t) 와 참조 정책(SFT)의 확률을 비교할 수 있습니다.",
    h("div", { class: "controls" }, picker.el, h("label", { class: "control" }, h("span", { class: "control-label" }, "프롬프트"), promptIn),
      slider({ label: "온도", min: 0.2, max: 2, step: 0.1, value: 1, onInput: (v) => (temperature = v), format: (v) => v.toFixed(1) }),
      toggle({ label: "greedy", value: false, onChange: (v) => (greedy = v) }), go, again),
    out);
}

function renderTrace(r) {
  const detail = h("div", { class: "trace-detail" });
  const chips = h("div", { class: "token-row", role: "list" },
    h("span", { class: "tok prompt" }, "BOS"),
    [...r.prompt].map((c) => h("span", { class: "tok prompt" }, c === " " ? "␣" : c)));
  const minLogp = Math.min(...r.steps.map((s) => s.logp), -4);
  r.steps.forEach((st, i) => {
    const conf = 1 - st.logp / minLogp; // 0..1 (1 = 확신)
    const chip = h("button", { class: "tok action", role: "listitem", style: { "--conf": conf.toFixed(2) }, title: `log π = ${fmt(st.logp)}` }, st.token === " " ? "␣" : st.token);
    chip.addEventListener("click", () => pick(i, chip));
    chips.append(chip);
  });
  function pick(i, chip) {
    chips.querySelectorAll(".tok.selected").forEach((c) => c.classList.remove("selected"));
    chip.classList.add("selected");
    const st = r.steps[i];
    const maxP = Math.max(...st.top.map((t) => Math.max(t.p, st.ref_top?.[t.token] ?? 0)));
    detail.replaceChildren(
      h("div", { class: "trace-state" }, h("span", { class: "muted" }, `t=${i}  상태 s_t = `), h("code", {}, JSON.stringify(st.state)),
        h("span", { class: "muted" }, "  →  행동 a_t = "), h("code", {}, JSON.stringify(st.token))),
      h("div", { class: "prob-legend" }, h("span", {}, h("i", { style: { background: seriesColor(0) } }), "정책 π_θ"),
        r.has_ref ? h("span", {}, h("i", { style: { background: seriesColor(1) } }), "참조 π_ref (SFT)") : null),
      h("div", { class: "prob-list" }, st.top.map((t) => h("div", { class: `prob-row${t.token === st.token ? " chosen" : ""}` },
        h("span", { class: "prob-tok" }, t.token === " " ? "␣" : t.token),
        h("div", { class: "prob-bars" },
          h("div", { class: "pbar", style: { width: `${(100 * t.p) / maxP}%`, background: seriesColor(0) } }),
          r.has_ref ? h("div", { class: "pbar ref", style: { width: `${(100 * (st.ref_top[t.token] ?? 0)) / maxP}%`, background: seriesColor(1) } }) : null),
        h("span", { class: "prob-val" }, `${(100 * t.p).toFixed(1)}%`, r.has_ref ? h("span", { class: "muted" }, ` / ${(100 * (st.ref_top[t.token] ?? 0)).toFixed(1)}%`) : null)))),
      h("div", { class: "stat-row" },
        stat("log π_θ(a_t|s_t)", fmt(st.logp)),
        r.has_ref ? stat("log π_ref(a_t|s_t)", fmt(st.ref_logp)) : null,
        r.has_ref ? stat("KL 항 (차이)", fmt(st.kl_term)) : null,
        stat("엔트로피", fmt(st.entropy))),
    );
  }
  const series = [{ name: "log π_θ(a_t|s_t)", points: r.steps.map((s, i) => [i, s.logp]) }];
  if (r.has_ref) series.push({ name: "log π_θ − log π_ref (KL 항)", points: r.steps.map((s, i) => [i, s.kl_term]) });
  const wrap = h("div", {},
    h("div", { class: "trace-result" },
      h("span", {}, "응답: ", h("code", {}, JSON.stringify(r.response))),
      h("span", { class: `reward-pill ${r.reward > 0 ? "pos" : r.reward < 0 ? "neg" : ""}` }, `보상 R = ${fmt(r.reward, 1)}`),
      r.has_ref ? h("span", { class: "muted" }, `시퀀스 KL 추정 = ${fmt(r.steps.reduce((a, s) => a + s.kl_term, 0))}`) : null),
    chips, h("p", { class: "hint" }, "색이 진할수록 정책이 확신한 토큰입니다. 토큰을 눌러 보세요."), detail,
    h("h4", {}, "토큰별 log-prob"), lineChart({ series, xLabel: "t", height: 180, refLines: [{ y: 0 }] }));
  setTimeout(() => chips.querySelector(".tok.action")?.click());
  return wrap;
}

const stat = (label, value) => h("div", { class: "stat" }, h("div", { class: "stat-label" }, label), h("div", { class: "stat-value" }, value));

// ---------------------------------------------------------------------------
// 01 텐서 레이아웃
// ---------------------------------------------------------------------------
function tensorsWidget(ctx) {
  const out = h("div", { class: "widget-out" });
  const picker = ckptPicker(ctx, { prefer: ["sentiment/sft", "sft"] });
  if (!picker.value) return card("배치 텐서 레이아웃", null, picker.el);
  let view = "tokens";
  let data = null;
  const go = h("button", { class: "btn primary" }, "배치 롤아웃");
  const views = select({ label: "보기", value: view, options: [["tokens", "input_ids (토큰)"], ["attention_mask", "attention_mask"], ["response_mask", "response_mask"], ["action_mask", "action_mask = response_mask[:, 1:]"], ["logprobs", "token_logprobs (L−1)"]], onChange: (v) => { view = v; if (data) draw(); } });
  function draw() {
    const t = data.tensors;
    const L = t.tokens[0].length;
    const grid = h("div", { class: "tensor-grid", style: { "--cols": L } });
    grid.append(h("div", { class: "tg-corner" }, "행\\열"), ...Array.from({ length: L }, (_, j) => h("div", { class: "tg-col" }, j)));
    t.tokens.forEach((row, i) => {
      grid.append(h("div", { class: "tg-row" }, i));
      row.forEach((tok, j) => {
        const att = t.attention_mask[i][j], resp = t.response_mask[i][j];
        const kind = !att ? "pad" : resp ? "resp" : "prompt";
        let text = tok === " " ? "␣" : tok;
        let extra = "";
        if (view === "attention_mask") text = att;
        if (view === "response_mask") text = resp;
        if (view === "action_mask") { text = j === 0 ? "" : t.response_mask[i][j]; extra = j === 0 ? " shifted" : ""; }
        if (view === "logprobs") {
          const v = j === 0 ? null : t.logprobs[i][j - 1];
          text = j === 0 ? "" : t.response_mask[i][j] ? v.toFixed(1) : "·";
          extra = j === 0 ? " shifted" : "";
        }
        grid.append(h("div", { class: `tg-cell ${kind}${extra}`, title: `[${i}, ${j}] ${tok} · attn=${att} resp=${resp}` }, text));
      });
    });
    out.replaceChildren(
      h("div", { class: "tensor-legend" }, h("span", { class: "tg-cell pad" }, "pad"), "왼쪽/오른쪽 패딩 ", h("span", { class: "tg-cell prompt" }, "x"), "프롬프트(BOS 포함) ", h("span", { class: "tg-cell resp" }, "y"), "정책이 생성한 응답(+EOS)"),
      h("div", { class: "tensor-scroll" }, grid),
      h("p", { class: "hint", html: inlineMd(view === "action_mask" || view === "logprobs"
        ? "`token_logprobs()` 의 출력은 길이 **L−1** 입니다: 열 j 의 값은 `log π(ids[j] | ids[:j])`. 그래서 0번 열이 비고, 행동 마스크는 `response_mask[:, 1:]` 가 됩니다."
        : "왼쪽 패딩 덕분에 모든 행의 **마지막 열**이 '다음 토큰 예측 위치'로 정렬됩니다. EOS 이후는 오른쪽 패딩(∅)입니다.") }));
  }
  run(go, out, async () => { data = await api("/api/rollout", { ckpt: picker.value, n: 5, with_tensors: true }); draw(); });
  return card("배치 텐서 레이아웃", "롤아웃 한 배치의 `input_ids`, `attention_mask`, `response_mask` 를 그대로 펼쳐 봅니다. 레슨 문서의 '가장 많이 헷갈리는 부분'을 눈으로 확인하세요.",
    h("div", { class: "controls" }, picker.el, views, go), out);
}

// ---------------------------------------------------------------------------
// 02 REINFORCE 시뮬레이터 (브라우저 계산)
// ---------------------------------------------------------------------------
const CANDIDATES = [
  { text: " was great and fun.", r: 2 },
  { text: " was good.", r: 1 },
  { text: " was fun and sad.", r: 0 },
  { text: " was boring.", r: -1 },
  { text: " was bad and awful.", r: -2 },
];

function banditWidget() {
  let lr = 0.5, batch = 8, shift = 0, timer = null;
  const init = [0.2, 0.4, 0, 0.6, 0.5]; // SFT 처럼 부정 응답도 꽤 높은 확률
  const mk = (useBaseline, seed) => ({ useBaseline, logits: [...init], rand: mulberry32(seed), hist: [], gradNorms: [] });
  let runs;
  const softmax = (z) => { const m = Math.max(...z); const e = z.map((v) => Math.exp(v - m)); const s = e.reduce((a, b) => a + b); return e.map((v) => v / s); };
  const sample = (p, rand) => { let u = rand(); for (let i = 0; i < p.length; i++) { if ((u -= p[i]) <= 0) return i; } return p.length - 1; };
  const expected = (p) => p.reduce((a, pi, i) => a + pi * CANDIDATES[i].r, 0);

  function reset() {
    runs = [mk(false, 7), mk(true, 7)];
    runs.forEach((rn) => rn.hist.push([0, expected(softmax(rn.logits))]));
    draw();
  }
  function step() {
    for (const rn of runs) {
      const p = softmax(rn.logits);
      const idx = Array.from({ length: batch }, () => sample(p, rn.rand));
      const R = idx.map((i) => CANDIDATES[i].r + shift);
      const b = rn.useBaseline ? R.reduce((a, v) => a + v) / batch : 0;
      const grad = new Array(p.length).fill(0);
      idx.forEach((i, k) => { for (let j = 0; j < p.length; j++) grad[j] += ((R[k] - b) * ((j === i) - p[j])) / batch; });
      rn.gradNorms.push(Math.hypot(...grad));
      rn.logits = rn.logits.map((z, j) => z + lr * grad[j]);
      rn.hist.push([rn.hist.length, expected(softmax(rn.logits))]);
    }
    draw();
    if (runs[0].hist.length > 150) stop();
  }
  const chart = lineChart({ series: [], xLabel: "업데이트", yLabel: "기대 보상 E[R]", height: 200, yMin: -2, yMax: 2, refLines: [{ y: 2, label: "최대 2" }] });
  const probsEl = h("div", { class: "bandit-probs" });
  const statsEl = h("div", { class: "stat-row" });
  function draw() {
    chart.update(runs.map((rn, i) => ({ name: rn.useBaseline ? "baseline = 배치 평균" : "baseline 없음", points: rn.hist, colorIndex: i })));
    probsEl.replaceChildren(...runs.map((rn, ri) => {
      const p = softmax(rn.logits);
      return h("div", { class: "bandit-col" }, h("div", { class: "bandit-title" }, h("i", { style: { background: seriesColor(ri) } }), rn.useBaseline ? "baseline 사용" : "baseline 없음"),
        CANDIDATES.map((c, i) => h("div", { class: "prob-row" }, h("span", { class: "prob-tok wide" }, `the movie${c.text}`), h("div", { class: "prob-bars" }, h("div", { class: "pbar", style: { width: `${100 * p[i]}%`, background: seriesColor(ri) } })), h("span", { class: "prob-val" }, `${(100 * p[i]).toFixed(0)}%  R=${c.r + shift}`))));
    }));
    const sd = (a) => { if (a.length < 2) return 0; const m = a.reduce((x, y) => x + y) / a.length; return Math.sqrt(a.reduce((x, y) => x + (y - m) ** 2, 0) / a.length); };
    statsEl.replaceChildren(...runs.map((rn) => stat(`${rn.useBaseline ? "baseline" : "no baseline"}: 기울기 크기 평균±표준편차`, rn.gradNorms.length ? `${fmt(rn.gradNorms.reduce((a, b) => a + b) / rn.gradNorms.length, 2)} ± ${fmt(sd(rn.gradNorms), 2)}` : "–")));
  }
  function stop() { clearInterval(timer); timer = null; playBtn.textContent = "▶ 학습"; }
  const playBtn = h("button", { class: "btn primary" }, "▶ 학습");
  playBtn.addEventListener("click", () => { if (timer) return stop(); timer = setInterval(step, 120); playBtn.textContent = "⏸ 정지"; });
  const stepBtn = h("button", { class: "btn" }, "1 스텝");
  stepBtn.addEventListener("click", step);
  const resetBtn = h("button", { class: "btn ghost" }, "초기화");
  resetBtn.addEventListener("click", () => { stop(); reset(); });
  reset();
  return card("REINFORCE 시뮬레이터 — baseline 의 효과",
    "프롬프트 `the movie` 에 대한 응답이 다섯 가지뿐인 장난감 정책입니다. 같은 난수로 **baseline 없음 / 배치 평균 baseline** 두 정책을 동시에 학습합니다. `보상 +5 이동` 을 켜서 모든 보상을 양수로 만들면 baseline 이 없을 때 학습이 얼마나 흔들리는지 보세요.",
    h("div", { class: "controls" },
      slider({ label: "학습률", min: 0.05, max: 2, step: 0.05, value: lr, onInput: (v) => (lr = v), format: (v) => v.toFixed(2) }),
      slider({ label: "배치 크기", min: 1, max: 32, step: 1, value: batch, onInput: (v) => (batch = v) }),
      toggle({ label: "보상 +5 이동", value: false, onChange: (v) => { shift = v ? 5 : 0; draw(); } }),
      playBtn, stepBtn, resetBtn),
    chart, probsEl, statsEl,
    h("p", { class: "hint", html: inlineMd("업데이트 규칙: `logits += lr · mean[(R − b) · (onehot(y) − π)]` — `∇ log softmax` 를 손으로 쓴 것입니다.") }));
}

// ---------------------------------------------------------------------------
// 03 GAE 계산기
// ---------------------------------------------------------------------------
function gaeWidget() {
  const toks = ["was", "really", "good", "and", "fun", ".", "EOS"];
  const kl = [0.05, 0.4, 0.1, 0.02, 0.3, 0.01, 0.0];
  let gamma = 1, lam = 0.95, beta = 0.3, score = 2, critic = "zero";
  const out = h("div", {});
  function compute() {
    const T = toks.length;
    const r = kl.map((k, t) => -beta * k + (t === T - 1 ? score : 0));
    // "정확한" critic: 실제 남은 보상 합 (γ 할인)
    const mc = new Array(T).fill(0);
    for (let t = T - 1; t >= 0; t--) mc[t] = r[t] + (t + 1 < T ? gamma * mc[t + 1] : 0);
    const rand = mulberry32(3);
    const V = critic === "zero" ? new Array(T).fill(0) : critic === "exact" ? mc : mc.map((v) => v + (rand() - 0.5) * 2);
    const delta = r.map((rt, t) => rt + gamma * (t + 1 < T ? V[t + 1] : 0) - V[t]);
    const A = new Array(T).fill(0);
    for (let t = T - 1; t >= 0; t--) A[t] = delta[t] + (t + 1 < T ? gamma * lam * A[t + 1] : 0);
    return { r, V, delta, A, ret: A.map((a, t) => a + V[t]) };
  }
  function draw() {
    const c = compute();
    const signed = (v) => (v >= 0 ? "var(--div-pos)" : "var(--div-neg)");
    out.replaceChildren(
      h("div", { class: "grid-2" },
        h("div", {}, h("h4", {}, "토큰별 보상 r_t"), barChart({ bars: toks.map((t, i) => ({ label: t, value: c.r[i], color: signed(c.r[i]), tip: `KL_t=${kl[i]} → −β·KL = ${fmt(-beta * kl[i])}` + (i === toks.length - 1 ? `, + 점수 ${score}` : "") })), height: 170, showValues: true })),
        h("div", {}, h("h4", {}, "advantage A_t"), barChart({ bars: toks.map((t, i) => ({ label: t, value: c.A[i], color: signed(c.A[i]) })), height: 170, showValues: true }))),
      h("div", { class: "table-wrap" }, h("table", { class: "num-table" },
        h("thead", {}, h("tr", {}, h("th", {}, ""), toks.map((t) => h("th", {}, t)))),
        h("tbody", {}, [["r_t", c.r], ["V(s_t)", c.V], ["δ_t", c.delta], ["A_t", c.A], ["return", c.ret]].map(([name, arr]) =>
          h("tr", {}, h("th", {}, name), arr.map((v) => h("td", {}, fmt(v, 2))))))),
      ));
  }
  draw();
  return card("GAE 계산기",
    "응답 토큰마다 KL 벌점이, **마지막 토큰에만** 점수가 들어갑니다. critic 이 0 이면 advantage 는 '남은 보상의 합'에 가깝고, critic 이 정확하면 δ 가 작아집니다. λ 를 0 으로 내리면 1-step TD 가 됩니다. (가독성을 위해 단어 단위로 표시했습니다.)",
    h("div", { class: "controls" },
      slider({ label: "γ", min: 0.8, max: 1, step: 0.01, value: gamma, onInput: (v) => { gamma = v; draw(); }, format: (v) => v.toFixed(2) }),
      slider({ label: "λ", min: 0, max: 1, step: 0.05, value: lam, onInput: (v) => { lam = v; draw(); }, format: (v) => v.toFixed(2) }),
      slider({ label: "β (kl_coef)", min: 0, max: 1, step: 0.05, value: beta, onInput: (v) => { beta = v; draw(); }, format: (v) => v.toFixed(2) }),
      slider({ label: "점수 R", min: -2, max: 4, step: 0.5, value: score, onInput: (v) => { score = v; draw(); }, format: (v) => v.toFixed(1) }),
      select({ label: "critic", value: critic, options: [["zero", "V = 0 (학습 초기)"], ["noisy", "부정확한 V"], ["exact", "정확한 V"]], onChange: (v) => { critic = v; draw(); } })),
    out);
}

// ---------------------------------------------------------------------------
// 03 PPO clip
// ---------------------------------------------------------------------------
function clipWidget() {
  let eps = 0.2;
  const xs = range(0, 2, 201);
  const obj = (rho, A) => Math.min(rho * A, Math.min(Math.max(rho, 1 - eps), 1 + eps) * A);
  const mkChart = (A) => lineChart({ series: [], xLabel: "ratio ρ = π_θ/π_old", height: 200, yMin: A > 0 ? 0 : -2, yMax: A > 0 ? 2 : 0, xFormat: (v) => v.toFixed(1), noDots: true });
  const cPos = mkChart(1), cNeg = mkChart(-1);
  function draw() {
    for (const [c, A] of [[cPos, 1], [cNeg, -1]]) {
      c.update([
        { name: "ρ·A (clip 없음)", points: xs.map((x) => [x, x * A]), dashed: true, colorIndex: 1 },
        { name: "PPO 목적 min(ρA, clip(ρ)A)", points: xs.map((x) => [x, obj(x, A)]), colorIndex: 0 },
      ]);
    }
  }
  const wrap = h("div", { class: "grid-2" },
    h("div", {}, h("h4", {}, "A > 0 (좋은 행동)"), cPos, h("p", { class: "hint" }, `ρ > 1+ε 이면 평평 → 확률을 더 올려도 이득 없음`)),
    h("div", {}, h("h4", {}, "A < 0 (나쁜 행동)"), cNeg, h("p", { class: "hint" }, `ρ < 1−ε 이면 평평 → 확률을 더 내려도 이득 없음`)));
  draw();
  return card("PPO clipped objective", "목적함수(최대화)를 ratio 의 함수로 그렸습니다. **평평한 구간에서는 기울기가 0** 이어서 한 번의 데이터로 정책이 너무 멀리 가지 못합니다. ε 를 바꿔 보세요.",
    h("div", { class: "controls" }, slider({ label: "ε (clip_range)", min: 0.05, max: 0.5, step: 0.05, value: eps, onInput: (v) => { eps = v; draw(); }, format: (v) => v.toFixed(2) })),
    wrap);
}

// ---------------------------------------------------------------------------
// 04 Bradley–Terry
// ---------------------------------------------------------------------------
function btWidget() {
  const xs = range(-6, 6, 121);
  return card("Bradley–Terry 모델", "보상 차이 Δr = r(x,y_w) − r(x,y_l) 이 클수록 'y_w 가 낫다'는 확률이 1 에 가까워지고 손실은 0 에 가까워집니다. 차이만 중요하므로 보상의 절대값은 의미가 없습니다.",
    h("div", { class: "grid-2" },
      h("div", {}, h("h4", {}, "P(y_w ≻ y_l) = σ(Δr)"), lineChart({ series: [{ name: "σ(Δr)", points: xs.map((x) => [x, sigmoid(x)]) }], xLabel: "Δr", height: 180, yMin: 0, yMax: 1, refLines: [{ y: 0.5, label: "0.5" }], noDots: true })),
      h("div", {}, h("h4", {}, "손실 −log σ(Δr)"), lineChart({ series: [{ name: "loss", colorIndex: 1, points: xs.map((x) => [x, -Math.log(sigmoid(x))]) }], xLabel: "Δr", height: 180, yMin: 0, refLines: [{ y: Math.LN2, label: "log 2 (Δr=0)" }], noDots: true }))));
}

// ---------------------------------------------------------------------------
// 04 보상 모델 놀이터
// ---------------------------------------------------------------------------
const RM_PRESETS = [
  " was great.", " was really good and fun.", " was terrible.", " was good and good.",
  " was fun fun fun fun fun fun fun.", " was great great amazing nice.", " wav fun fun.", " was boring and great.",
];

function rmWidget(ctx) {
  if (!ctx.status.reward_models.includes("sentiment")) {
    return card("보상 모델 놀이터", null, h("div", { class: "notice" }, "감성 보상 모델이 없습니다. ", h("a", { href: "#/lab" }, "실험실"), "에서 ‘보상 모델’ 작업을 실행하세요."));
  }
  const out = h("div", { class: "widget-out" });
  const prompt = h("input", { class: "text-input", value: "the movie", "aria-label": "프롬프트" });
  const custom = h("input", { class: "text-input wide", value: " was nice nice nice.", "aria-label": "직접 쓴 응답" });
  const go = h("button", { class: "btn primary" }, "채점");
  run(go, out, async () => {
    const items = [...RM_PRESETS, custom.value].map((r) => ({ prompt: prompt.value, response: r }));
    const res = await api("/api/rm_score", { task: "sentiment", items });
    const maxAbs = Math.max(...res.items.map((i) => Math.max(Math.abs(i.rm_score), Math.abs(i.rule_reward))), 1);
    const bar = (v, k) => h("div", { class: "mini-bar" }, h("div", { class: `mini-fill ${v < 0 ? "neg" : ""}`, style: { width: `${(50 * Math.abs(v)) / maxAbs}%`, [v < 0 ? "right" : "left"]: "50%", background: seriesColor(k) } }));
    out.replaceChildren(h("div", { class: "table-wrap" }, h("table", { class: "rm-table" },
      h("thead", {}, h("tr", {}, h("th", {}, "응답"), h("th", {}, "문법"), h("th", {}, "규칙 보상"), h("th", {}, "보상 모델 점수"))),
      h("tbody", {}, res.items.map((it) => h("tr", { class: it.well_formed ? "" : "malformed" },
        h("td", {}, h("code", {}, prompt.value + it.response)),
        h("td", {}, it.well_formed ? "✓" : h("span", { class: "warn-text" }, "✗ 비문")),
        h("td", {}, h("div", { class: "bar-cell" }, bar(it.rule_reward, 1), h("span", {}, fmt(it.rule_reward, 1)))),
        h("td", {}, h("div", { class: "bar-cell" }, bar(it.rm_score, 0), h("span", {}, fmt(it.rm_score, 2))))))))),
      h("p", { class: "hint", html: inlineMd("보상 모델은 **SFT 가 만든 문장**만 보고 학습했습니다. 비문(✗)에도 높은 점수를 주는 경우를 찾아보세요 — 이것이 PPO 가 파고드는 틈입니다.") }));
  });
  setTimeout(() => go.click());
  return card("보상 모델 놀이터", "규칙 보상과 학습된 보상 모델 r_φ 가 같은 문장을 어떻게 채점하는지 비교합니다. 마지막 줄은 직접 써 볼 수 있습니다.",
    h("div", { class: "controls" }, h("label", { class: "control" }, h("span", { class: "control-label" }, "프롬프트"), prompt), h("label", { class: "control grow" }, h("span", { class: "control-label" }, "직접 쓴 응답"), custom), go), out);
}

// ---------------------------------------------------------------------------
// 04 reward hacking 비교
// ---------------------------------------------------------------------------
function hackingWidget(ctx) {
  const ck = ctx.status.checkpoints.filter((c) => c.task === "sentiment");
  if (!ck.length) return card("Reward hacking 비교", null, noCkpt("sentiment"));
  const order = ["sft", "ppo", "ppo_hack", "ppo_rm", "dpo", "reinforce", "grpo"];
  const chosen = new Set(order.filter((n) => ck.some((c) => c.name === n)).slice(0, 4));
  const out = h("div", { class: "widget-out" });
  const boxes = h("div", { class: "checks" }, ck.map((c) => {
    const cb = h("input", { type: "checkbox", checked: chosen.has(c.name) });
    cb.addEventListener("change", () => (cb.checked ? chosen.add(c.name) : chosen.delete(c.name)));
    return h("label", { class: "toggle" }, cb, c.name);
  }));
  const go = h("button", { class: "btn primary" }, "같은 프롬프트로 샘플 비교");
  run(go, out, async () => {
    const names = ck.filter((c) => chosen.has(c.name)).map((c) => c.name);
    const results = await Promise.all(names.map((n) => api("/api/rollout", { ckpt: ck.find((c) => c.name === n).path, n: 64, seed: 1 })));
    const hist = await Promise.all(names.map((n) => api(`/api/runs/sentiment/${n}`).catch(() => [])));
    const metric = (key) => names.map((n, i) => ({ label: n, value: results[i].metrics[key], color: seriesColor(i) }));
    const histSeries = (key) => names.map((n, i) => ({ name: n, colorIndex: i, points: hist[i].filter((r) => r[key] != null).map((r) => [r.step, r[key]]) })).filter((s) => s.points.length > 1);
    out.replaceChildren(
      h("div", { class: "grid-3" },
        h("div", {}, h("h4", {}, "평균 규칙 보상"), barChart({ bars: metric("reward"), height: 160, showValues: true })),
        h("div", {}, h("h4", {}, "긍정률"), barChart({ bars: metric("positive_rate"), height: 160, yMax: 1, showValues: true })),
        h("div", {}, h("h4", {}, "문법 정확도 (well_formed)"), barChart({ bars: metric("well_formed"), height: 160, yMax: 1, showValues: true }))),
      h("div", { class: "grid-2" },
        h("div", {}, h("h4", {}, "학습 중 보상 (eval/reward)"), lineChart({ series: histSeries("eval/reward"), xLabel: "step", height: 180 })),
        h("div", {}, h("h4", {}, "학습 중 문법 정확도 (eval/well_formed)"), lineChart({ series: histSeries("eval/well_formed"), xLabel: "step", height: 180, yMin: 0, yMax: 1 }))),
      h("div", { class: "sample-cols" }, names.map((n, i) => h("div", { class: "sample-col" },
        h("div", { class: "bandit-title" }, h("i", { style: { background: seriesColor(i) } }), n),
        results[i].samples.slice(0, 6).map((s) => h("div", { class: "sample" }, h("span", { class: `reward-pill small ${s.reward > 0 ? "pos" : s.reward < 0 ? "neg" : ""}` }, fmt(s.reward, 0)), h("code", {}, s.prompt + s.response)))))),
    );
  });
  return card("Reward hacking 비교", "같은 프롬프트로 여러 체크포인트를 샘플링합니다. **보상은 오르는데 문법이 무너지는** 모델을 찾아보세요 (`ppo_hack`: β=0.05, `ppo_rm`: 보상 모델 보상).",
    h("div", { class: "controls" }, boxes, go), out);
}

// ---------------------------------------------------------------------------
// 05 DPO 손실 곡선
// ---------------------------------------------------------------------------
function dpoCurveWidget() {
  let beta = 0.1;
  const xs = range(-40, 40, 161);
  const cLoss = lineChart({ series: [], xLabel: "log-ratio margin m", height: 190, yMin: 0, noDots: true });
  const cW = lineChart({ series: [], xLabel: "log-ratio margin m", height: 190, yMin: 0, yMax: 1, noDots: true });
  function draw() {
    const betas = [beta, 0.5].filter((b, i, a) => a.indexOf(b) === i);
    cLoss.update(betas.map((b, i) => ({ name: `β=${b}`, points: xs.map((m) => [m, -Math.log(sigmoid(b * m))]), colorIndex: i, dashed: i > 0 })));
    cW.update(betas.map((b, i) => ({ name: `β=${b}`, points: xs.map((m) => [m, sigmoid(-b * m)]), colorIndex: i, dashed: i > 0 })));
  }
  draw();
  return card("DPO 손실과 기울기 가중치",
    "m = [log π_θ(y_w) − log π_ref(y_w)] − [log π_θ(y_l) − log π_ref(y_l)]. 손실은 −log σ(β·m), 한 쌍의 기울기 크기는 **σ(−β·m)** 에 비례합니다. 이미 잘 구분한 쌍(m ≫ 0)은 거의 학습되지 않습니다. 점선은 비교용 β=0.5.",
    h("div", { class: "controls" }, slider({ label: "β", min: 0.02, max: 1, step: 0.02, value: beta, onInput: (v) => { beta = +v.toFixed(2); draw(); }, format: (v) => v.toFixed(2) })),
    h("div", { class: "grid-2" }, h("div", {}, h("h4", {}, "손실 −log σ(βm)"), cLoss), h("div", {}, h("h4", {}, "기울기 가중치 σ(−βm)"), cW)));
}

// ---------------------------------------------------------------------------
// 05 DPO 선호 쌍
// ---------------------------------------------------------------------------
function dpoPairsWidget(ctx) {
  if (!ctx.status.pairs.includes("sentiment")) {
    return card("선호 쌍과 암묵적 보상", null, h("div", { class: "notice" }, "선호 데이터가 없습니다. ", h("a", { href: "#/lab" }, "실험실"), "에서 ‘보상 모델’ 작업을 실행하면 선호 쌍도 함께 만들어집니다."));
  }
  const out = h("div", { class: "widget-out" });
  const picker = ckptPicker(ctx, { task: "sentiment", prefer: ["dpo", "sft"], label: "정책 π_θ" });
  let beta = 0.1;
  const go = h("button", { class: "btn primary" }, "계산");
  run(go, out, async () => {
    const r = await api("/api/dpo_pairs", { task: "sentiment", ckpt: picker.value, n: 10, beta });
    const margins = r.pairs.map((p) => p.reward_chosen - p.reward_rejected);
    out.replaceChildren(
      h("h4", {}, "쌍별 암묵적 보상 차이 β·m (양수면 chosen 을 더 선호)"),
      barChart({ bars: r.pairs.map((p, i) => ({ label: `#${i + 1}`, value: margins[i], color: margins[i] >= 0 ? "var(--div-pos)" : "var(--div-neg)", tipTitle: `${p.prompt}: ${p.chosen} ≻ ${p.rejected}`, tip: "β·m = " })), height: 160, showValues: true }),
      h("div", { class: "table-wrap" }, h("table", { class: "num-table left" },
        h("thead", {}, h("tr", {}, ["#", "chosen y_w", "rejected y_l", "log π_θ(y_w)", "log π_ref(y_w)", "log π_θ(y_l)", "log π_ref(y_l)", "손실"].map((t) => h("th", {}, t)))),
        h("tbody", {}, r.pairs.map((p, i) => h("tr", {}, h("td", {}, i + 1), h("td", {}, h("code", {}, p.prompt + p.chosen)), h("td", {}, h("code", {}, p.prompt + p.rejected)),
          [p.pi_chosen, p.ref_chosen, p.pi_rejected, p.ref_rejected, p.loss].map((v) => h("td", {}, fmt(v, 2)))))))),
      h("p", { class: "hint", html: inlineMd("`sft` 를 고르면 π_θ = π_ref 라서 모든 손실이 log 2 ≈ 0.693 입니다. `dpo` 를 고르면 rejected 의 log-prob 이 크게 떨어진 것을 볼 수 있습니다 — chosen 도 함께 떨어지는지 확인해 보세요.") }));
  });
  setTimeout(() => go.click());
  return card("선호 쌍과 암묵적 보상", "실제 선호 데이터(`runs/sentiment/rm/pairs.jsonl`)에서 정책과 참조 정책의 log-prob 을 계산합니다.",
    h("div", { class: "controls" }, picker.el, slider({ label: "β", min: 0.05, max: 1, step: 0.05, value: beta, onInput: (v) => (beta = v), format: (v) => v.toFixed(2) }), go), out);
}

// ---------------------------------------------------------------------------
// 06 GRPO 그룹
// ---------------------------------------------------------------------------
function grpoWidget(ctx) {
  const out = h("div", { class: "widget-out" });
  const promptIn = h("input", { class: "text-input", value: "7+12=", "aria-label": "프롬프트" });
  const picker = ckptPicker(ctx, { prefer: ["grpo", "sft"], task: ctx.status.checkpoints.some((c) => c.task === "arithmetic") ? "arithmetic" : null,
    onChange: (v) => { promptIn.value = defaultPrompt(taskOf(ctx, v)); } });
  if (!picker.value) return card("GRPO 그룹 샘플링", null, picker.el);
  let G = 8, scale = true, seed = 0;
  const one = h("button", { class: "btn primary" }, "그룹 하나 샘플링");
  const many = h("button", { class: "btn" }, "프롬프트 16개 × G");
  run(one, out, async () => {
    seed += 1;
    const r = await api("/api/rollout", { ckpt: picker.value, prompts: [promptIn.value], group_size: G, scale_rewards: scale, seed });
    const rs = r.samples.map((s) => s.reward);
    const mean = rs.reduce((a, b) => a + b) / G;
    const std = Math.sqrt(rs.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(G - 1, 1));
    out.replaceChildren(
      h("div", { class: "stat-row" }, stat("그룹 평균 보상", fmt(mean, 2)), stat("그룹 표준편차", fmt(std, 2)),
        stat("학습 신호", std === 0 ? "없음 (A=0)" : "있음")),
      h("div", { class: "grid-2" },
        h("div", { class: "group-list" }, r.samples.map((s, i) => h("div", { class: "sample" },
          h("span", { class: "muted" }, `y${i + 1}`), h("code", {}, s.prompt + s.response),
          h("span", { class: `reward-pill small ${s.reward > 0 ? "pos" : "neg"}` }, `r=${fmt(s.reward, 1)}`)))),
        h("div", {}, h("h4", {}, "advantage A_i"), barChart({ bars: r.samples.map((s, i) => ({ label: `y${i + 1}`, value: s.advantage, color: s.advantage >= 0 ? "var(--div-pos)" : "var(--div-neg)", tip: `r=${s.reward} → A=` })), height: 200, showValues: true }))),
      ...(std === 0 ? [h("div", { class: "notice" }, "그룹의 보상이 모두 같아 advantage 가 전부 0 입니다. 이 프롬프트에서는 아무것도 배우지 않습니다.")] : []));
  });
  run(many, out, async () => {
    const r = await api("/api/rollout", { ckpt: picker.value, n: 16 * G, group_size: G, scale_rewards: scale, seed: 99 });
    const groups = [];
    for (let i = 0; i < r.samples.length; i += G) groups.push(r.samples.slice(i, i + G));
    const zero = groups.filter((g) => g.every((s) => s.reward === g[0].reward)).length;
    out.replaceChildren(
      h("div", { class: "stat-row" }, stat("정확도(샘플)", fmt(r.metrics.reward, 2)), stat("A=0 인 그룹", `${zero} / ${groups.length}`)),
      barChart({ bars: groups.map((g) => ({ label: g[0].prompt, value: g.filter((s) => s.reward > 0).length / G, color: g.every((s) => s.reward === g[0].reward) ? "var(--muted-bar)" : seriesColor(0), tip: "그룹 정답률 " })), height: 200, yMax: 1, rotateLabels: true, showValues: false }),
      h("p", { class: "hint" }, "회색 막대 = 그룹 전체가 맞거나 전체가 틀린 프롬프트 (학습 신호 없음). G 를 줄이면 회색이 늘어납니다."));
  });
  return card("GRPO 그룹 샘플링", "같은 프롬프트에서 G 개를 뽑고 그룹 안에서 정규화한 advantage 를 봅니다. 정답(✓)은 양수, 오답은 음수를 받습니다.",
    h("div", { class: "controls" }, picker.el, h("label", { class: "control" }, h("span", { class: "control-label" }, "프롬프트"), promptIn),
      slider({ label: "G", min: 2, max: 16, step: 1, value: G, onInput: (v) => (G = v) }),
      toggle({ label: "std 로 나누기", value: true, onChange: (v) => (scale = v) }), one, many),
    out);
}

export const WIDGETS = {
  pipeline: diagramWidget("pipeline", "전체 파이프라인", "노드를 누르면 설명이, **단계별 재생**을 누르면 학습 흐름이 순서대로 강조됩니다."),
  status: statusWidget,
  mdp_loop: diagramWidget("mdp_loop", "MDP 루프", "언어 모델의 생성 과정을 강화학습의 상태-행동-보상 루프로 그렸습니다."),
  trace: traceWidget,
  tensors: tensorsWidget,
  reinforce_flow: diagramWidget("reinforce_flow", "REINFORCE 데이터 흐름", null),
  bandit: banditWidget,
  ppo_flow: diagramWidget("ppo_flow", "PPO-RLHF 구조", "RLHF 는 메모리에 네 모델을 올립니다. ❄ 는 고정, ∇ 는 학습되는 모델입니다."),
  gae: gaeWidget,
  clip: clipWidget,
  bt_curve: btWidget,
  rm_playground: rmWidget,
  hacking: hackingWidget,
  dpo_flow: diagramWidget("dpo_flow", "DPO 구조", "PPO 와 비교해 보세요: 생성(롤아웃)·보상 모델·critic 이 모두 사라졌습니다."),
  dpo_curve: dpoCurveWidget,
  dpo_pairs: dpoPairsWidget,
  grpo_flow: diagramWidget("grpo_flow", "GRPO 구조", null),
  grpo_group: grpoWidget,
};

export { card, stat, ckptPicker };
