// 실험실: 브라우저에서 학습 작업을 시작하고, 지표를 실시간 차트로 본다. + 실행 비교 페이지.

import { h, api, fmt, errorBox, select, inlineMd } from "./lib.js";
import { lineChart, seriesColor } from "./charts.js";

const KIND_NAMES = { pretrain: "SFT 사전학습", train: "RL 학습", reward_model: "보상 모델", dpo: "DPO" };
const STATUS_NAMES = { queued: "대기", running: "실행 중", done: "완료", error: "오류", stopped: "중단" };

// 한 차트에는 단위가 같은 지표만 (이중 축 금지)
const CHART_GROUPS = [
  { title: "보상", keys: ["score", "eval/reward", "reward"] },
  { title: "비율 지표 (0~1)", keys: ["eval/positive_rate", "eval/accuracy", "eval/well_formed", "eval/distinct", "positive_rate", "accuracy", "well_formed", "zero_adv_groups"], yMin: 0, yMax: 1 },
  { title: "KL (참조 정책과의 거리)", keys: ["kl"] },
  { title: "손실", keys: ["loss", "lm_loss", "pg_loss", "vf_loss", "train_loss", "val_loss"] },
  { title: "PPO 진단", keys: ["clipfrac", "approx_kl"] },
  { title: "선호 정확도", keys: ["reward_acc", "train_acc", "val_acc"], yMin: 0, yMax: 1 },
  { title: "DPO log-prob", keys: ["chosen_logp", "rejected_logp"] },
  { title: "DPO margin", keys: ["margin"] },
];

const RECIPES = [
  { title: "Reward hacking 재현", desc: "감성 PPO 를 β=0.05 로 → 보상은 오르고 문법은 무너짐", kind: "train", params: { task: "sentiment", algo: "ppo", kl_coef: 0.05, name: "ppo_hack" } },
  { title: "RLHF (보상 모델 보상)", desc: "보상 모델 점수로 PPO → 과최적화 관찰", kind: "train", params: { task: "sentiment", algo: "ppo", reward: "rm" } },
  { title: "baseline 없는 REINFORCE", desc: "분산이 큰 정책 경사", kind: "train", params: { task: "sentiment", algo: "reinforce", baseline: "none", name: "reinforce_nobase" } },
  { title: "GRPO, G=2", desc: "작은 그룹 → 학습 신호 없는 그룹 증가", kind: "train", params: { task: "arithmetic", algo: "grpo", group_size: 2, name: "grpo_g2" } },
  { title: "공격적인 DPO", desc: "lr=1e-4, 3 epoch → chosen 확률까지 하락", kind: "dpo", params: { task: "sentiment", lr: 1e-4, epochs: 3, name: "dpo_aggressive" } },
  { title: "노이즈 라벨 보상 모델", desc: "라벨 30% 뒤집기", kind: "reward_model", params: { task: "sentiment", label_noise: 0.3 } },
];

let activeSource = null;

export async function labPage(ctx, root) {
  const status = ctx.status;
  const formBox = h("div", {});
  const jobsBox = h("div", { class: "job-list" });
  const liveBox = h("div", { class: "live" }, h("div", { class: "empty" }, "작업을 시작하거나 목록에서 선택하세요."));
  let kind = "train";

  function field(label, input, help) {
    return h("label", { class: "field" }, h("span", { class: "field-label" }, label), input, help ? h("span", { class: "field-help" }, help) : null);
  }
  const num = (name, value, step = "any") => h("input", { type: "number", name, value: value ?? "", step, class: "text-input" });
  const sel = (name, options, value) => h("select", { name, class: "text-input" }, options.map((o) => { const [v, t] = Array.isArray(o) ? o : [o, o]; return h("option", { value: v, selected: v === value }, t); }));

  function drawForm(prefill = {}) {
    const task = prefill.task || "sentiment";
    const algo = prefill.algo || "ppo";
    const d = status.defaults[task] || {};
    const form = h("form", { class: "job-form" });
    const fields = [field("과제", sel("task", [["sentiment", "sentiment (감성 리뷰)"], ["arithmetic", "arithmetic (덧셈)"]], task))];
    if (kind === "pretrain") fields.push(field("스텝 수", num("steps", prefill.steps ?? d.pretrain_steps, 1)));
    if (kind === "train") {
      const a = { ...(d[algo] || {}), ...prefill };
      fields.push(
        field("알고리즘", sel("algo", [["reinforce", "REINFORCE"], ["ppo", "PPO"], ["grpo", "GRPO"]], algo)),
        field("보상", sel("reward", [["task", "규칙 / 정답 검증"], ["rm", "학습된 보상 모델 (RLHF)"]], prefill.reward || "task"), "보상 모델은 감성 과제에서 먼저 학습해야 합니다"),
        field("KL 계수 β", num("kl_coef", a.kl_coef)),
        field("학습률", num("lr", a.lr)),
        field("스텝 수", num("steps", a.steps, 1)),
        field("배치 크기", num("batch_size", a.batch_size ?? 64, 1)),
        algo === "grpo" ? field("그룹 크기 G", num("group_size", a.group_size ?? 8, 1)) : null,
        algo === "reinforce" ? field("baseline", sel("baseline", [["batch_mean", "배치 평균"], ["ema", "지수이동평균"], ["none", "없음"]], prefill.baseline || "batch_mean")) : null,
        field("실행 이름", h("input", { name: "name", value: prefill.name || "", placeholder: algo, class: "text-input" }), "같은 이름이면 덮어씁니다"),
      );
    }
    if (kind === "reward_model") fields.push(field("선호 쌍 개수", num("pairs", prefill.pairs ?? 2000, 1)), field("라벨 노이즈", num("label_noise", prefill.label_noise ?? 0)));
    if (kind === "dpo") fields.push(field("β", num("beta", prefill.beta ?? 0.1)), field("학습률", num("lr", prefill.lr ?? 2e-5)), field("epoch", num("epochs", prefill.epochs ?? 1, 1)),
      field("실행 이름", h("input", { name: "name", value: prefill.name || "", placeholder: "dpo", class: "text-input" })));
    fields.push(field("seed", num("seed", prefill.seed ?? 0, 1)));
    const needs = kind !== "pretrain" && !status.checkpoints.some((c) => c.task === task && c.name === "sft");
    form.append(...[h("div", { class: "fields" }, fields),
      needs ? h("div", { class: "notice" }, `${task} SFT 체크포인트가 없습니다. 먼저 ‘SFT 사전학습’을 실행하세요 (대기열에 넣어도 순서대로 실행됩니다).`) : null,
      h("button", { class: "btn primary", type: "submit" }, "▶ 학습 시작")].filter(Boolean));
    form.addEventListener("change", (e) => {
      if (["task", "algo"].includes(e.target.name)) {
        const fd = Object.fromEntries(new FormData(form));
        drawForm({ task: fd.task, algo: fd.algo });
      }
    });
    form.addEventListener("submit", async (e) => {
      e.preventDefault();
      const params = Object.fromEntries([...new FormData(form)].filter(([, v]) => v !== ""));
      try {
        const job = await api("/api/jobs", { kind, params });
        await refreshJobs();
        watch(job.id);
      } catch (err) { form.append(errorBox(err)); }
    });
    formBox.replaceChildren(form);
  }

  const kindTabs = h("div", { class: "segmented", role: "tablist" }, Object.entries(KIND_NAMES).map(([k, t]) => {
    const b = h("button", { class: k === kind ? "active" : "", role: "tab" }, t);
    b.addEventListener("click", () => { kind = k; kindTabs.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b)); drawForm(); });
    return b;
  }));

  const recipes = h("div", { class: "recipes" }, RECIPES.map((r) => {
    const b = h("button", { class: "recipe" }, h("strong", {}, r.title), h("span", {}, r.desc));
    b.addEventListener("click", () => {
      kind = r.kind;
      kindTabs.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x.textContent === KIND_NAMES[kind]));
      drawForm(r.params);
      formBox.scrollIntoView({ behavior: "smooth", block: "nearest" });
    });
    return b;
  }));

  async function refreshJobs() {
    const st = await api("/api/status");
    Object.assign(status, st);
    jobsBox.replaceChildren(...(st.jobs.length ? st.jobs.map((j) => {
      const item = h("button", { class: `job-item ${j.status}`, "data-id": j.id },
        h("span", { class: `status-chip ${j.status}` }, STATUS_NAMES[j.status]),
        h("span", { class: "job-title" }, `#${j.id} ${KIND_NAMES[j.kind]}`), h("span", { class: "muted" }, j.title));
      item.addEventListener("click", () => watch(j.id));
      return item;
    }) : [h("div", { class: "empty small" }, "아직 작업이 없습니다.")]));
    return st;
  }

  function watch(id) {
    activeSource?.close();
    jobsBox.querySelectorAll(".job-item").forEach((el) => el.classList.toggle("selected", +el.dataset.id === id));
    const job = status.jobs.find((j) => j.id === id) || { id, kind: "train", title: "" };
    const rows = [];
    const charts = new Map();
    const chartsBox = h("div", { class: "grid-2 live-charts" });
    const samplesBox = h("div", { class: "samples-feed" });
    const head = h("div", { class: "live-head" });
    const stopBtn = h("button", { class: "btn small danger" }, "■ 중단");
    stopBtn.addEventListener("click", () => api(`/api/jobs/${id}/stop`, {}));
    const lastRow = h("div", { class: "stat-row" });
    liveBox.replaceChildren(head, lastRow, chartsBox, h("h4", {}, "생성 샘플 (최근 기록 시점)"), samplesBox);

    function setHead(st, extra) {
      head.replaceChildren(...[h("div", {}, h("span", { class: `status-chip ${st}` }, STATUS_NAMES[st] || st), h("strong", {}, ` #${id} ${KIND_NAMES[job.kind]} `), h("span", { class: "muted" }, job.title)),
        ["queued", "running"].includes(st) ? stopBtn : null, extra || null].filter(Boolean));
    }
    setHead(job.status || "queued");

    function onRow(row, samples) {
      rows.push(row);
      for (const g of CHART_GROUPS) {
        const keys = g.keys.filter((k) => rows.some((r) => r[k] != null));
        if (!keys.length) continue;
        const series = keys.map((k) => ({ name: k, colorIndex: g.keys.indexOf(k), points: rows.filter((r) => r[k] != null).map((r) => [r.step, r[k]]) }));
        if (!charts.has(g.title)) {
          const c = lineChart({ series: [], xLabel: "step", height: 180, yMin: g.yMin, yMax: g.yMax, legend: true });
          charts.set(g.title, c);
          chartsBox.append(h("div", { class: "chart-card" }, h("h4", {}, g.title), c));
        }
        charts.get(g.title).update(series);
      }
      const show = Object.entries(row).filter(([k]) => !["step", "time"].includes(k)).slice(0, 6);
      lastRow.replaceChildren(h("div", { class: "stat" }, h("div", { class: "stat-label" }, "step / 시간"), h("div", { class: "stat-value" }, `${row.step} · ${row.time}s`)),
        ...show.map(([k, v]) => h("div", { class: "stat" }, h("div", { class: "stat-label" }, k), h("div", { class: "stat-value" }, fmt(v)))));
      if (samples.length) samplesBox.replaceChildren(...samples.map((s) => h("code", { class: "sample-line" }, s)));
    }

    const es = new EventSource(`/api/jobs/${id}/events`);
    activeSource = es;
    es.onmessage = (m) => {
      const ev = JSON.parse(m.data);
      if (ev.type === "log") onRow(ev.row, ev.samples);
      if (ev.type === "status") {
        const done = ["done", "error", "stopped"].includes(ev.status);
        setHead(ev.status, ev.error ? errorBox(ev.error) : ev.result ? h("span", { class: "muted" }, `저장: ${ev.result}`) : null);
        if (done) { es.close(); refreshJobs().then(() => ctx.onStatus?.()); }
        else refreshJobs();
      }
    };
    es.onerror = () => { if (es.readyState === EventSource.CLOSED) return; };
  }

  root.replaceChildren(
    h("div", { class: "page-head" }, h("h1", {}, "실험실"), h("p", { class: "lead" }, "하이퍼파라미터를 바꿔 학습을 직접 실행하고, 지표가 변하는 모습을 실시간으로 봅니다. 결과는 ", h("code", {}, "runs/<task>/<name>/"), " 에 저장되어 레슨 위젯과 ‘실행 비교’에서 바로 쓸 수 있습니다.")),
    h("div", { class: "lab-layout" },
      h("div", { class: "lab-side" },
        h("section", { class: "card" }, h("h3", {}, "새 작업"), kindTabs, formBox),
        h("section", { class: "card" }, h("h3", {}, "실험 레시피"), h("p", { class: "card-desc" }, "레슨에서 다룬 현상을 재현하는 설정입니다. 누르면 양식이 채워집니다."), recipes),
        h("section", { class: "card" }, h("h3", {}, "작업 목록"), jobsBox)),
      h("section", { class: "card lab-main" }, liveBox)));
  drawForm();
  const st = await refreshJobs();
  const running = st.jobs.find((j) => j.status === "running") || st.jobs.find((j) => j.status === "queued") || st.jobs[0];
  if (running) watch(running.id);
}

export function leaveLab() {
  activeSource?.close();
  activeSource = null;
}

// ---------------------------------------------------------------------------
// 실행 비교
// ---------------------------------------------------------------------------
export async function comparePage(ctx, root) {
  let task = ctx.status.runs.some((r) => r.task === "sentiment") ? "sentiment" : "arithmetic";
  const box = h("div", {});
  async function draw() {
    const runs = ctx.status.runs.filter((r) => r.task === task);
    if (!runs.length) { box.replaceChildren(h("div", { class: "notice" }, "이 과제의 실행 기록이 없습니다. ", h("a", { href: "#/lab" }, "실험실"), "에서 학습해 보세요.")); return; }
    const hist = Object.fromEntries(await Promise.all(runs.map(async (r) => [r.name, await api(`/api/runs/${task}/${r.name}`)])));
    const keys = [...new Set(Object.values(hist).flat().flatMap((row) => Object.keys(row)))].filter((k) => !["step", "time", "epoch"].includes(k)).sort();
    let metric = keys.includes("eval/reward") ? "eval/reward" : keys[0];
    const selected = new Set(runs.filter((r) => r.name !== "rm" && r.name !== "sft").map((r) => r.name).slice(0, 6));
    const chartBox = h("div", {});
    const tableBox = h("div", {});
    function render() {
      const names = runs.map((r) => r.name).filter((n) => selected.has(n));
      const series = names.map((n) => ({ name: n, colorIndex: runs.findIndex((r) => r.name === n), points: hist[n].filter((r) => r[metric] != null).map((r) => [r.step, r[metric]]) })).filter((s) => s.points.length);
      chartBox.replaceChildren(series.length ? lineChart({ series, xLabel: "step", yLabel: metric, height: 300, legend: true }) : h("div", { class: "empty" }, "선택한 실행에 이 지표가 없습니다."));
      tableBox.replaceChildren(h("div", { class: "table-wrap" }, h("table", { class: "num-table left" },
        h("thead", {}, h("tr", {}, h("th", {}, "실행"), h("th", {}, "처음"), h("th", {}, "마지막"), h("th", {}, "변화"), h("th", {}, "스텝"))),
        h("tbody", {}, series.map((s) => {
          const a = s.points[0][1], b = s.points[s.points.length - 1][1];
          return h("tr", {}, h("td", {}, h("i", { class: "swatch", style: { background: seriesColor(s.colorIndex) } }), s.name), h("td", {}, fmt(a)), h("td", {}, fmt(b)), h("td", { class: b >= a ? "up" : "down" }, `${b >= a ? "▲" : "▼"} ${fmt(Math.abs(b - a))}`), h("td", {}, s.points[s.points.length - 1][0]));
        })))));
    }
    const checks = h("div", { class: "checks" }, runs.map((r, i) => {
      const cb = h("input", { type: "checkbox", checked: selected.has(r.name) });
      cb.addEventListener("change", () => { cb.checked ? selected.add(r.name) : selected.delete(r.name); render(); });
      return h("label", { class: "toggle" }, cb, h("i", { class: "swatch", style: { background: seriesColor(i) } }), r.name);
    }));
    box.replaceChildren(h("div", { class: "controls" }, select({ label: "지표", value: metric, options: keys, onChange: (v) => { metric = v; render(); } })), checks, chartBox, tableBox);
    render();
  }
  root.replaceChildren(
    h("div", { class: "page-head" }, h("h1", {}, "실행 비교"), h("p", { class: "lead", html: inlineMd("`runs/` 에 저장된 학습 기록(`history.json`)을 겹쳐 봅니다. 같은 지표를 여러 알고리즘·하이퍼파라미터로 비교해 보세요.") })),
    h("section", { class: "card" }, h("div", { class: "controls" }, select({ label: "과제", value: task, options: [["sentiment", "sentiment"], ["arithmetic", "arithmetic"]], onChange: (v) => { task = v; draw(); } })), box));
  await draw();
}
