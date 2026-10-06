// 동작 구조 다이어그램: 노드(모델·데이터·신호)와 화살표, 단계별 재생 애니메이션.

import { h, inlineMd } from "./lib.js";

const NS = "http://www.w3.org/2000/svg";
const s = (tag, attrs = {}, ...kids) => {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v);
  for (const c of kids.flat()) if (c != null) el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return el;
};

const KIND_LABEL = {
  trainable: "학습되는 모델",
  frozen: "고정된 모델",
  data: "데이터",
  signal: "보상·손실 신호",
  op: "연산",
};

function border(n, tx, ty) {
  // 노드 중심에서 (tx,ty) 방향으로 나갈 때 사각형 테두리와 만나는 점
  const cx = n.x + n.w / 2, cy = n.y + n.h / 2;
  const dx = tx - cx, dy = ty - cy;
  if (!dx && !dy) return [cx, cy];
  const sx = (n.w / 2 + 4) / Math.abs(dx || 1e-9), sy = (n.h / 2 + 4) / Math.abs(dy || 1e-9);
  const k = Math.min(sx, sy);
  return [cx + dx * k, cy + dy * k];
}

export function diagram(spec) {
  const W = spec.width || 1000, H = spec.height || 400;
  const nodes = Object.fromEntries(spec.nodes.map((n) => [n.id, { h: 56, w: 170, ...n }]));
  const root = h("figure", { class: "diagram" });
  const svg = s("svg", { viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": spec.title || "다이어그램" });
  svg.append(s("defs", {},
    s("marker", { id: `arrow-${spec.id}`, viewBox: "0 0 10 10", refX: 9, refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" },
      s("path", { d: "M0,0 L10,5 L0,10 z", class: "arrow-head" }))));

  const edgeEls = spec.edges.map((e, i) => {
    const a = nodes[e.from], b = nodes[e.to];
    const pts = e.via ? [...e.via] : [];
    const first = pts[0] || [b.x + b.w / 2, b.y + b.h / 2];
    const last = pts[pts.length - 1] || [a.x + a.w / 2, a.y + a.h / 2];
    const p0 = border(a, ...first), p1 = border(b, ...last);
    const all = [p0, ...pts, p1];
    const d = all.map((p, k) => `${k ? "L" : "M"}${p[0].toFixed(1)},${p[1].toFixed(1)}`).join("");
    const g = s("g", { class: `edge ${e.kind || ""}`, "data-i": i });
    g.append(s("path", { d, class: "edge-line", "marker-end": `url(#arrow-${spec.id})` }));
    g.append(s("path", { d, class: "edge-flow" }));
    if (e.label) {
      const mid = all.length > 2 ? all[Math.floor(all.length / 2)] : [(p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2];
      const [lx, ly] = [mid[0] + (e.dx || 0), mid[1] + (e.dy || 0)];
      const w = e.label.length * 7.4 + 12;
      g.append(s("rect", { x: lx - w / 2, y: ly - 11, width: w, height: 20, rx: 10, class: "edge-label-bg" }));
      g.append(s("text", { x: lx, y: ly + 3.5, class: "edge-label", "text-anchor": "middle" }, e.label));
    }
    svg.append(g);
    return g;
  });

  const detail = h("div", { class: "diagram-detail" }, h("span", { class: "muted" }, "노드를 누르면 설명이 나옵니다."));
  const nodeEls = {};
  for (const n of Object.values(nodes)) {
    const g = s("g", { class: `node ${n.kind || "op"}`, tabindex: 0, role: "button", "aria-label": n.label });
    g.append(s("rect", { x: n.x, y: n.y, width: n.w, height: n.h, rx: 12 }));
    const lines = n.sub ? [n.label, n.sub] : [n.label];
    lines.forEach((t, k) => g.append(s("text", {
      x: n.x + n.w / 2, y: n.y + n.h / 2 + (lines.length === 1 ? 5 : k === 0 ? -3 : 15),
      class: k === 0 ? "node-label" : "node-sub", "text-anchor": "middle",
    }, t)));
    if (n.kind === "frozen") g.append(s("text", { x: n.x + n.w - 14, y: n.y + 17, class: "node-badge", "text-anchor": "middle" }, "❄"));
    if (n.kind === "trainable") g.append(s("text", { x: n.x + n.w - 14, y: n.y + 17, class: "node-badge", "text-anchor": "middle" }, "∇"));
    const showDetail = () => {
      for (const el of Object.values(nodeEls)) el.classList.remove("selected");
      g.classList.add("selected");
      detail.replaceChildren(...[
        h("div", { class: "detail-title" }, h("span", { class: `kind-chip ${n.kind || "op"}` }, KIND_LABEL[n.kind || "op"]), n.label),
        h("div", { html: inlineMd(n.desc || "") }),
        n.link ? h("a", { href: n.link, class: "detail-link" }, "관련 레슨으로 이동 →") : null,
      ].filter(Boolean));
    };
    g.addEventListener("click", showDetail);
    g.addEventListener("keydown", (ev) => { if (ev.key === "Enter" || ev.key === " ") { ev.preventDefault(); showDetail(); } });
    nodeEls[n.id] = g;
    svg.append(g);
  }

  // ---- 단계별 재생 ----
  const caption = h("div", { class: "diagram-caption" });
  let stepIdx = -1, timer = null;
  function setStep(i) {
    stepIdx = i;
    const st = spec.steps?.[i];
    svg.classList.toggle("stepping", !!st);
    edgeEls.forEach((el, k) => el.classList.toggle("active", !!st && st.edges.includes(k)));
    for (const [id, el] of Object.entries(nodeEls)) el.classList.toggle("active", !!st && st.nodes.includes(id));
    caption.innerHTML = st ? `<b>${i + 1}/${spec.steps.length}</b> ${inlineMd(st.caption)}` : "";
    prev.disabled = !st || i === 0;
    next.disabled = !spec.steps || i >= spec.steps.length - 1;
  }
  const play = h("button", { class: "btn small primary" }, "▶ 단계별 재생");
  const prev = h("button", { class: "btn small", disabled: true, "aria-label": "이전 단계" }, "◀");
  const next = h("button", { class: "btn small", "aria-label": "다음 단계" }, "▶");
  const reset = h("button", { class: "btn small ghost" }, "전체 보기");
  play.addEventListener("click", () => {
    if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ 단계별 재생"; return; }
    setStep(stepIdx + 1 >= spec.steps.length ? 0 : stepIdx + 1);
    play.textContent = "⏸ 일시정지";
    timer = setInterval(() => {
      if (stepIdx + 1 >= spec.steps.length) { clearInterval(timer); timer = null; play.textContent = "▶ 다시 재생"; return; }
      setStep(stepIdx + 1);
    }, 2600);
  });
  prev.addEventListener("click", () => setStep(Math.max(0, stepIdx - 1)));
  next.addEventListener("click", () => setStep(stepIdx + 1));
  reset.addEventListener("click", () => { clearInterval(timer); timer = null; play.textContent = "▶ 단계별 재생"; setStep(-1); });

  const kinds = [...new Set(Object.values(nodes).map((n) => n.kind || "op"))];
  root.append(...[
    spec.steps ? h("div", { class: "diagram-toolbar" }, play, prev, next, reset) : null,
    h("div", { class: "diagram-canvas" }, svg),
    caption,
    h("div", { class: "diagram-legend" }, kinds.map((k) => h("span", { class: `kind-chip ${k}` }, KIND_LABEL[k]))),
    detail,
  ].filter(Boolean));
  return root;
}

// ===========================================================================
// 레슨별 다이어그램 정의
// ===========================================================================
export const DIAGRAMS = {
  pipeline: {
    id: "pipeline", title: "전체 파이프라인", width: 1000, height: 400,
    nodes: [
      { id: "corpus", x: 20, y: 40, w: 160, label: "사전학습 코퍼스", sub: "task.corpus()", kind: "data",
        desc: "과제의 '언어'가 담긴 문장들. 감성 과제는 긍정·부정이 반반, 덧셈 과제는 정답이 40% 뿐입니다." },
      { id: "sft", x: 240, y: 40, w: 180, label: "SFT 정책 π_SFT", sub: "pretrain.py", kind: "trainable", link: "#/lesson/overview",
        desc: "다음 토큰 예측으로 학습한 언어 모델. 모든 RL 의 **출발점**입니다." },
      { id: "ref", x: 240, y: 170, w: 180, label: "참조 정책 π_ref", sub: "SFT 의 얼린 복사본", kind: "frozen", link: "#/lesson/reinforce",
        desc: "학습하지 않습니다. `log π_θ − log π_ref` 로 정책이 처음에서 얼마나 멀어졌는지(KL)를 잽니다." },
      { id: "online", x: 490, y: 40, w: 220, label: "온라인 RL", sub: "REINFORCE · PPO · GRPO", kind: "op", link: "#/lesson/reinforce",
        desc: "정책이 **직접 생성한 응답**에 보상을 받아 업데이트합니다. 레슨 02·03·06." },
      { id: "reward", x: 490, y: 170, w: 220, label: "보상 R(x, y)", sub: "규칙 · 정답 검증 · RM", kind: "signal", link: "#/lesson/mdp",
        desc: "응답이 끝난 뒤 한 번 주어지는 스칼라. 규칙(감성 단어 수), 정답 검증(덧셈), 또는 학습된 보상 모델 점수." },
      { id: "pairs", x: 240, y: 300, w: 180, label: "선호 쌍 (x, y_w, y_l)", sub: "preference.py", kind: "data", link: "#/lesson/reward",
        desc: "같은 프롬프트의 두 응답 중 더 나은 것(y_w)과 못한 것(y_l). 실제로는 사람이, 여기서는 규칙 보상이 라벨링합니다." },
      { id: "rm", x: 490, y: 300, w: 220, label: "보상 모델 r_φ", sub: "Bradley–Terry", kind: "frozen", link: "#/lesson/reward",
        desc: "선호 쌍으로 학습한 뒤 고정해서 RL 의 보상으로 씁니다 (RLHF). 분포 밖 문장에서 틀릴 수 있습니다." },
      { id: "dpo", x: 790, y: 300, w: 180, label: "DPO", sub: "보상 모델 없이", kind: "trainable", link: "#/lesson/dpo",
        desc: "선호 쌍으로 정책을 **직접** 학습. 샘플링·보상 모델·critic 이 없습니다." },
      { id: "tuned", x: 790, y: 40, w: 180, label: "조정된 정책 π_θ", sub: "runs/<task>/<algo>", kind: "trainable",
        desc: "결과 모델. 실험실에서 학습하고 시각화 위젯에서 SFT 와 비교해 보세요." },
    ],
    edges: [
      { from: "corpus", to: "sft", label: "다음 토큰 예측", dy: -24 },
      { from: "sft", to: "ref", label: "복사 후 고정", kind: "frozen" },
      { from: "sft", to: "online", label: "초기값", dy: -18 },
      { from: "ref", to: "online", label: "KL 제약", kind: "frozen", dx: 20 },
      { from: "reward", to: "online", label: "보상", kind: "signal" },
      { from: "online", to: "tuned" },
      { from: "sft", to: "pairs", label: "샘플링 + 라벨", via: [[200, 125], [200, 328]], dy: 0 },
      { from: "pairs", to: "rm", label: "BT 학습" },
      { from: "rm", to: "reward", label: "RLHF", kind: "signal" },
      { from: "pairs", to: "dpo", label: "직접 최적화", via: [[330, 380], [880, 380]] },
      { from: "dpo", to: "tuned" },
    ],
    steps: [
      { caption: "**SFT**: 코퍼스로 다음 토큰 예측을 학습해 기본 언어 능력을 갖춘 정책을 만듭니다.", nodes: ["corpus", "sft"], edges: [0] },
      { caption: "SFT 모델을 **복사해 얼려** 참조 정책으로 둡니다. 이후 KL 의 기준점이 됩니다.", nodes: ["sft", "ref"], edges: [1] },
      { caption: "**온라인 RL**: SFT 에서 출발해, 생성한 응답의 보상을 높이되 π_ref 에서 너무 멀어지지 않게 학습합니다.", nodes: ["sft", "ref", "online", "reward", "tuned"], edges: [2, 3, 4, 5] },
      { caption: "**RLHF**: 선호 쌍으로 보상 모델을 학습하고, 그 점수를 RL 의 보상으로 씁니다.", nodes: ["sft", "pairs", "rm", "reward", "online"], edges: [6, 7, 8, 4] },
      { caption: "**DPO**: 같은 선호 쌍을 보상 모델 없이 정책 학습에 바로 씁니다.", nodes: ["pairs", "dpo", "tuned"], edges: [9, 10] },
    ],
  },

  mdp_loop: {
    id: "mdp", title: "텍스트 생성 MDP", width: 1000, height: 300,
    nodes: [
      { id: "state", x: 20, y: 110, w: 210, h: 64, label: "상태 s_t", sub: "\"the movie was g\"", kind: "data",
        desc: "프롬프트 + 지금까지 생성한 토큰 전체. 코드에서는 `input_ids[:, :t+1]`." },
      { id: "policy", x: 300, y: 110, w: 170, h: 64, label: "정책 π_θ", sub: "TinyGPT", kind: "trainable", link: "#/lesson/mdp",
        desc: "상태를 읽고 다음 토큰 분포 π_θ(·|s_t) 를 출력하는 언어 모델." },
      { id: "dist", x: 540, y: 110, w: 180, h: 64, label: "분포 π(·|s_t)", sub: "softmax(logits)", kind: "op",
        desc: "어휘 크기만큼의 확률. 아래 '토큰 생성 추적' 위젯에서 실제 분포를 볼 수 있습니다." },
      { id: "action", x: 790, y: 110, w: 190, h: 64, label: "행동 a_t", sub: "샘플한 다음 토큰 \"o\"", kind: "data",
        desc: "분포에서 샘플링한 토큰 하나. 이것이 RL 의 '행동'입니다." },
      { id: "eos", x: 790, y: 225, w: 190, h: 56, label: "EOS 인가?", sub: "에피소드 종료", kind: "op",
        desc: "EOS 가 나오거나 `max_new_tokens` 에 도달하면 응답이 끝납니다." },
      { id: "reward", x: 520, y: 225, w: 220, h: 56, label: "보상 R(x, y)", sub: "응답 전체에 한 번", kind: "signal", link: "#/lesson/reward",
        desc: "중간 토큰에는 보상이 없습니다 (희소 보상). 이것이 credit assignment 문제의 원인입니다." },
    ],
    edges: [
      { from: "state", to: "policy", label: "forward", dy: -22 },
      { from: "policy", to: "dist" },
      { from: "dist", to: "action", label: "샘플링", dy: -22 },
      { from: "action", to: "state", label: "s(t+1) = s(t) + a(t)  결정적 전이", via: [[885, 40], [125, 40]] },
      { from: "action", to: "eos" },
      { from: "eos", to: "reward", label: "끝났으면", kind: "signal", dy: -20 },
    ],
    steps: [
      { caption: "현재 상태(지금까지의 문자열)를 정책에 넣습니다.", nodes: ["state", "policy"], edges: [0] },
      { caption: "정책이 다음 토큰의 확률 분포를 냅니다.", nodes: ["policy", "dist"], edges: [1] },
      { caption: "분포에서 행동(토큰)을 하나 샘플링합니다.", nodes: ["dist", "action"], edges: [2] },
      { caption: "토큰을 이어 붙여 다음 상태가 됩니다. 전이는 **결정적**입니다.", nodes: ["action", "state"], edges: [3] },
      { caption: "EOS 가 나오면 에피소드가 끝나고, 그때서야 **보상이 한 번** 주어집니다.", nodes: ["action", "eos", "reward"], edges: [4, 5] },
    ],
  },

  reinforce_flow: {
    id: "reinforce", title: "REINFORCE 한 스텝", width: 1000, height: 330,
    nodes: [
      { id: "prompts", x: 20, y: 30, w: 150, label: "프롬프트 x", kind: "data" },
      { id: "policy", x: 220, y: 30, w: 170, label: "정책 π_θ", sub: "generate()", kind: "trainable", desc: "현재 정책으로 응답을 샘플링합니다." },
      { id: "resp", x: 440, y: 30, w: 170, label: "응답 y", sub: "롤아웃", kind: "data" },
      { id: "R", x: 660, y: 30, w: 150, label: "보상 R", kind: "signal", desc: "`task_reward_fn` 또는 `reward_model_fn`." },
      { id: "ref", x: 440, y: 150, w: 170, label: "π_ref", sub: "log π_ref(y|x)", kind: "frozen" },
      { id: "kl", x: 660, y: 150, w: 150, label: "KL 추정", sub: "log π_θ − log π_ref", kind: "op" },
      { id: "adv", x: 840, y: 90, w: 140, h: 64, label: "A = R − βKL − b", kind: "signal", desc: "baseline b 를 빼서 분산을 줄인 advantage." },
      { id: "loss", x: 440, y: 260, w: 250, h: 56, label: "loss = −A · log π_θ(y|x)", kind: "signal", desc: "보상 가중 최대우도. 이 손실의 기울기가 정책 경사입니다." },
      { id: "update", x: 220, y: 260, w: 170, h: 56, label: "∇ 업데이트", sub: "Adam", kind: "op" },
    ],
    edges: [
      { from: "prompts", to: "policy" },
      { from: "policy", to: "resp", label: "샘플", dy: -20 },
      { from: "resp", to: "R" },
      { from: "resp", to: "ref" },
      { from: "ref", to: "kl" },
      { from: "R", to: "adv", kind: "signal" },
      { from: "kl", to: "adv", label: "−β", kind: "signal" },
      { from: "adv", to: "loss", via: [[910, 288]] },
      { from: "loss", to: "update" },
      { from: "update", to: "policy", label: "θ ← θ − η∇L", kind: "grad" },
    ],
    steps: [
      { caption: "현재 정책으로 프롬프트마다 응답을 샘플링합니다.", nodes: ["prompts", "policy", "resp"], edges: [0, 1] },
      { caption: "응답에 보상을 매기고, 참조 정책과의 log-prob 차이로 KL 을 추정합니다.", nodes: ["resp", "R", "ref", "kl"], edges: [2, 3, 4] },
      { caption: "보상에서 KL 벌점과 baseline 을 빼 advantage 를 만듭니다.", nodes: ["R", "kl", "adv"], edges: [5, 6] },
      { caption: "`−A · log π_θ(y|x)` 를 미분해 정책을 업데이트합니다. A>0 인 응답의 확률이 올라갑니다.", nodes: ["adv", "loss", "update", "policy"], edges: [7, 8, 9] },
    ],
  },

  ppo_flow: {
    id: "ppo", title: "PPO-RLHF 의 네 모델", width: 1000, height: 400,
    nodes: [
      { id: "actor", x: 20, y: 40, w: 170, label: "Actor π_θ", sub: "정책", kind: "trainable", desc: "응답을 생성하고 업데이트되는 정책." },
      { id: "roll", x: 260, y: 40, w: 170, label: "롤아웃", sub: "y, log π_old", kind: "data", desc: "업데이트 전 정책의 log-prob 을 `old_logp` 로 저장해 둡니다." },
      { id: "ref", x: 500, y: 20, w: 190, label: "Reference π_ref", sub: "log π_ref", kind: "frozen" },
      { id: "rm", x: 500, y: 120, w: 190, label: "Reward r_φ / 규칙", sub: "점수 R", kind: "frozen" },
      { id: "critic", x: 20, y: 230, w: 170, label: "Critic V", sub: "value head", kind: "trainable", desc: "각 토큰 위치의 기대 누적 보상 V(s_t) 를 예측합니다." },
      { id: "tokr", x: 760, y: 70, w: 220, label: "토큰별 보상 r_t", sub: "−β·KL_t (+R at EOS)", kind: "signal" },
      { id: "gae", x: 760, y: 200, w: 220, label: "GAE", sub: "A_t, returns", kind: "signal", desc: "δ_t = r_t + γV_{t+1} − V_t 를 거꾸로 누적합니다." },
      { id: "loss", x: 400, y: 290, w: 330, h: 64, label: "PPO 손실 × ppo_epochs", sub: "clip(ρ)·A  +  c·(V − returns)²", kind: "signal" },
    ],
    edges: [
      { from: "actor", to: "roll", label: "생성", dy: -20 },
      { from: "roll", to: "ref" },
      { from: "roll", to: "rm" },
      { from: "ref", to: "tokr", label: "KL", dy: -14 },
      { from: "rm", to: "tokr", label: "R", kind: "signal", dy: 12 },
      { from: "roll", to: "critic", label: "V(s_t)", via: [[345, 258]] },
      { from: "tokr", to: "gae" },
      { from: "critic", to: "gae", label: "values", via: [[105, 385], [870, 385]] },
      { from: "gae", to: "loss" },
      { from: "loss", to: "actor", kind: "grad", label: "∇ actor", via: [[230, 310], [230, 160], [105, 160]] },
      { from: "loss", to: "critic", kind: "grad", label: "∇ critic", via: [[300, 340], [300, 290]] },
    ],
    steps: [
      { caption: "Actor 가 응답을 생성하고, 그때의 log-prob 을 π_old 로 저장합니다.", nodes: ["actor", "roll"], edges: [0] },
      { caption: "얼린 참조 정책과 보상 모델(또는 규칙)이 KL 과 점수를 계산해 **토큰별 보상**을 만듭니다.", nodes: ["roll", "ref", "rm", "tokr"], edges: [1, 2, 3, 4] },
      { caption: "Critic 이 각 위치의 가치 V(s_t) 를 예측합니다.", nodes: ["roll", "critic"], edges: [5] },
      { caption: "토큰 보상과 가치로 GAE 를 계산해 **토큰별 advantage** 를 얻습니다.", nodes: ["tokr", "critic", "gae"], edges: [6, 7] },
      { caption: "같은 롤아웃으로 여러 epoch 동안 clip 손실과 value 손실을 최소화해 Actor 와 Critic 을 함께 업데이트합니다.", nodes: ["gae", "loss", "actor", "critic"], edges: [8, 9, 10] },
    ],
  },

  dpo_flow: {
    id: "dpo", title: "DPO 한 스텝", width: 1000, height: 300,
    nodes: [
      { id: "pair", x: 20, y: 110, w: 170, h: 64, label: "선호 쌍", sub: "(x, y_w, y_l)", kind: "data" },
      { id: "pi", x: 240, y: 40, w: 240, label: "π_θ", sub: "log π_θ(y_w), log π_θ(y_l)", kind: "trainable" },
      { id: "ref", x: 240, y: 190, w: 240, label: "π_ref", sub: "log π_ref(y_w), log π_ref(y_l)", kind: "frozen" },
      { id: "ratio", x: 540, y: 110, w: 210, h: 64, label: "암묵적 보상", sub: "β·log π_θ/π_ref  (w, l)", kind: "signal",
        desc: "언어 모델 자체가 보상 모델 역할을 합니다: r(x,y) = β log π_θ(y|x)/π_ref(y|x) + 상수." },
      { id: "loss", x: 800, y: 110, w: 180, h: 64, label: "−log σ(r_w − r_l)", sub: "Bradley–Terry", kind: "signal" },
    ],
    edges: [
      { from: "pair", to: "pi" },
      { from: "pair", to: "ref" },
      { from: "pi", to: "ratio" },
      { from: "ref", to: "ratio" },
      { from: "ratio", to: "loss", label: "margin", kind: "signal" },
      { from: "loss", to: "pi", kind: "grad", label: "∇ (샘플링 없음)", via: [[890, 20], [360, 20]], dx: 120 },
    ],
    steps: [
      { caption: "선호 쌍의 두 응답을 정책과 참조 정책에 모두 넣어 log-prob 을 구합니다. **생성은 하지 않습니다.**", nodes: ["pair", "pi", "ref"], edges: [0, 1] },
      { caption: "log-prob 비율로 각 응답의 암묵적 보상을 계산합니다.", nodes: ["pi", "ref", "ratio"], edges: [2, 3] },
      { caption: "chosen 의 암묵적 보상이 rejected 보다 커지도록 Bradley–Terry 손실을 최소화합니다.", nodes: ["ratio", "loss", "pi"], edges: [4, 5] },
    ],
  },

  grpo_flow: {
    id: "grpo", title: "GRPO 한 스텝", width: 1000, height: 330,
    nodes: [
      { id: "x", x: 20, y: 130, w: 140, label: "프롬프트 x", sub: "\"7+12=\"", kind: "data" },
      { id: "pi", x: 210, y: 130, w: 150, label: "π_θ", sub: "G 번 샘플", kind: "trainable" },
      { id: "ys", x: 410, y: 90, w: 170, h: 130, label: "y_1 … y_G", sub: "그룹", kind: "data", desc: "같은 프롬프트에서 뽑은 G 개 응답." },
      { id: "ver", x: 630, y: 40, w: 160, label: "검증기", sub: "정답이면 1", kind: "signal", desc: "보상 모델 대신 프로그램으로 채점 (RLVR)." },
      { id: "norm", x: 630, y: 150, w: 160, label: "그룹 정규화", sub: "(r − mean) / std", kind: "op", desc: "critic 이 하던 baseline 역할을 그룹 평균이 대신합니다." },
      { id: "adv", x: 840, y: 150, w: 140, label: "A_i", sub: "응답 전체에 동일", kind: "signal" },
      { id: "loss", x: 630, y: 255, w: 350, h: 56, label: "clip(ρ)·A_i − β·KL_k3(π_θ‖π_ref)", kind: "signal" },
      { id: "nocritic", x: 20, y: 255, w: 150, h: 56, label: "Critic 없음", sub: "메모리 절약", kind: "op", desc: "PPO 와 달리 가치 모델을 학습하지 않습니다." },
    ],
    edges: [
      { from: "x", to: "pi" },
      { from: "pi", to: "ys", label: "G 개", dy: -20 },
      { from: "ys", to: "ver", label: "r_1..r_G", dy: 16, dx: -10 },
      { from: "ver", to: "norm", kind: "signal" },
      { from: "norm", to: "adv", kind: "signal" },
      { from: "adv", to: "loss", kind: "signal" },
      { from: "loss", to: "pi", kind: "grad", via: [[600, 283], [600, 315], [285, 315]], label: "∇" },
    ],
    steps: [
      { caption: "프롬프트 하나에서 **G 개의 응답**을 샘플링합니다.", nodes: ["x", "pi", "ys"], edges: [0, 1] },
      { caption: "검증기로 각 응답을 채점합니다 (정답 1 / 오답 0).", nodes: ["ys", "ver"], edges: [2] },
      { caption: "그룹 안에서 평균을 빼고 표준편차로 나눠 advantage 를 만듭니다. 모두 같으면 A=0 → 학습 신호 없음.", nodes: ["ver", "norm", "adv"], edges: [3, 4] },
      { caption: "PPO 형 clip 목적 + KL 손실로 정책을 업데이트합니다. Critic 은 필요 없습니다.", nodes: ["adv", "loss", "pi", "nocritic"], edges: [5, 6] },
    ],
  },
};
