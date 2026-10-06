// LaTeX → MathML 변환기.
// 브라우저에 내장된 MathML 조판(Chrome 109+, Firefox, Safari)을 쓰므로 외부 라이브러리 없이
// 분수·첨자·합 기호·늘어나는 괄호가 제대로 그려진다. 학습 문서에 쓰는 LaTeX 범위를 지원한다.

const GREEK = {
  alpha: "α", beta: "β", gamma: "γ", delta: "δ", epsilon: "ϵ", varepsilon: "ε", zeta: "ζ", eta: "η",
  theta: "θ", iota: "ι", kappa: "κ", lambda: "λ", mu: "μ", nu: "ν", xi: "ξ", pi: "π", rho: "ρ",
  sigma: "σ", tau: "τ", phi: "ϕ", varphi: "φ", chi: "χ", psi: "ψ", omega: "ω",
  Gamma: "Γ", Delta: "Δ", Theta: "Θ", Lambda: "Λ", Pi: "Π", Sigma: "Σ", Phi: "Φ", Psi: "Ψ", Omega: "Ω",
  ell: "ℓ", infty: "∞", nabla: "∇", partial: "∂",
};

// 연산자·관계 기호 → <mo>
const OPERATORS = {
  cdot: "⋅", times: "×", pm: "±", sim: "∼", approx: "≈", propto: "∝", ne: "≠", neq: "≠",
  le: "≤", leq: "≤", ge: "≥", geq: "≥", ll: "≪", gg: "≫", succ: "≻", prec: "≺", in: "∈",
  to: "→", rightarrow: "→", leftarrow: "←", Rightarrow: "⇒", Leftrightarrow: "⇔", iff: "⟺",
  mid: "∣", lvert: "|", rvert: "|", "|": "‖", Vert: "‖", vert: "|", ldots: "…", dots: "…", cdots: "⋯", star: "⋆", ast: "∗",
  langle: "⟨", rangle: "⟩", lbrace: "{", rbrace: "}", "{": "{", "}": "}", "%": "%", "#": "#", "&": "&",
};

// 이름 있는 함수 (기울임 없이, 함수 적용 간격)
const FUNCTIONS = new Set(["log", "exp", "ln", "min", "max", "sin", "cos", "tanh", "arg", "lim", "det", "KL", "softmax", "clip", "mean", "std"]);
// display 모드에서 아래/위에 첨자가 붙는 큰 연산자
const LARGE = { sum: "∑", prod: "∏", min: "min", max: "max", lim: "lim", arg: "arg" };
const SPACES = { ",": "0.1667em", ":": "0.2222em", ";": "0.2778em", " ": "0.25em", quad: "1em", qquad: "2em", "!": "-0.1667em" };
const FONTS = {
  mathbb: { E: "𝔼", R: "ℝ", P: "ℙ", N: "ℕ", Z: "ℤ", Q: "ℚ", D: "𝔻" },
  mathcal: { L: "ℒ", D: "𝒟", J: "𝒥", R: "ℛ", P: "𝒫", N: "𝒩", O: "𝒪", E: "ℰ", F: "ℱ", H: "ℋ", M: "ℳ", B: "ℬ", S: "𝒮", A: "𝒜", T: "𝒯", X: "𝒳", Y: "𝒴" },
};
const BIG = { big: "1.2em", Big: "1.6em", bigg: "2.1em", Bigg: "2.6em", bigl: "1.2em", bigr: "1.2em", Bigl: "1.6em", Bigr: "1.6em" };
const ACCENTS = { hat: "^", widehat: "^", bar: "‾", overline: "‾", tilde: "~", widetilde: "~", vec: "→", dot: "˙" };

const escXml = (s) => s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function tokenize(src) {
  const out = [];
  const re = /\\([a-zA-Z]+|.)|(\d+(?:\.\d+)?)|([a-zA-Z])|(\s+)|(.)/gy;
  let m;
  while ((m = re.exec(src))) {
    if (m[1] !== undefined) out.push({ t: "cmd", v: m[1] });
    else if (m[2] !== undefined) out.push({ t: "num", v: m[2] });
    else if (m[3] !== undefined) out.push({ t: "letter", v: m[3] });
    else if (m[4] !== undefined) continue;
    else out.push({ t: "char", v: m[5] });
  }
  return out;
}

class Parser {
  constructor(tokens, display) {
    this.toks = tokens;
    this.i = 0;
    this.display = display;
  }
  peek() { return this.toks[this.i]; }
  next() { return this.toks[this.i++]; }

  // 그룹 또는 원자 하나를 인자로 읽는다
  arg() {
    const t = this.peek();
    if (!t) return "<mrow></mrow>";
    if (t.t === "char" && t.v === "{") {
      this.next();
      const body = this.expr("}");
      this.next(); // "}"
      return `<mrow>${body}</mrow>`;
    }
    return this.atom() ?? "<mrow></mrow>";
  }

  // 중괄호 안의 원문 텍스트 (\text{...} 용)
  rawGroup() {
    const t = this.peek();
    if (!(t && t.t === "char" && t.v === "{")) return this.next()?.v ?? "";
    this.next();
    let depth = 1, s = "";
    while (this.peek()) {
      const k = this.next();
      if (k.t === "char" && k.v === "{") depth++;
      if (k.t === "char" && k.v === "}" && --depth === 0) break;
      s += k.t === "cmd" ? (k.v.length === 1 && /[ ,;:]/.test(k.v) ? " " : "\\" + k.v) : k.v;
    }
    return s;
  }

  expr(stop) {
    let out = "";
    while (this.peek()) {
      const t = this.peek();
      if (stop && t.t === "char" && t.v === stop) break;
      if (t.t === "cmd" && t.v === "right") break;
      const a = this.atom();
      if (a == null) continue;
      out += this.scripts(a);
    }
    return out;
  }

  scripts(base) {
    let sub = null, sup = null;
    for (;;) {
      const t = this.peek();
      if (t && t.t === "char" && t.v === "_") { this.next(); sub = this.arg(); continue; }
      if (t && t.t === "char" && t.v === "^") { this.next(); sup = this.arg(); continue; }
      if (t && t.t === "char" && t.v === "'") { this.next(); sup = "<mo>′</mo>"; continue; }
      break;
    }
    if (!sub && !sup) return base;
    const under = this.display && base.includes('data-large="1"');
    if (under) {
      if (sub && sup) return `<munderover>${base}${sub}${sup}</munderover>`;
      return sub ? `<munder>${base}${sub}</munder>` : `<mover>${base}${sup}</mover>`;
    }
    if (sub && sup) return `<msubsup>${base}${sub}${sup}</msubsup>`;
    return sub ? `<msub>${base}${sub}</msub>` : `<msup>${base}${sup}</msup>`;
  }

  delim(t, size) {
    if (!t) return "";
    let v = t.t === "cmd" ? (OPERATORS[t.v] ?? (t.v === "|" ? "‖" : t.v)) : t.v;
    if (v === ".") return "";
    const attrs = size ? ` stretchy="true" symmetric="true" minsize="${size}" maxsize="${size}"` : ' stretchy="true"';
    return `<mo${attrs}>${escXml(v)}</mo>`;
  }

  atom() {
    const t = this.next();
    if (!t) return null;
    if (t.t === "num") return `<mn>${t.v}</mn>`;
    if (t.t === "letter") return `<mi>${t.v}</mi>`;
    if (t.t === "char") {
      const c = t.v;
      if (c === "{") { const body = this.expr("}"); this.next(); return `<mrow>${body}</mrow>`; }
      if (c === "}") return null;
      if (c === "-") return "<mo>−</mo>";
      if (c === "*") return "<mo>∗</mo>";
      if (c === "|") return '<mo stretchy="false">|</mo>';
      if ("()[]".includes(c)) return `<mo stretchy="false">${c}</mo>`;
      if (c === "~") return '<mspace width="0.25em"></mspace>';
      return `<mo>${escXml(c)}</mo>`;
    }
    // 명령
    const v = t.v;
    if (v in SPACES) return `<mspace width="${SPACES[v]}"></mspace>`;
    if (v in GREEK) return `<mi>${GREEK[v]}</mi>`;
    if (v === "frac" || v === "dfrac" || v === "tfrac") {
      const a = this.arg(), b = this.arg();
      const style = v === "tfrac" ? ' displaystyle="false"' : v === "dfrac" ? ' displaystyle="true"' : "";
      return `<mfrac${style}>${a}${b}</mfrac>`;
    }
    if (v === "sqrt") return `<msqrt>${this.arg()}</msqrt>`;
    if (v in FONTS) {
      const s = this.rawGroup();
      return `<mi mathvariant="normal">${escXml([...s].map((ch) => FONTS[v][ch] ?? ch).join(""))}</mi>`;
    }
    if (v === "text" || v === "textrm" || v === "mbox") return `<mtext>${escXml(this.rawGroup())}</mtext>`;
    if (v === "mathrm" || v === "operatorname" || v === "textbf" || v === "mathbf" || v === "mathit") {
      const s = this.rawGroup();
      const variant = v === "mathbf" || v === "textbf" ? "bold" : v === "mathit" ? "italic" : "normal";
      const isFn = v === "operatorname" || FUNCTIONS.has(s);
      return `${isFn ? '<mspace width="0.1667em"></mspace>' : ""}<mi mathvariant="${variant}">${escXml(s)}</mi>${isFn ? '<mo>&#x2061;</mo><mspace width="0.1667em"></mspace>' : ""}`;
    }
    if (v in LARGE) {
      if (v === "sum" || v === "prod") return `<mo largeop="true" movablelimits="true" data-large="1">${LARGE[v]}</mo>`;
      return `<mrow data-large="1"><mi>${LARGE[v]}</mi></mrow>`;
    }
    // LaTeX 처럼 함수 이름 앞뒤에 얇은 간격 (예: β log π)
    if (FUNCTIONS.has(v)) return `<mspace width="0.1667em"></mspace><mi>${v}</mi><mo>&#x2061;</mo><mspace width="0.1667em"></mspace>`;
    if (v in BIG) return this.delim(this.next(), BIG[v]);
    if (v === "left") {
      const open = this.delim(this.next());
      const body = this.expr();
      this.next(); // \right
      const close = this.delim(this.next());
      return `<mrow>${open}${body}${close}</mrow>`;
    }
    if (v in ACCENTS) return `<mover accent="true">${this.arg()}<mo>${ACCENTS[v]}</mo></mover>`;
    if (v === "underbrace") { const a = this.arg(); return `<munder>${a}<mo>⏟</mo></munder>`; }
    if (v in OPERATORS) return `<mo>${escXml(OPERATORS[v])}</mo>`;
    return `<mtext>${escXml(v)}</mtext>`; // 모르는 명령은 이름을 그대로 보여 준다
  }
}

export function texToMathML(tex, display = false) {
  const p = new Parser(tokenize(tex.trim()), display);
  const body = p.expr().replace(/ data-large="1"/g, "");
  const alt = escXml(tex.trim());
  return `<math${display ? ' display="block"' : ""} alttext="${alt}"><mrow>${body}</mrow></math>`;
}
