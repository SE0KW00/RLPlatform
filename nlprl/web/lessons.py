"""웹 레슨 정의: 개념 문서 + 시각화 위젯 + 코드 워크스루.

코드 워크스루의 각 단계는 실제 소스 파일의 구간을 가리킨다. 줄 번호 대신
`start` / `end` 앵커 문자열로 구간을 찾기 때문에, 코드가 조금 바뀌어도 설명이 어긋나지 않는다.
  - 앵커는 해당 줄에 포함된 부분 문자열이다.
  - "^" 로 시작하면 줄 전체(오른쪽 공백 제외)가 정확히 일치해야 한다.
  - `end` 는 `start` 줄부터 아래로 검색한다. 생략하면 한 줄짜리 구간.
  - `after` 를 주면 그 앵커가 나온 뒤부터 `start` 를 찾는다 (같은 줄이 여러 번 나올 때).
`tests/test_web.py` 가 모든 앵커가 실제로 찾아지는지 검사한다.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

LESSONS = [
    {
        "id": "overview",
        "num": "00",
        "title": "전체 구조와 SFT",
        "subtitle": "사전학습된 정책에서 출발해 RL 로 조정하기까지의 큰 그림",
        "doc": "docs/00_roadmap.md",
        "widgets": ["pipeline", "status"],
        "walkthrough": [
            {"file": "nlprl/pretrain.py", "start": "def lm_loss(", "end": "return (loss * m).sum() / m.sum()",
             "title": "SFT 손실 = 다음 토큰 교차 엔트로피",
             "body": "RL 이전 단계는 평범한 언어 모델 학습입니다. `logits[:, :-1]` 이 `ids[:, 1:]` 을 예측하도록 "
                     "교차 엔트로피를 계산하고, 패딩 위치(`mask=0`)는 손실에서 뺍니다.\n\n"
                     "이렇게 만든 SFT 모델이 **RL 의 출발점(π_θ 초기값)** 이자 **KL 제약의 기준(π_ref)** 이 됩니다."},
            {"file": "nlprl/pretrain.py", "start": "    for step in range(1, steps + 1):", "end": "        sched.step()",
             "title": "학습 루프",
             "body": "과제의 `corpus()` 에서 문장을 뽑아 `BOS + 문장 + EOS` 로 인코딩하고 오른쪽 패딩합니다. "
                     "그 뒤로는 일반적인 `zero_grad → backward → clip → step` 입니다."},
            {"file": "nlprl/algorithms/common.py", "start": "def make_reference(", "end": "    return ref",
             "title": "참조 정책 π_ref 만들기",
             "body": "RL 을 시작할 때 정책을 **깊은 복사해서 얼려 둡니다** (`requires_grad_(False)`). "
                     "이후 모든 알고리즘은 `log π_θ − log π_ref` 로 '처음에서 얼마나 멀어졌는지'를 측정합니다."},
            {"file": "nlprl/pipeline.py", "start": "DEFAULTS = {", "end": "^}",
             "title": "과제별 기본 하이퍼파라미터",
             "body": "CLI 와 웹 실험실이 같은 기본값을 씁니다. `sentiment` 는 reward hacking 을 막기 위해 "
                     "`kl_coef=0.3`, `arithmetic` 은 검증 가능한 보상이라 KL 없이(`0.0`) 학습합니다."},
        ],
    },
    {
        "id": "mdp",
        "num": "01",
        "title": "텍스트 생성을 RL 로 보기",
        "subtitle": "상태 = 지금까지의 토큰, 행동 = 다음 토큰, 보상 = 응답이 끝난 뒤 한 번",
        "doc": "docs/01_text_generation_as_rl.md",
        "widgets": ["mdp_loop", "trace", "tensors"],
        "walkthrough": [
            {"file": "nlprl/tasks.py", "start": "class SentimentTask(Task):", "end": "return float(self.sentiment_score(response))",
             "title": "환경 = 과제 + 보상 함수",
             "body": "RL 의 '환경'은 프롬프트를 주고, 응답이 끝나면 점수를 돌려주는 역할만 합니다. "
                     "`sentiment` 의 보상은 **긍정 단어 수 − 부정 단어 수** 로, 문법은 전혀 보지 않습니다. "
                     "이 허점이 레슨 04 의 reward hacking 으로 이어집니다."},
            {"file": "nlprl/generation.py", "start": "    prompt_ids = [tok.encode(p, add_bos=True) for p in prompts]",
             "end": "    resp_mask = torch.zeros_like(mask)",
             "title": "왼쪽 패딩으로 배치 만들기",
             "body": "프롬프트 길이가 달라도 **마지막 열이 항상 '다음 토큰을 예측할 위치'** 가 되도록 왼쪽에 패딩합니다. "
                     "`block_size` 를 넘지 않는지도 확인합니다."},
            {"file": "nlprl/generation.py", "start": "    for _ in range(max_new_tokens):", "end": "            break",
             "title": "에피소드 = 토큰을 하나씩 샘플링",
             "body": "매 스텝 `logits[:, -1, :]` → softmax → `multinomial` 로 **행동(다음 토큰)** 을 고릅니다. "
                     "이미 EOS 를 낸 행은 `pad` 로 채우고 `attention_mask=0` 으로 표시합니다. "
                     "`resp_mask` 는 '정책이 직접 고른 토큰'만 1 이 됩니다."},
            {"file": "nlprl/model.py", "start": "def position_ids_from_mask(", "end": "return (attention_mask.long().cumsum(-1) - 1).clamp(min=0)",
             "title": "왼쪽 패딩이 있어도 위치는 0 부터",
             "body": "`cumsum(mask) − 1` 로 실제 토큰의 위치를 다시 셉니다. 이게 없으면 같은 문장도 패딩 길이에 따라 "
                     "다른 확률을 갖게 됩니다 (`test_left_padding_does_not_change_logprobs` 가 검증)."},
            {"file": "nlprl/generation.py", "start": "def token_logprobs(", "end": "^    return lp",
             "title": "log π(a_t | s_t) 계산",
             "body": "모든 RL 알고리즘의 핵심 재료입니다. 한 번의 forward 로 모든 위치의 분포를 얻고, "
                     "`gather` 로 **실제로 나온 다음 토큰**의 log-prob 만 뽑습니다. 결과는 길이 `L−1` 이므로 "
                     "행동 마스크는 `response_mask[:, 1:]` 입니다."},
        ],
    },
    {
        "id": "reinforce",
        "num": "02",
        "title": "REINFORCE",
        "subtitle": "보상이 높았던 응답의 확률은 올리고, 낮았던 응답의 확률은 내린다",
        "doc": "docs/02_reinforce.md",
        "widgets": ["reinforce_flow", "bandit"],
        "walkthrough": [
            {"file": "nlprl/algorithms/reinforce.py", "start": "        prompts = task.sample_prompts(cfg.batch_size, rng)",
             "end": "        amask = roll.action_mask",
             "title": "① 롤아웃과 보상",
             "body": "현재 정책으로 배치를 샘플링하고 과제 보상(또는 보상 모델 점수)을 받습니다. 이 단계에는 기울기가 없습니다."},
            {"file": "nlprl/algorithms/reinforce.py", "start": "        logp = token_logprobs(policy, roll.input_ids, roll.attention_mask)",
             "end": "            total_reward = score - cfg.kl_coef * kl",
             "title": "② KL 벌점을 보상에 섞기",
             "body": "`logp` 는 기울기가 필요하므로 `no_grad` 밖에서 계산합니다. 샘플 y 가 π_θ 에서 나왔으므로 "
                     "`Σ_t (log π_θ − log π_ref)` 의 기댓값이 정확히 KL(π_θ‖π_ref) 입니다. "
                     "최종 보상은 `R − β·KL`."},
            {"file": "nlprl/algorithms/reinforce.py", "start": "            if baseline == \"batch_mean\":", "end": "            adv = total_reward - b",
             "title": "③ baseline 으로 분산 줄이기",
             "body": "상수 b 를 빼도 기울기의 기댓값은 그대로입니다 (`E[∇log π] = 0`). 하지만 분산이 크게 줄어듭니다. "
                     "`batch_mean` 은 GRPO·RLOO 의 원형입니다. 시각화 탭의 시뮬레이터에서 baseline 을 켜고 꺼 보세요."},
            {"file": "nlprl/algorithms/reinforce.py", "start": "        seq_logp = (logp * amask).sum(-1)", "end": "        opt.step()",
             "title": "④ 정책 경사 = 보상 가중 최대우도",
             "body": "`loss = −(A · log π_θ(y|x)).mean()`. A>0 이면 그 응답의 확률을 올리고, A<0 이면 내립니다. "
                     "이 한 줄이 REINFORCE 의 전부입니다."},
        ],
    },
    {
        "id": "ppo",
        "num": "03",
        "title": "PPO 와 RLHF",
        "subtitle": "critic 으로 토큰별 credit assignment, clipping 으로 안전한 업데이트",
        "doc": "docs/03_ppo_rlhf.md",
        "widgets": ["ppo_flow", "gae", "clip"],
        "walkthrough": [
            {"file": "nlprl/algorithms/ppo.py", "start": "class PPOConfig(RLConfig):", "end": "    whiten_advantages: bool = True",
             "title": "PPO 의 하이퍼파라미터",
             "body": "`ppo_epochs` 번 같은 롤아웃을 재사용하고, `clip_range`(ε) 로 한 번에 움직일 수 있는 폭을 제한합니다. "
                     "LLM 에서는 보통 γ=1, λ=0.95."},
            {"file": "nlprl/model.py", "start": "        self.value_head = nn.Linear(cfg.n_embd, 1) if with_value_head else None",
             "title": "critic = value head",
             "body": "정책과 같은 Transformer 몸통 위에 `Linear(d, 1)` 을 얹어 **각 위치의 상태 가치 V(s_t)** 를 예측합니다. "
                     "대형 모델에서는 critic 을 별도 모델로 두는 경우가 많습니다."},
            {"file": "nlprl/algorithms/ppo.py", "start": "            old_logp, old_values = token_logprobs(policy, ids, attn, return_values=True)",
             "end": "            rewards[torch.arange(len(last)), last] += score",
             "title": "토큰 단위 보상 만들기",
             "body": "**모든 응답 토큰**에 `−β·(log π_old − log π_ref)` 를, **마지막 토큰**에만 점수 R 을 더합니다. "
                     "그래서 critic 은 '앞으로 받을 점수 − 앞으로 받을 KL 벌점'을 배우게 됩니다."},
            {"file": "nlprl/algorithms/ppo.py", "start": "def compute_gae(", "end": "    return adv, returns",
             "title": "GAE: 토큰별 advantage",
             "body": "뒤에서부터 `δ_t = r_t + γV(s_{t+1}) − V(s_t)`, `A_t = δ_t + γλA_{t+1}` 을 계산합니다. "
                     "응답이 끝난 뒤(mask=0)의 가치는 0 으로 둡니다. 시각화 탭에서 γ, λ 를 바꿔 보세요."},
            {"file": "nlprl/algorithms/ppo.py", "start": "                # --- clipped surrogate objective ---",
             "end": "                pg_loss = masked_mean(torch.max(pg1, pg2), m)",
             "title": "clipped surrogate objective",
             "body": "`ρ = exp(logp − old_logp)`. 손실은 `max(−Aρ, −A·clip(ρ))` 로, **이득이 생기는 방향으로만** clip 이 걸립니다. "
                     "시각화 탭의 그래프에서 A>0 / A<0 일 때 평평해지는 구간을 확인하세요."},
            {"file": "nlprl/algorithms/ppo.py", "start": "                # --- clipped value loss ---",
             "end": "                loss = pg_loss + cfg.vf_coef * vf_loss",
             "title": "value loss (역시 clip)",
             "body": "critic 은 GAE 로 만든 `returns` 를 회귀합니다. 값도 `old_values ± ε` 범위로 clip 해 급격한 변화를 막습니다."},
            {"file": "nlprl/algorithms/ppo.py", "start": "                    stats[\"clipfrac\"]",
             "end": "                    stats[\"approx_kl\"]",
             "title": "감시 지표: clipfrac, approx_kl",
             "body": "`clipfrac` 이 0.3 을 넘거나 `approx_kl` 이 계속 커지면 학습률이 너무 크거나 epoch 가 너무 많다는 신호입니다."},
        ],
    },
    {
        "id": "reward",
        "num": "04",
        "title": "보상 모델과 Reward Hacking",
        "subtitle": "사람 선호 → Bradley–Terry 보상 모델, 그리고 정책이 그 허점을 파고드는 순간",
        "doc": "docs/04_reward_model_and_hacking.md",
        "widgets": ["bt_curve", "rm_playground", "hacking"],
        "walkthrough": [
            {"file": "nlprl/preference.py", "start": "        for g in range(len(uniq)):",
             "end": "            pairs.append(PreferencePair(uniq[g], c, r))",
             "title": "선호 쌍 만들기 (사람 라벨러 대역)",
             "body": "프롬프트마다 4개를 샘플링해 규칙 보상 최고/최저를 (chosen, rejected) 로 묶습니다. "
                     "점수가 같으면 버리고, `label_noise` 확률로 뒤집어 라벨러 실수를 흉내냅니다."},
            {"file": "nlprl/model.py", "start": "class RewardModel(nn.Module):", "end": "        return self.score_head(h_last).squeeze(-1)",
             "title": "보상 모델 구조",
             "body": "GPT 몸통 + **마지막 실제 토큰** 의 hidden 위에 `Linear(d, 1)`. 문장 전체를 읽은 뒤 스칼라 하나를 냅니다."},
            {"file": "nlprl/model.py", "start": "def last_token_index(", "end": "    return T - 1 - attention_mask.long().flip(-1).argmax(-1)",
             "title": "마지막 토큰 찾기",
             "body": "마스크를 뒤집어 처음 나오는 1 의 위치를 찾으면 왼쪽/오른쪽 패딩 모두에서 마지막 실제 토큰 인덱스가 됩니다."},
            {"file": "nlprl/preference.py", "start": "    def batch_loss(batch):",
             "end": "        return -F.logsigmoid(rc - rr).mean(), (rc > rr).float().mean()",
             "title": "Bradley–Terry 손실",
             "body": "`−log σ(r(x,y_w) − r(x,y_l))`. 점수의 **차이** 만 학습하므로 보상 모델 점수의 절대값(오프셋)은 의미가 없습니다."},
            {"file": "nlprl/algorithms/common.py", "start": "def reward_model_fn(", "end": "    return fn",
             "title": "RM 점수를 RL 보상으로",
             "body": "롤아웃의 `input_ids` 를 그대로 보상 모델에 넣습니다. 이 함수를 `--reward rm` 으로 PPO 에 꽂으면 RLHF 가 완성됩니다."},
            {"file": "nlprl/tasks.py", "start": "    def sentiment_score(text: str) -> int:", "end": "return sum(w in POSITIVE for w in words)",
             "title": "해킹당하는 보상",
             "body": "긍정 단어를 **세기만** 하므로 `fun fun fun fun` 이 최고 점수입니다. KL 이 약하면 정책은 정확히 이 길을 찾아냅니다."},
        ],
    },
    {
        "id": "dpo",
        "num": "05",
        "title": "DPO",
        "subtitle": "보상 모델도 샘플링도 없이, 선호 쌍으로 정책을 직접 학습",
        "doc": "docs/05_dpo.md",
        "widgets": ["dpo_flow", "dpo_curve", "dpo_pairs"],
        "walkthrough": [
            {"file": "nlprl/preference.py", "start": "def encode_pairs(", "end": "    return ids, mask, rmask",
             "title": "chosen / rejected 를 한 배치로",
             "body": "`[chosen N개; rejected N개]` 를 쌓아 forward 한 번으로 처리합니다. `rmask` 는 응답 토큰(+EOS)만 1."},
            {"file": "nlprl/algorithms/dpo.py", "start": "def sequence_logprob(", "end": "    return (lp * rmask[:, 1:].float()).sum(-1)",
             "title": "log π(y|x)",
             "body": "응답 토큰들의 log-prob 합. 정책과 참조 정책 각각에 대해 계산합니다."},
            {"file": "nlprl/algorithms/dpo.py", "start": "def dpo_loss(", "end": "    return loss, chosen_reward, rejected_reward",
             "title": "DPO 손실",
             "body": "암묵적 보상 `β·log(π_θ/π_ref)` 의 차이를 Bradley–Terry 에 넣은 것. 처음엔 π_θ=π_ref 라 손실이 정확히 log 2 입니다. "
                     "시각화 탭에서 β 에 따른 손실과 기울기 가중치 σ(−margin) 를 보세요."},
            {"file": "nlprl/algorithms/dpo.py", "start": "            ids, mask, rmask = encode_pairs(tok, batch)",
             "end": "            loss, rc, rr = dpo_loss(pi[:n], pi[n:], rf[:n], rf[n:], cfg.beta)",
             "title": "학습 루프",
             "body": "정책은 기울기와 함께, 참조 정책은 `no_grad` 로 계산합니다. 롤아웃이 없어서 지도학습처럼 빠릅니다 — "
                     "대신 **자기가 생성한 문장을 한 번도 평가받지 않습니다.**"},
        ],
    },
    {
        "id": "grpo",
        "num": "06",
        "title": "GRPO 와 RLVR",
        "subtitle": "critic 대신 같은 프롬프트의 G 개 응답끼리 비교한다",
        "doc": "docs/06_grpo_rlvr.md",
        "widgets": ["grpo_flow", "grpo_group"],
        "walkthrough": [
            {"file": "nlprl/tasks.py", "after": "class ArithmeticTask(Task):", "start": "    def reward(self, prompt, response):", "end": "        return 1.0 if response == f\"{a + b}.\" else 0.0",
             "title": "검증 가능한 보상 (RLVR)",
             "body": "정답이면 1, 아니면 0. 보상 모델이 없으니 과최적화도 없습니다. 수학·코드 추론 모델 학습의 기본 형태입니다."},
            {"file": "nlprl/algorithms/grpo.py", "start": "class GRPOConfig(RLConfig):", "end": "    loss_type: str",
             "title": "GRPO 설정과 변형",
             "body": "`group_size`(G) 가 핵심입니다. `scale_rewards=False` 는 Dr. GRPO, `loss_type=\"token\"` 은 DAPO 의 제안입니다."},
            {"file": "nlprl/algorithms/grpo.py", "start": "def group_advantages(", "end": "    return adv.view(-1)",
             "title": "그룹 상대 advantage",
             "body": "`[B] → [B/G, G]` 로 바꿔 그룹 평균을 빼고 표준편차로 나눕니다. critic 이 할 일을 그룹 통계가 대신합니다. "
                     "그룹 보상이 모두 같으면 advantage 가 0 → **학습 신호 없음**."},
            {"file": "nlprl/algorithms/grpo.py", "start": "        uniq = task.sample_prompts(cfg.batch_size // cfg.group_size, rng)",
             "end": "        prompts = [p for p in uniq for _ in range(cfg.group_size)]",
             "title": "프롬프트를 G 번 반복",
             "body": "같은 프롬프트를 G 번 넣어 서로 다른 응답 G 개를 얻습니다."},
            {"file": "nlprl/algorithms/grpo.py", "start": "            ratio = torch.exp(logp - old_logp)",
             "end": "            kl = torch.exp(log_r) - log_r - 1  # k3",
             "title": "clip 목적 + k3 KL",
             "body": "PPO 와 같은 clip 목적에, KL 은 보상이 아니라 **손실에 직접** 더합니다. k3 추정치 "
                     "`e^x − x − 1` (x = log π_ref − log π_θ) 은 항상 0 이상이고 분산이 작습니다."},
            {"file": "nlprl/algorithms/grpo.py", "start": "            per_token = -(surr - cfg.kl_coef * kl)",
             "end": "                loss = masked_mean(per_token, amask)",
             "title": "손실 집계 방식",
             "body": "원래 GRPO 는 응답마다 토큰 평균 → 응답 평균. 이 방식은 긴 응답의 토큰 하나하나를 덜 반영하는 길이 편향이 있어 "
                     "DAPO 는 배치 전체 토큰 평균을 제안했습니다."},
        ],
    },
    {
        "id": "next",
        "num": "07",
        "title": "더 나아가기",
        "subtitle": "실제 LLM 규모로 확장할 때 바뀌는 것과 읽을거리",
        "doc": "docs/07_next_steps.md",
        "widgets": [],
        "walkthrough": [],
    },
]


def _find(lines: list[str], anchor: str, start: int = 0) -> int:
    exact = anchor.startswith("^")
    a = anchor[1:] if exact else anchor.rstrip("\n")
    for i in range(start, len(lines)):
        if (lines[i].rstrip() == a) if exact else (a in lines[i]):
            return i
    raise ValueError(f"anchor not found: {anchor!r}")


def resolve_step(step: dict) -> dict:
    """앵커 → 1부터 시작하는 줄 번호 [line_start, line_end]."""
    lines = (ROOT / step["file"]).read_text(encoding="utf-8").splitlines()
    s = _find(lines, step["start"], _find(lines, step["after"]) if step.get("after") else 0)
    e = _find(lines, step["end"], s) if step.get("end") else s
    return {**step, "line_start": s + 1, "line_end": e + 1}


def lessons_index() -> list[dict]:
    return [{k: l[k] for k in ("id", "num", "title", "subtitle", "doc")} for l in LESSONS]


def lesson(lesson_id: str) -> dict:
    l = next(x for x in LESSONS if x["id"] == lesson_id)
    return {**l, "walkthrough": [resolve_step(s) for s in l["walkthrough"]]}
