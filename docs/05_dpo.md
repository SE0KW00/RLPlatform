# 05. DPO — 보상 모델 없이 선호 최적화

코드: [`nlprl/algorithms/dpo.py`](../nlprl/algorithms/dpo.py)

## 1. 유도 — 네 단계

RLHF 는 "보상 모델 학습 → PPO" 두 단계였습니다. DPO 는 수식 변형으로 이 둘을 **한 번의 지도학습**으로 합칩니다.

**① 풀고 싶은 문제** — RLHF 와 같은 목표: 보상은 크게, 참조 정책과의 거리는 작게.

$$
\max_{\pi} \; \mathbb{E}_{x \sim \mathcal{D},\; y \sim \pi(\cdot \mid x)} \big[ r(x, y) \big] \;-\; \beta \, \mathrm{KL}\big( \pi(\cdot \mid x) \,\|\, \pi_{\mathrm{ref}}(\cdot \mid x) \big)
$$

**② 최적해는 닫힌 형태로 쓸 수 있다.**

$$
\pi^{*}(y \mid x) = \frac{1}{Z(x)} \, \pi_{\mathrm{ref}}(y \mid x) \, \exp\!\Big( \frac{r(x, y)}{\beta} \Big)
$$

> **읽는 법** — 최적 정책은 참조 정책을 보상이 높은 쪽으로 $\exp(r/\beta)$ 배만큼 **기울인** 분포다. $\beta$ 가 작을수록 더 많이 기울어진다. $Z(x) = \sum_y \pi_{\mathrm{ref}}(y \mid x) \exp(r(x, y)/\beta)$ 는 확률의 합을 1 로 맞추는 정규화 상수인데, 가능한 모든 문장 $y$ 에 대한 합이라 **직접 계산할 수 없다.**

**③ 이 식을 보상 $r$ 에 대해 정리한다.** (양변에 log 를 취하고 이항)

$$
r(x, y) = \beta \log \frac{\pi^{*}(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} + \beta \log Z(x)
$$

**④ Bradley–Terry 식(04장)에는 보상의 차이만 들어간다** → 두 응답에 똑같이 붙은 $\beta \log Z(x)$ 가 **소거**된다.

$$
r(x, y_w) - r(x, y_l) = \beta \log \frac{\pi^{*}(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi^{*}(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)}
$$

이제 $\pi^{*}$ 자리에 학습할 정책 $\pi_\theta$ 를 넣고 Bradley–Terry 손실을 그대로 쓰면 **DPO 손실**이 됩니다.

$$
\mathcal{L}_{\mathrm{DPO}}(\theta) = -\,\mathbb{E}_{(x,\, y_w,\, y_l)} \left[ \log \sigma\!\left( \beta \log \frac{\pi_\theta(y_w \mid x)}{\pi_{\mathrm{ref}}(y_w \mid x)} - \beta \log \frac{\pi_\theta(y_l \mid x)}{\pi_{\mathrm{ref}}(y_l \mid x)} \right) \right]
$$

즉 **언어 모델 자체가 암묵적 보상 모델**이 됩니다 ("Your Language Model is Secretly a Reward Model"). 응답 $y$ 의 암묵적 보상은

$$
\hat{r}_\theta(x, y) = \beta \log \frac{\pi_\theta(y \mid x)}{\pi_{\mathrm{ref}}(y \mid x)} = \beta \Big( \log \pi_\theta(y \mid x) - \log \pi_{\mathrm{ref}}(y \mid x) \Big)
$$

학습 시 필요한 것: 정책, 참조 정책, 선호 쌍. 샘플링도 critic 도 보상 모델도 없습니다.

| 수식 | 코드 (`dpo.py`) |
|---|---|
| $\log \pi_\theta(y \mid x)$ (응답 토큰 log-prob 의 합) | `sequence_logprob(policy, ...)` → `pi` |
| $\log \pi_{\mathrm{ref}}(y \mid x)$ | `sequence_logprob(ref, ...)` → `rf` |
| $\hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$ | `beta * ((pi_c - ref_c) - (pi_r - ref_r))` |
| $\mathcal{L}_{\mathrm{DPO}}$ | `-F.logsigmoid(logits).mean()` |

### 기울기가 말해 주는 것

$m = \hat{r}_\theta(x, y_w) - \hat{r}_\theta(x, y_l)$ 라 두면, 손실의 기울기는

$$
\nabla_\theta \mathcal{L}_{\mathrm{DPO}} = -\,\beta \; \sigma(-m) \; \Big[ \nabla_\theta \log \pi_\theta(y_w \mid x) - \nabla_\theta \log \pi_\theta(y_l \mid x) \Big]
$$

- 괄호 안: chosen 의 확률은 올리고 rejected 의 확률은 내리는 방향.
- 가중치 $\sigma(-m)$: 이미 잘 구분하는 쌍($m \gg 0$)은 거의 0 → **틀리고 있는 쌍에 집중**한다.

## 2. 로그 읽기

| 지표 | 의미 |
|---|---|
| `reward_acc` | 암묵적 보상 $\hat{r}_\theta$ 가 chosen 을 더 높게 평가하는 쌍의 비율 |
| `margin` | 암묵적 보상 차이 $m$ 의 평균 |
| `chosen_logp` / `rejected_logp` | 선택/거절 응답의 $\log \pi_\theta(y \mid x)$ |

초기에는 $\pi_\theta = \pi_{\mathrm{ref}}$ 라서 $m = 0$, 손실이 정확히 $-\log \sigma(0) = \log 2 \approx 0.693$ 입니다 (`test_dpo_loss_at_init_is_log2`).

## 실습

```bash
python -m nlprl reward-model --task sentiment     # pairs.jsonl 생성 (DPO 가 재사용)
python -m nlprl dpo --task sentiment              # 기본: β=0.1, lr=2e-5, 1 epoch
python -m nlprl dpo --task sentiment --lr 1e-4 --epochs 3 --name dpo_aggressive
```

참고 결과 (seed 0):

| 설정 | 긍정률 | 문법 정확도 |
|---|---|---|
| SFT | 0.48 | 0.98 |
| DPO β=0.1, lr=2e-5, 1 epoch | 0.82 | 0.87 |
| DPO β=0.1, lr=1e-4, 3 epoch | 0.80 | **0.06** |

공격적인 설정에서는 `rejected_logp` 가 −60 까지 떨어지는데 **`chosen_logp` 도 함께 떨어집니다** (−13 → −17).
DPO 는 두 응답의 *차이*만 키우면 되므로, 둘 다 확률을 낮추면서 분포 밖 이상한 문장에 확률을 몰아주는 현상이 생길 수 있습니다.
이는 실제 연구에서도 보고된 DPO 의 약점이며, 온라인 RL(PPO/GRPO)과의 중요한 차이입니다: **DPO 는 자기가 생성한 문장을 한 번도 평가받지 않습니다.**

## 연습문제

1. β 를 0.01, 0.1, 0.5, 1.0 으로 바꿔 보세요. β 는 PPO 의 `kl_coef` 와 같은 역할입니다. 결과가 그 해석과 맞나요?
2. **IPO** 손실 `(logits − 1/(2β))²` 를 구현해 보고 과적합 양상을 비교하세요.
3. **Online / iterative DPO**: DPO 1 epoch → 새 정책으로 선호 쌍 다시 생성 → 반복. `--init` 과 `--regen-pairs` 로 쉽게 해볼 수 있습니다. 문법 붕괴가 줄어드나요?
4. 손실에 chosen 응답의 SFT(NLL) 항을 더하면 (`RPO`) `chosen_logp` 가 떨어지는 현상이 사라지나요?
