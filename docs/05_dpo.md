# 05. DPO — 보상 모델 없이 선호 최적화

코드: [`nlprl/algorithms/dpo.py`](../nlprl/algorithms/dpo.py)

## 1. 유도 (세 줄 요약)

1. KL 제약 보상 최대화 `max E[r] − β·KL(π‖π_ref)` 의 최적해는 닫힌 형태로 쓸 수 있다:
   $$\pi^*(y|x) = \frac{1}{Z(x)}\,\pi_{\text{ref}}(y|x)\,\exp\!\big(r(x,y)/\beta\big)$$
2. 이를 r 에 대해 풀면: $r(x,y) = \beta\log\frac{\pi^*(y|x)}{\pi_{\text{ref}}(y|x)} + \beta\log Z(x)$
3. Bradley–Terry 식에 넣으면 **Z(x) 가 소거**된다:
   $$\mathcal{L}_{DPO} = -\log\sigma\Big(\beta\Big[\log\tfrac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \log\tfrac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)}\Big]\Big)$$

즉 **언어 모델 자체가 암묵적 보상 모델**이 됩니다 ("Your Language Model is Secretly a Reward Model").
학습 시 필요한 것: 정책, 참조 정책, 선호 쌍. 샘플링도 critic 도 보상 모델도 없습니다.

## 2. 로그 읽기

| 지표 | 의미 |
|---|---|
| `reward_acc` | 암묵적 보상 β·log(π/π_ref) 이 chosen 을 더 높게 평가하는 비율 |
| `margin` | 암묵적 보상 차이의 평균 |
| `chosen_logp` / `rejected_logp` | 선택/거절 응답의 log π_θ(y∣x) |

초기에는 π_θ = π_ref 이므로 손실이 정확히 log 2 ≈ 0.693 입니다 (`test_dpo_loss_at_init_is_log2`).

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
