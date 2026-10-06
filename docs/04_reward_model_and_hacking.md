# 04. 보상 모델과 Reward Hacking

코드: [`nlprl/preference.py`](../nlprl/preference.py), [`nlprl/model.py`](../nlprl/model.py) 의 `RewardModel`

## 1. 왜 보상 모델이 필요한가

"좋은 답변"은 규칙으로 쓸 수 없습니다. 대신 사람에게 **두 응답 중 어느 쪽이 나은지** 고르게 하면 훨씬 일관된 라벨을 얻을 수 있습니다.
이 비교 데이터로 응답에 점수(숫자 하나)를 매기는 모델 $r_\phi$ 를 학습합니다. 이때 쓰는 확률 모형이 **Bradley–Terry 모델**입니다.

$$
P(y_w \succ y_l \mid x) = \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big), \qquad \sigma(z) = \frac{1}{1 + e^{-z}}
$$

> **읽는 법** — 프롬프트 $x$ 에 대해 $y_w$ 가 $y_l$ 보다 낫다고 판단될 확률은, 두 응답의 **점수 차이**를 시그모이드에 넣은 값이다. 점수 차이가 0 이면 반반(0.5), 클수록 1 에 가깝다.

보상 모델은 이 확률이 실제 라벨과 맞도록, 즉 **선택된 응답의 점수가 더 높아지도록** 학습합니다.

$$
\mathcal{L}_{\mathrm{RM}}(\phi) = -\,\mathbb{E}_{(x,\, y_w,\, y_l)} \Big[ \log \sigma\big( r_\phi(x, y_w) - r_\phi(x, y_l) \big) \Big]
$$

| 기호 | 뜻 | 코드 |
|---|---|---|
| $y_w$ (winner) | 선택된 응답 (chosen) | `PreferencePair.chosen` |
| $y_l$ (loser) | 거절된 응답 (rejected) | `PreferencePair.rejected` |
| $y_w \succ y_l$ | "$y_w$ 가 $y_l$ 보다 낫다" | — |
| $r_\phi(x, y)$ | 보상 모델의 점수, $\phi$ 는 그 파라미터 | `rm(ids, mask)` |
| $\sigma$ | 시그모이드 함수 | `F.logsigmoid` |

점수의 **차이**만 학습되므로, 모든 점수에 같은 상수를 더해도 손실은 같습니다 — 보상 모델 점수의 절대값 자체에는 의미가 없습니다.

이 플랫폼에서는 SFT 모델로 프롬프트당 4개 응답을 샘플링하고, 규칙 보상으로 최고/최저를 골라 "사람 라벨"을 대신합니다
(`make_preference_pairs()`). `--label-noise 0.2` 로 라벨러의 실수를 흉내낼 수 있습니다.

보상 모델은 SFT 모델의 몸통을 복사하고 마지막 토큰 위에 `Linear(d, 1)` 을 얹어 만듭니다.

## 2. Reward hacking (보상 해킹)

> 측정치가 목표가 되면, 그것은 더 이상 좋은 측정치가 아니다. — Goodhart 의 법칙

보상은 우리가 원하는 것의 **대리 지표(proxy)** 일 뿐입니다. 정책은 대리 지표를 최대화하는 가장 쉬운 길을 찾습니다.

**실험 A — 규칙 보상의 허점.** `sentiment` 보상은 긍정 단어 개수만 셉니다. KL 을 약하게 걸면:

```bash
python -m nlprl train --task sentiment --algo ppo --kl-coef 0.05 --name hack
python -m nlprl sample --ckpt runs/sentiment/hack/policy.pt
# [reward=+7.0] the hotel was fun fun fun fun fun fun fun.
```

보상은 7점이지만 문법 정확도(`well_formed`)는 0 입니다. KL 제약은 "SFT 모델이 쓸 법한 문장에서 너무 벗어나지 마라"는 안전장치입니다.

**실험 B — 보상 모델 과최적화(overoptimization).** 학습된 보상 모델은 학습 데이터 분포 밖에서 틀린 점수를 줍니다.

```bash
python -m nlprl reward-model --task sentiment                    # 선호 쌍 2000개 + RM 학습 (val_acc ≈ 1.0)
python -m nlprl train --task sentiment --algo ppo --reward rm    # RM 점수로 PPO
```

참고 결과: 학습 중 **RM 점수(`score`)는 5.5** 까지 오르지만, **실제 규칙 보상(`eval/reward`)은 1.7**, 문법 정확도는 0.65 로 떨어집니다.
RM 이 본 적 없는 문장(예: `good and good`)에 과대한 점수를 주고, 정책이 그 틈을 파고든 것입니다.
Gao et al. (2022) "Scaling Laws for Reward Model Overoptimization" 이 대형 모델에서 보여 준 현상의 축소판입니다.

## 연습문제

1. 실험 B 에서 `score`(RM 점수)와 `eval/reward`(진짜 점수)를 같은 그래프에 그려 보세요. 둘이 갈라지기 시작하는 KL 값은 얼마인가요?
2. `--label-noise 0.1, 0.3` 으로 RM 을 학습하고, 검증 정확도와 PPO 결과가 어떻게 변하는지 보세요.
3. 보상 함수를 고쳐서 실험 A 의 해킹을 막아 보세요 (예: 문법이 틀리면 −1). 그러면 정책은 또 다른 허점을 찾을까요?
4. **앙상블 보상 모델**: 서로 다른 seed 로 RM 3개를 학습하고 점수의 최솟값(또는 평균 − 표준편차)을 보상으로 쓰면 과최적화가 줄어드나요?
