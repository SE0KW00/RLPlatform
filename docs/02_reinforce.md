# 02. REINFORCE — 정책 경사의 출발점

코드: [`nlprl/algorithms/reinforce.py`](../nlprl/algorithms/reinforce.py)

## 1. 핵심 아이디어

보상 함수 $R$ 은 문자열을 받아 숫자를 돌려줄 뿐이라 미분할 수 없습니다. 그래도 **기대 보상의 기울기**는 구할 수 있습니다.

$$
\nabla_\theta \, \mathbb{E}_{y \sim \pi_\theta} \big[ R(y) \big] = \mathbb{E}_{y \sim \pi_\theta} \big[ R(y) \, \nabla_\theta \log \pi_\theta(y) \big]
$$

> **읽는 법** — 기대 보상을 키우는 방향은, 샘플한 응답의 log 확률을 키우는 방향 $\nabla_\theta \log \pi_\theta(y)$ 에 **그 응답의 보상 $R(y)$ 를 가중치로 곱해 평균** 낸 것이다.

**왜 성립하나? (세 단계)**

1. 기댓값을 합으로 풀어 쓰면, $R$ 은 $\theta$ 와 무관하므로 미분 밖으로 나옵니다: $\nabla_\theta \sum_y \pi_\theta(y) R(y) = \sum_y R(y) \, \nabla_\theta \pi_\theta(y)$
2. log 의 미분 $\nabla \log \pi = \frac{\nabla \pi}{\pi}$ 를 뒤집으면 $\nabla_\theta \pi_\theta(y) = \pi_\theta(y) \, \nabla_\theta \log \pi_\theta(y)$ 입니다.
3. 대입하면 $\sum_y \pi_\theta(y) \, R(y) \, \nabla_\theta \log \pi_\theta(y)$ — 다시 $\pi_\theta$ 에 대한 기댓값이 되므로 **샘플 평균으로 근사**할 수 있습니다.

결론: **보상이 높았던 응답의 확률은 올리고, 낮았던 응답의 확률은 내린다.** "보상으로 가중한 최대우도 학습"과 같습니다.

| 수식 | 코드 (`reinforce.py`) |
|---|---|
| $\log \pi_\theta(y \mid x) = \sum_t \log \pi_\theta(y_t \mid s_t)$ | `seq_logp = (logp * amask).sum(-1)` |
| $A = R - b$ (advantage, 아래 2절) | `adv = total_reward - b` |
| $-\frac{1}{B} \sum_{i=1}^{B} A_i \log \pi_\theta(y_i \mid x_i)$ | `loss = -(adv * seq_logp).mean()` |

손실에 마이너스가 붙는 이유: 옵티마이저는 손실을 **줄이는** 방향으로 움직이므로, 키우고 싶은 값에 −1 을 곱합니다.

## 2. Baseline 으로 분산 줄이기

보상이 모두 양수라면 모든 응답의 확률을 올리게 되고, 신호는 "얼마나 더" 올리느냐의 차이뿐입니다 → 기울기의 분산이 큽니다. 그래서 보상에서 기준값 $b$ 를 뺀 **advantage** 를 씁니다.

$$
A = R - b
$$

상수 $b$ 를 빼도 기울기의 **기댓값은 변하지 않습니다**. 확률의 합은 항상 1 이라 그 미분이 0 이기 때문입니다.

$$
\mathbb{E}_{y \sim \pi_\theta} \big[ \nabla_\theta \log \pi_\theta(y) \big] = \sum_y \nabla_\theta \pi_\theta(y) = \nabla_\theta \sum_y \pi_\theta(y) = \nabla_\theta \, 1 = 0
$$

따라서 $\mathbb{E}\big[(R - b) \nabla_\theta \log \pi_\theta\big] = \mathbb{E}\big[R \, \nabla_\theta \log \pi_\theta\big] - b \cdot 0$ 입니다. 평균은 그대로, 흔들림만 줄어듭니다.

이 코드는 세 가지 baseline 을 제공합니다 (`--baseline`):

- `none` : $b = 0$
- `batch_mean` : 같은 배치의 평균 보상 (기본값; RLOO·GRPO 의 원형)
- `ema` : 지수이동평균

## 3. KL 벌점

참조 정책과의 거리는 **샘플한 응답 위에서** 두 정책의 log 확률 차이를 더해 추정합니다.

$$
\widehat{\mathrm{KL}}(x, y) = \sum_{t} \Big( \log \pi_\theta(y_t \mid s_t) - \log \pi_{\mathrm{ref}}(y_t \mid s_t) \Big)
$$

$y$ 가 $\pi_\theta$ 에서 샘플링되었으므로 이 값의 기댓값이 정확히 $\mathrm{KL}(\pi_\theta \,\|\, \pi_{\mathrm{ref}})$ 입니다. 그리고 KL 을 **보상의 일부**로 다룹니다 (InstructGPT 이후의 표준).

$$
R'(x, y) = R(x, y) - \beta \, \widehat{\mathrm{KL}}(x, y)
$$

```python
kl = ((logp - ref_logp) * amask).sum(-1)   # KL 추정치
total_reward = score - kl_coef * kl        # R' = R − β·KL
```

## 실습

```bash
python -m nlprl train --task sentiment --algo reinforce
python -m nlprl train --task sentiment --algo reinforce --baseline none --name reinforce_nobase
python -m nlprl plot runs/sentiment/reinforce/history.json runs/sentiment/reinforce_nobase/history.json \
    --metric eval/positive_rate --out baseline.png
```

참고 결과 (seed 0, CPU 약 50초): 긍정률 0.48 → **0.97**, 문법 정확도 0.96 유지.

로그 읽는 법:

| 지표 | 의미 |
|---|---|
| `score` | 학습 배치의 평균 보상 (KL 벌점 제외) |
| `kl` | 참조 정책과의 시퀀스 KL. 계속 커지면 정책이 멀리 이동 중 |
| `eval/positive_rate` | 긍정 리뷰 비율 (우리가 진짜 원하는 것) |
| `eval/well_formed` | 문법에 맞는 문장 비율 (reward hacking 감시용) |
| `eval/distinct` | 서로 다른 응답의 비율 (mode collapse 감시용) |

## 연습문제

1. `--baseline none` 과 `batch_mean` 의 학습 곡선을 비교하세요. 학습 속도와 안정성이 어떻게 다른가요?
2. `--kl-coef 0` 으로 학습하면 무슨 일이 일어나나요? `sample` 로 결과 문장을 직접 확인하세요. (힌트: 04장)
3. **RLOO** 를 구현해 보세요: 각 프롬프트에서 k 개를 샘플링하고, 응답 i 의 baseline 을 "나머지 k−1 개 보상의 평균"으로 둡니다.
   GRPO 의 `group_advantages()` 와 무엇이 다른가요?
4. 현재 손실은 시퀀스의 log-prob 합에 advantage 를 곱합니다. 긴 응답일수록 기울기가 커지는데, 이것이 응답 길이에 어떤 편향을 줄 수 있을까요?
