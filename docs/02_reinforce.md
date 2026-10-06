# 02. REINFORCE — 정책 경사의 출발점

코드: [`nlprl/algorithms/reinforce.py`](../nlprl/algorithms/reinforce.py)

## 1. 핵심 아이디어

보상 함수 R 은 "문자열 → 숫자" 라서 미분할 수 없습니다. 그래도 **기대 보상의 기울기**는 구할 수 있습니다 (log-derivative trick):

$$
\nabla_\theta\,\mathbb{E}_{y\sim\pi_\theta}[R(y)] = \mathbb{E}_{y\sim\pi_\theta}\big[R(y)\,\nabla_\theta \log\pi_\theta(y)\big]
$$

해석: **보상이 높았던 응답의 확률은 올리고, 낮았던 응답의 확률은 내린다.**
이는 "보상으로 가중한 최대우도 학습"과 같습니다. 코드로는 한 줄입니다.

```python
loss = -(advantage * seq_logp).mean()     # seq_logp = Σ_t log π_θ(y_t | x, y_<t)
```

## 2. Baseline 으로 분산 줄이기

보상이 모두 양수라면 모든 응답의 확률을 올리게 되고, 신호는 "얼마나 더" 올리느냐의 차이뿐입니다 → 분산이 큼.
상수 b 를 빼도 기댓값은 그대로입니다 ( `E[∇ log π] = 0` 이므로). 그래서

$$ A = R - b $$

를 advantage 로 씁니다. 이 코드는 세 가지 baseline 을 제공합니다 (`--baseline`):

- `none` : b = 0
- `batch_mean` : 같은 배치의 평균 보상 (기본값; RLOO·GRPO 의 원형)
- `ema` : 지수이동평균

## 3. KL 벌점

```python
kl = ((logp - ref_logp) * amask).sum(-1)   # log π_θ(y|x) - log π_ref(y|x), 샘플 기반 KL 추정치
total_reward = score - kl_coef * kl
```

y 가 π_θ 에서 샘플링되었으므로 `log π_θ(y) − log π_ref(y)` 의 기댓값이 정확히 KL(π_θ‖π_ref) 입니다.
KL 을 보상의 일부로 다루는 것이 InstructGPT 이후의 표준입니다.

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
