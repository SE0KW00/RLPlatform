# 03. PPO 와 RLHF

코드: [`nlprl/algorithms/ppo.py`](../nlprl/algorithms/ppo.py)

ChatGPT 를 만든 InstructGPT 파이프라인은 **SFT → 보상 모델 → PPO** 입니다. 이 장은 PPO 부분입니다.

## 1. REINFORCE 의 세 가지 문제와 PPO 의 해법

| 문제 | PPO 의 해법 | 코드 |
|---|---|---|
| 롤아웃 한 번에 업데이트 한 번 → 비효율 | 같은 롤아웃으로 `ppo_epochs` 번, 미니배치로 업데이트 | `for _ in range(cfg.ppo_epochs)` |
| 응답 전체에 같은 advantage → 어떤 토큰이 좋았는지 모름 | critic $V(s_t)$ + GAE 로 **토큰별** advantage | `compute_gae()` |
| 업데이트가 너무 크면 정책 붕괴 | 확률 비율 $\rho_t$ 를 $[1-\epsilon,\, 1+\epsilon]$ 로 clip | `torch.clamp(ratio, ...)` |

이 장에 나오는 기호:

| 기호 | 뜻 | 코드 |
|---|---|---|
| $s_t,\ a_t$ | $t$ 번째 상태(지금까지의 토큰), 행동(다음 토큰) | `ids[:, :t+1]`, `ids[:, t+1]` |
| $\pi_{\mathrm{old}}$ | 롤아웃을 만들 때의 정책 (업데이트 전, 고정) | `old_logp` |
| $\pi_\theta$ | 지금 업데이트 중인 정책 | `logp` |
| $V(s_t)$ | critic 이 예측한 "여기서부터 받을 보상의 합" | `values` |
| $r_t$ | 토큰 $t$ 에서 받는 보상 | `rewards[:, t]` |
| $\gamma,\ \lambda$ | 할인율, GAE 혼합 계수 | `gamma`, `lam` |
| $\epsilon$ | clip 범위 | `clip_range` |

## 2. 토큰 단위 보상 설계

PPO 는 토큰마다 보상이 필요합니다. 응답 길이를 $T$ 라 하면:

$$
r_t = -\beta \Big( \log \pi_{\mathrm{old}}(a_t \mid s_t) - \log \pi_{\mathrm{ref}}(a_t \mid s_t) \Big) \qquad (t = 1, \dots, T)
$$

$$
r_T \;\leftarrow\; r_T + R(x, y)
$$

> **읽는 법** — 모든 응답 토큰은 "참조 정책보다 얼마나 더 확신했는가"만큼 KL 벌점을 받고, **마지막 토큰에만** 응답 전체의 점수 $R$ 이 더해진다.

```python
rewards = -kl_coef * (old_logp - ref_logp)       # 모든 응답 토큰: KL 벌점
rewards[i, last_token] += score                  # 마지막 토큰: 보상 모델/규칙 점수
```

KL 이 토큰마다 보상에 들어가기 때문에, critic 은 "앞으로 받을 점수 − 앞으로 받을 KL 벌점"을 예측하도록 학습됩니다.

## 3. GAE (Generalized Advantage Estimation)

**한 스텝의 '예상 밖 이득'** (TD 오차):

$$
\delta_t = r_t + \gamma \, V(s_{t+1}) - V(s_t)
$$

> **읽는 법** — 실제로 받은 보상 $r_t$ 와 다음 상태의 가치 $V(s_{t+1})$ 를 더한 값이, 지금 상태에서 기대했던 가치 $V(s_t)$ 보다 얼마나 큰가. 양수면 "생각보다 좋은 행동"이다.

**advantage** 는 앞으로의 TD 오차를 $\gamma\lambda$ 배씩 줄여 가며 더한 것입니다. 코드는 뒤에서부터 한 번에 계산하는 점화식을 씁니다.

$$
A_t = \delta_t + \gamma\lambda \, A_{t+1} = \sum_{l=0}^{T-t} (\gamma\lambda)^{l} \, \delta_{t+l}
$$

critic 이 따라가야 할 목표값(returns)은 advantage 에 원래 가치를 더한 것입니다.

$$
\hat{R}_t = A_t + V(s_t)
$$

- $\lambda = 1$ : Monte-Carlo (편향 없음, 분산 큼) — `tests` 의 `test_gae_matches_monte_carlo_when_lambda_1` 참고
- $\lambda = 0$ : 1-step TD, 즉 $A_t = \delta_t$ (분산 작음, critic 이 틀리면 편향)
- LLM RLHF 에서는 보통 $\gamma = 1,\ \lambda = 0.95$

## 4. Clipped surrogate objective

먼저 **확률 비율** — 새 정책이 같은 행동을 이전보다 몇 배 더 (덜) 고르는가:

$$
\rho_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\mathrm{old}}(a_t \mid s_t)} = \exp\Big( \log \pi_\theta(a_t \mid s_t) - \log \pi_{\mathrm{old}}(a_t \mid s_t) \Big)
$$

정책 손실 (최소화):

$$
L^{\mathrm{CLIP}}(\theta) = -\,\mathbb{E}_t \Big[ \min\big( \rho_t A_t,\; \mathrm{clip}(\rho_t,\, 1-\epsilon,\, 1+\epsilon) \, A_t \big) \Big]
$$

> **읽는 법** — 좋은 행동($A_t > 0$)은 확률을 올리되 $1+\epsilon$ 배까지만, 나쁜 행동($A_t < 0$)은 확률을 내리되 $1-\epsilon$ 배까지만 인정한다. 그 이상 움직여도 손실이 줄지 않으므로 기울기가 0 이 된다.

→ "이번 데이터로는 이만큼만 움직여라" 라는 신뢰 영역(trust region) 의 근사. 웹 플랫폼의 **PPO clipped objective** 위젯에서 그래프로 확인할 수 있습니다.

critic 손실도 비슷하게 clip 합니다 ($V^{\mathrm{clip}} = V_{\mathrm{old}} + \mathrm{clip}(V_\theta - V_{\mathrm{old}},\, -\epsilon,\, \epsilon)$):

$$
L^{V}(\theta) = \frac{1}{2} \, \mathbb{E}_t \Big[ \max\big( (V_\theta(s_t) - \hat{R}_t)^2,\; (V^{\mathrm{clip}}(s_t) - \hat{R}_t)^2 \big) \Big]
$$

최종 손실은 $L = L^{\mathrm{CLIP}} + c_V \, L^{V}$ ($c_V$ = `vf_coef`) 입니다.

| 수식 | 코드 (`ppo.py`) |
|---|---|
| $\rho_t$ | `ratio = torch.exp(logp - old_logp[mb])` |
| $-\rho_t A_t$, $-\mathrm{clip}(\rho_t) A_t$ | `pg1`, `pg2` |
| $L^{\mathrm{CLIP}}$ (min 에 −1 을 곱하면 max) | `masked_mean(torch.max(pg1, pg2), m)` |
| $L^{V}$ | `vf_loss` |

로그의 `clipfrac`(clip 된 토큰 비율)과 `approx_kl`(π_old 대비 이동량)이 이 신뢰 영역을 감시하는 지표입니다.
`clipfrac` 가 0.3 이상이면 학습률이 너무 크거나 epoch 가 너무 많다는 신호입니다.

## 5. 모델 4개

실제 RLHF 는 메모리에 **정책, 참조 정책, 보상 모델, critic** 4개를 올립니다.
여기서는 critic 을 정책과 몸통을 공유하는 `value_head` 로 구현했습니다 (`TinyGPT(with_value_head=True)`).
실제 대형 모델에서는 보통 critic 을 별도 모델로 두는데, 그 이유를 생각해 보세요 (연습문제 3).

## 실습

```bash
python -m nlprl train --task sentiment  --algo ppo                    # β=0.3 (기본)
python -m nlprl train --task arithmetic --algo ppo                    # 정답 보상
python -m nlprl train --task sentiment  --algo ppo --kl-coef 0.05 --name ppo_lowkl
python -m nlprl sample --ckpt runs/sentiment/ppo_lowkl/policy.pt
```

참고 결과 (seed 0):

| 실행 | 긍정률 / 정확도 | 문법 정확도 | 시퀀스 KL |
|---|---|---|---|
| sentiment, β=0.5 | 0.98 | 0.91 | 2.2 |
| sentiment, β=0.2 | 0.99 | 0.73 | 1.7 |
| sentiment, β=0.05 | 1.00 | **0.00** | 58.7 |
| arithmetic, β=0 | 0.34 → **0.87** | 0.99 | — |

β=0.05 의 결과 문장은 `the game was fun fun fun fun fun fun fun.` 입니다. 이것이 **reward hacking** 입니다 (04장).

## 연습문제

1. `ppo_epochs` 를 1, 4, 8 로 바꿔 보며 `clipfrac`, `approx_kl`, 학습 속도를 비교하세요.
2. `whiten_advantages=False` 로 바꾸면 어떻게 되나요?
3. value head 를 정책과 공유하면 value loss 의 기울기가 정책 표현을 바꿉니다. `vf_coef` 를 0.1, 0.5, 2.0 으로 바꿔 보고 영향이 있는지 확인하세요.
4. **적응형 KL 제어** (InstructGPT/TRL 의 `AdaptiveKLController`)를 구현하세요: 목표 KL 보다 크면 β 를 키우고 작으면 줄입니다.
5. 응답이 `max_new_tokens` 에서 잘려 EOS 가 없을 때 마지막 토큰에 점수를 주는 것은 옳을까요? 잘린 응답에 벌점을 주는 방식을 구현해 보세요.
