# 부록. 기호와 수식 읽는 법

레슨 문서의 수식에 나오는 기호를 한곳에 모았습니다. 각 기호의 **읽는 법**, **뜻**, 그리고 이 플랫폼 **코드에서 대응하는 변수**를 함께 적었습니다.

## 1. 수식 읽기 기초

| 표기 | 읽는 법 | 뜻 |
|---|---|---|
| $\pi_\theta(a \mid s)$ | "파이 세타, s 가 주어졌을 때 a" | 상태 $s$ 에서 행동 $a$ 를 고를 확률. 세로선 $\mid$ 는 "~가 주어졌을 때" |
| $y \sim \pi_\theta(\cdot \mid x)$ | "y 는 파이 세타에서 뽑는다" | $\sim$ 는 "그 분포에서 샘플링", $\cdot$ 는 "모든 가능한 값" 자리 |
| $\mathbb{E}_{y \sim \pi}[f(y)]$ | "y 를 파이에서 뽑을 때 f 의 기댓값" | 무한히 많이 뽑았을 때의 평균. 코드에서는 배치 평균 `.mean()` |
| $\sum_{t=1}^{T} f_t$ | "t 가 1 부터 T 까지 f 의 합" | 코드에서는 `.sum(-1)` (보통 마스크를 곱한 뒤) |
| $\nabla_\theta f$ | "f 의 세타에 대한 기울기" | $f$ 를 가장 빨리 키우는 파라미터 방향. 코드에서는 `loss.backward()` |
| $\log$ | 자연로그 | 확률의 곱을 합으로 바꿔 준다: $\log(ab) = \log a + \log b$ |
| $y_{<t}$ | "t 보다 앞의 y" | $y_1, \dots, y_{t-1}$ |
| $\hat{x}$ | "x 햇" | $x$ 의 **추정치** (샘플로 근사한 값) |
| $x^{*}$ | "x 스타" | **최적의** $x$ |
| $a \succ b$ | "a 가 b 보다 선호된다" | 선호 관계 |
| $\sigma(z)$ | 시그모이드 | $1 / (1 + e^{-z})$ — 아무 실수를 0~1 사이 확률로 바꾼다 |
| $\mathrm{clip}(z, a, b)$ | 클립 | $z$ 를 $[a, b]$ 범위로 자른다. 코드에서는 `torch.clamp` |

## 2. 언어 모델과 강화학습

| 기호 | 뜻 | 코드 |
|---|---|---|
| $x$ | 프롬프트 | `prompts` |
| $y = (y_1, \dots, y_T)$ | 응답 (토큰 $T$ 개) | `roll.responses`, 응답 마스크 구간 |
| $s_t$ | 상태 = 프롬프트 + 지금까지 생성한 토큰 | `input_ids[:, :t+1]` |
| $a_t$ | 행동 = 다음 토큰 | `input_ids[:, t+1]` |
| $\theta$ | 정책 파라미터 | `policy.parameters()` |
| $\pi_\theta$ | 학습 중인 정책 (언어 모델) | `policy` |
| $\pi_{\mathrm{ref}}$ | 참조 정책 = 얼려 둔 SFT 모델 | `ref` (`make_reference()`) |
| $\pi_{\mathrm{old}}$ | 롤아웃을 만들 때의 정책 (PPO·GRPO) | `old_logp` |
| $\log \pi_\theta(y_t \mid s_t)$ | 토큰 하나의 log 확률 | `token_logprobs()` 의 한 원소 |
| $\log \pi_\theta(y \mid x)$ | 응답 전체의 log 확률 = 토큰 log 확률의 합 | `(logp * mask).sum(-1)` |
| $R(x, y)$ | 응답 전체의 보상 | `score`, `task.reward()` |
| $r_t$ | 토큰 $t$ 의 보상 (PPO) | `rewards[:, t]` |
| $\beta$ | KL 벌점 계수 (DPO 에서는 온도) | `kl_coef`, `beta` |
| $\mathrm{KL}(p  \Vert  q)$ | 분포 $p$ 가 $q$ 와 얼마나 다른가 (0 이상) | `kl` |

## 3. 알고리즘별 기호

| 기호 | 등장 | 뜻 | 코드 |
|---|---|---|---|
| $A$, $A_t$, $A_i$ | 02·03·06 | advantage: 기준보다 얼마나 좋았나 | `adv` |
| $b$ | 02 | baseline (기준값) | `b` |
| $V(s_t)$ | 03 | critic 이 예측한 상태 가치 | `values` |
| $\gamma$ | 03 | 할인율 (LLM 에서는 보통 1) | `gamma` |
| $\lambda$ | 03 | GAE 혼합 계수 | `lam` |
| $\delta_t$ | 03 | TD 오차: 한 스텝의 예상 밖 이득 | `delta` |
| $\hat{R}_t$ | 03 | critic 학습 목표 (returns) | `returns` |
| $\rho_t$ | 03·06 | 확률 비율 $\pi_\theta / \pi_{\mathrm{old}}$ | `ratio` |
| $\epsilon$ | 03·06 | clip 범위 | `clip_range` |
| $r_\phi(x, y)$ | 04 | 보상 모델 점수, $\phi$ 는 그 파라미터 | `rm(ids, mask)` |
| $y_w,\ y_l$ | 04·05 | 선택된(chosen) / 거절된(rejected) 응답 | `chosen`, `rejected` |
| $\hat{r}_\theta(x, y)$ | 05 | DPO 의 암묵적 보상 $\beta \log \frac{\pi_\theta}{\pi_{\mathrm{ref}}}$ | `rc`, `rr` |
| $m$ | 05 | 암묵적 보상 차이 (margin) | `margin` |
| $Z(x)$ | 05 | 정규화 상수 (계산 불가, 소거됨) | — |
| $G$ | 06 | 프롬프트당 샘플 수 (그룹 크기) | `group_size` |
| $D_{i,t}$ | 06 | k3 KL 추정치 $e^u - u - 1$ | `kl` |

## 4. 수식 ↔ 코드 한눈에 보기

| 알고리즘 | 핵심 수식 | 코드 한 줄 |
|---|---|---|
| REINFORCE | $-A \log \pi_\theta(y \mid x)$ | `-(adv * seq_logp).mean()` |
| PPO | $-\min(\rho A,\ \mathrm{clip}(\rho) A)$ | `torch.max(pg1, pg2)` |
| 보상 모델 | $-\log \sigma(r_w - r_l)$ | `-F.logsigmoid(rc - rr).mean()` |
| DPO | $-\log \sigma(\hat{r}_w - \hat{r}_l)$ | `-F.logsigmoid(logits).mean()` |
| GRPO | $A_i = (R_i - \operatorname{mean}) / \operatorname{std}$ | `group_advantages()` |
