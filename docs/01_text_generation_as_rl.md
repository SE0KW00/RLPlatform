# 01. 텍스트 생성을 강화학습으로 보기

## 1. MDP 로의 대응

| RL 용어 | LLM 에서의 의미 | 이 코드에서 |
|---|---|---|
| 상태 $s_t$ | 프롬프트 + 지금까지 생성한 토큰들 | `input_ids[:, :t+1]` |
| 행동 $a_t$ | 다음 토큰 하나 (어휘 크기만큼의 이산 행동) | `input_ids[:, t+1]` |
| 정책 $\pi_\theta(a_t \mid s_t)$ | 언어 모델의 다음 토큰 분포 | `TinyGPT` 의 `logits` → softmax |
| 전이 $P(s_{t+1} \mid s_t, a_t)$ | **결정적**: 상태 뒤에 토큰을 이어 붙임 | `torch.cat([ids, nxt])` |
| 에피소드 | 프롬프트 하나에 대한 응답 하나 (EOS 까지) | `generate()` 한 번 |
| 보상 $R(x, y)$ | 보통 **응답이 끝난 뒤 한 번만** 주어짐 | `task.reward(prompt, response)` |

LLM RL 의 특이점:

- **전이가 결정적**이고 환경이 단순하다 → 어려움은 전부 "정책"과 "보상" 쪽에 있다.
- **보상이 희소(sparse)** 하다. 수십~수천 토큰 중 마지막에 점수 하나 → credit assignment 문제.
- **행동 공간이 거대**하다 (실제 LLM 은 어휘 10만 개 이상).
- 출발점이 무작위 정책이 아니라 **이미 똑똑한 사전학습 모델**이다. 그래서 RL 의 목표는
  "처음부터 배우기"가 아니라 "기존 분포를 원하는 방향으로 살짝 옮기기"이고,
  너무 멀리 가지 않도록 **KL 제약**을 거는 것이 표준이 되었다.

## 2. 목적 함수

RL 단계에서 크게 만들고 싶은 값은 다음 한 줄입니다.

$$
J(\theta) = \mathbb{E}_{x \sim \mathcal{D}} \, \mathbb{E}_{y \sim \pi_\theta(\cdot \mid x)} \big[ R(x, y) \big] \;-\; \beta \, \mathrm{KL}\big( \pi_\theta \,\|\, \pi_{\mathrm{ref}} \big)
$$

> **읽는 법** — 데이터에서 프롬프트 $x$ 를 뽑고, 정책 $\pi_\theta$ 로 응답 $y$ 를 생성했을 때 받는 보상 $R$ 의 **평균을 크게** 만들되, 처음 정책 $\pi_{\mathrm{ref}}$ 와 달라진 정도(KL)에는 $\beta$ 만큼 **벌점**을 준다.

| 기호 | 뜻 | 코드 |
|---|---|---|
| $\theta$ | 정책(언어 모델)의 파라미터 | `policy.parameters()` |
| $x \sim \mathcal{D}$ | 데이터 분포 $\mathcal{D}$ 에서 뽑은 프롬프트 | `task.sample_prompts()` |
| $y \sim \pi_\theta(\cdot \mid x)$ | $x$ 가 주어졌을 때 정책으로 샘플링한 응답 | `generate()` |
| $R(x, y)$ | 응답 전체에 대한 보상 (숫자 하나) | `task.reward()` |
| $\pi_{\mathrm{ref}}$ | RL 시작 시점(SFT) 정책을 얼려 둔 것 | `make_reference()` |
| $\beta$ | KL 벌점의 세기 | `kl_coef` |
| $\mathbb{E}[\,\cdot\,]$ | 기댓값 — 실제로는 **배치 평균**으로 근사 | `.mean()` |

### 응답의 확률 = 토큰 확률의 곱

응답 $y = (y_1, y_2, \dots, y_T)$ 의 확률은 각 토큰이 "앞의 모든 토큰이 주어졌을 때" 나올 확률을 곱한 것이고, 로그를 취하면 **합**이 됩니다.

$$
\log \pi_\theta(y \mid x) = \sum_{t=1}^{T} \log \pi_\theta\big( y_t \mid x, y_{<t} \big)
$$

여기서 $y_{<t}$ 는 "$t$ 번째보다 앞에 생성된 토큰들"입니다. 코드에서는 `token_logprobs()` 의 결과에 응답 마스크를 곱해 더합니다: `(logp * action_mask).sum(-1)`.

## 3. 텐서 레이아웃 (가장 많이 헷갈리는 부분)

`generation.py` 상단 주석을 꼭 읽어 보세요. 요약:

```
input_ids     = [pad pad BOS t h e _ m o v i e  _ w a s _ g o o d . EOS pad]
response_mask = [ 0   0   0  0 0 0 0 0 0 0 0 0  1 1 1 1 1 1 1 1 1 1  1   0 ]
logprobs[:, i] = log π(input_ids[i+1] | input_ids[:i+1])   → 길이 L-1
action_mask    = response_mask[:, 1:]
```

- 배치 생성은 프롬프트 길이가 달라서 **왼쪽 패딩**을 쓴다. 그래야 모든 행의 "다음 토큰" 위치가 같다.
- 왼쪽 패딩이 있어도 위치 임베딩이 0 부터 시작하도록 `position_ids_from_mask()` 를 쓴다.
  (`tests/test_nlprl.py::test_left_padding_does_not_change_logprobs` 가 이를 검증)

## 4. 두 가지 과제

| 과제 | 프롬프트 → 응답 예 | 보상 | 무엇의 축소판인가 |
|---|---|---|---|
| `sentiment` | `the movie` → ` was really good and fun.` | 긍정 단어 수 − 부정 단어 수 | RLHF (선호/유용성 같은 '부드러운' 목표) |
| `arithmetic` | `7+12=` → `19.` | 정답이면 1, 아니면 0 | RLVR (수학·코드처럼 채점 가능한 목표) |

`sentiment` 의 사전학습 코퍼스는 긍정·부정이 반반이므로 SFT 모델의 긍정률은 약 50% 입니다.
`arithmetic` 의 코퍼스는 **정답률이 40% 뿐**이고 나머지는 무작위 오답입니다.

## 실습

```bash
python -m nlprl pretrain --task sentiment       # 약 1분
python -m nlprl pretrain --task arithmetic      # 약 40초
python -m nlprl sample --ckpt runs/sentiment/sft/policy.pt --n 10
python -m nlprl sample --ckpt runs/arithmetic/sft/policy.pt --n 200 | tail -1
python -m nlprl sample --ckpt runs/arithmetic/sft/policy.pt --n 200 --greedy | tail -1
```

마지막 두 줄을 비교해 보세요. 덧셈 SFT 모델은 샘플링하면 약 30% 만 맞히지만, **greedy 로 디코딩하면 거의 다 맞힙니다.**
즉 모델은 정답을 "알고" 있지만 확률 질량이 오답에 퍼져 있습니다. 06장에서 RL 이 이 분포를 날카롭게 만드는 것을 보게 됩니다.

## 연습문제

1. `tasks.py` 에 새 과제를 추가해 보세요. 예: 프롬프트로 주어진 단어를 거꾸로 쓰기, 응답 길이를 정확히 N 글자로 맞추기.
2. `generate()` 에 top-k 샘플링을 추가하고, 온도(temperature)를 0.5/1.0/1.5 로 바꿔 SFT 모델의 `distinct`, `well_formed` 가 어떻게 변하는지 기록하세요.
3. 왜 생성할 때 오른쪽 패딩이 아니라 왼쪽 패딩을 써야 할까요? 오른쪽 패딩이면 `logits[:, -1, :]` 가 무엇을 의미하게 되나요?
