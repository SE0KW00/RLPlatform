# 06. GRPO 와 RLVR — 추론 모델 학습의 기본

코드: [`nlprl/algorithms/grpo.py`](../nlprl/algorithms/grpo.py)

## 1. RLVR: 검증 가능한 보상

수학 정답, 코드의 테스트 통과처럼 **프로그램으로 채점할 수 있는 보상**을 쓰면 보상 모델이 필요 없고, 보상 모델 과최적화(04장)도 없습니다.
DeepSeek-R1, OpenAI o1 계열 추론 모델의 핵심 학습 방식입니다. 여기서는 `arithmetic` 과제가 그 축소판입니다.

## 2. GRPO: critic 없는 PPO

PPO 의 critic 은 정책만큼 큰 모델이라 비쌉니다. GRPO 는 **같은 프롬프트에서 G 개를 샘플링해 그룹 안에서 비교**합니다.

```python
def group_advantages(rewards, G):
    g = rewards.view(-1, G)
    return ((g - g.mean(1, keepdim=True)) / (g.std(1, keepdim=True) + eps)).view(-1)
```

- 그룹 평균이 baseline 역할을 합니다 (02장 REINFORCE 의 `batch_mean` 을 프롬프트별로 한 것).
- 응답의 모든 토큰이 같은 advantage 를 받습니다 (토큰별 credit assignment 는 포기).
- KL 은 보상이 아니라 **손실에 직접** k3 추정치로 더합니다: `exp(ref−logp) − (ref−logp) − 1 ≥ 0`.

### 학습 신호가 사라지는 경우

그룹의 G 개 응답이 **모두 맞거나 모두 틀리면** std=0 → advantage=0 → 그 프롬프트에서는 아무것도 배우지 않습니다.
로그의 `zero_adv_groups` 가 이 비율입니다. 너무 쉬운/어려운 문제가 많으면 학습이 멈추는 이유이며,
DAPO 의 "dynamic sampling"(이런 그룹을 버리고 다시 뽑기)이 이를 해결하려는 기법입니다.

## 3. 변형들 (`GRPOConfig` 로 실험 가능)

| 옵션 | 원래 GRPO | 변형 | 출처 |
|---|---|---|---|
| `scale_rewards` | std 로 나눔 | 나누지 않음 (난이도 편향 제거) | Dr. GRPO |
| `loss_type` | `"grpo"`: 시퀀스마다 토큰 평균 후 평균 | `"token"`: 배치 전체 토큰 평균 (길이 편향 제거) | DAPO / Dr. GRPO |
| `num_iterations` | μ (롤아웃 재사용 횟수) | 1 이면 완전 on-policy | DeepSeekMath |

## 실습

```bash
python -m nlprl train --task arithmetic --algo grpo
python -m nlprl train --task arithmetic --algo grpo --group-size 2 --name grpo_g2
python -m nlprl train --task arithmetic --algo grpo --group-size 16 --name grpo_g16
python -m nlprl plot runs/arithmetic/*/history.json --metric eval/accuracy --out arithmetic.png
```

참고 결과 (seed 0, 샘플링 정확도): SFT 0.32 → GRPO **0.73** (200 step), REINFORCE 0.71, PPO 0.87.
(작은 실험이라 seed 에 따라 ±0.1 정도 흔들립니다. 여러 seed 로 돌려 보세요: `--seed 1`)

### "RL 은 새로운 능력을 가르치는가, 이미 있는 것을 끌어올리는가?"

01장에서 SFT 모델의 **greedy 정확도는 이미 ~97%** 였습니다. RL 은 새로운 덧셈 능력을 가르친 것이 아니라,
모델이 이미 "알던" 정답 쪽으로 확률 질량을 모은 것입니다 (pass@1 ↑). 최근 연구에서 활발히 논의되는 주제입니다:
RL 후 pass@1 은 오르지만 pass@k(큰 k)는 오히려 떨어질 수 있다는 보고가 있습니다.

## 연습문제

1. `--group-size` 를 2, 4, 8, 16 으로 바꾸며 `zero_adv_groups` 와 최종 정확도를 비교하세요. (배치 크기 64 는 고정)
2. `scale_rewards=False`, `loss_type="token"` 을 각각 켜 보세요. 이 과제에서는 응답 길이가 거의 일정한데, 길이가 다양한 과제라면 어떤 차이가 생길까요?
3. **pass@k 측정기**를 만들어 보세요: 프롬프트마다 k 개를 샘플링해 하나라도 맞으면 성공. SFT 와 GRPO 모델의 pass@1, pass@8 을 비교하세요.
4. **커리큘럼**: `ArithmeticTask(max_operand=...)` 를 학습 중에 점점 키워 보세요.
5. `ArithmeticTask(p_correct=0.05)` 로 사전학습하면 (정답이 거의 없는 데이터) RL 이 여전히 작동하나요? 왜 그런가요?
