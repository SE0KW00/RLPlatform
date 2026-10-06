# 00. 학습 로드맵

이 플랫폼의 목표는 **"LLM 을 강화학습으로 조정한다"는 말이 코드 수준에서 정확히 무엇을 뜻하는지** 손으로 확인하는 것입니다.
모든 알고리즘은 라이브러리(TRL, OpenRLHF 등) 없이 PyTorch 로만, 파일 하나에 한 알고리즘씩 구현되어 있습니다.
모델은 아주 작은 GPT(약 24만 파라미터)라서 **노트북 CPU 에서 각 실험이 1분 안팎**이면 끝납니다.

> 💡 `python -m nlprl serve` 로 웹 플랫폼을 띄우면 이 문서들을 **동작 구조 시각화**, **코드 워크스루**와 함께 볼 수 있습니다.

## 전체 그림

```
            ┌──────────── 사전학습 / SFT (pretrain.py) ────────────┐
            │  다음 토큰 예측으로 과제의 "언어"를 배운 정책 π_SFT    │
            └──────────────────────────┬───────────────────────────┘
                                       │ (복사해서 π_ref 로 얼려 둠)
          ┌────────────────────────────┼─────────────────────────────┐
          ▼                            ▼                             ▼
  온라인 RL (보상 함수)        선호 데이터 (y_w ≻ y_l)        검증 가능한 보상 (RLVR)
  02 REINFORCE                 04 보상 모델 → 03 PPO           06 GRPO
  03 PPO                       05 DPO (보상 모델 없이)
```

## 추천 순서

| 단계 | 문서 | 코드 | 핵심 질문 |
|---|---|---|---|
| 1 | [01 텍스트 생성을 RL 로 보기](01_text_generation_as_rl.md) | `model.py`, `generation.py`, `tasks.py` | 상태·행동·보상·에피소드는 각각 무엇인가? |
| 2 | [02 REINFORCE](02_reinforce.md) | `algorithms/reinforce.py` | 미분 불가능한 보상으로 어떻게 기울기를 얻는가? |
| 3 | [03 PPO 와 RLHF](03_ppo_rlhf.md) | `algorithms/ppo.py` | 왜 critic, GAE, clipping, KL 이 필요한가? |
| 4 | [04 보상 모델과 reward hacking](04_reward_model_and_hacking.md) | `preference.py` | 사람 선호를 어떻게 보상으로 바꾸고, 무엇이 잘못될 수 있나? |
| 5 | [05 DPO](05_dpo.md) | `algorithms/dpo.py` | RL 없이 선호를 최적화할 수 있는 이유는? |
| 6 | [06 GRPO 와 RLVR](06_grpo_rlvr.md) | `algorithms/grpo.py` | critic 없이 어떻게 advantage 를 구하나? 추론 모델은 어떻게 학습되나? |
| 7 | [07 더 나아가기](07_next_steps.md) | — | 실제 LLM 으로 확장하려면? 무엇을 더 읽을까? |

각 문서 끝에는 **실습**과 **연습문제**가 있습니다. 연습문제는 대부분 코드 몇 줄을 바꿔 보고 결과를 관찰하는 형태입니다.

## 선수 지식

- PyTorch 기초 (텐서, autograd, optimizer)
- 확률 기초 (기댓값, log-likelihood, KL divergence)
- Transformer 가 다음 토큰을 예측한다는 정도의 이해

RL 사전 지식은 없어도 됩니다. 필요한 개념은 문서에서 LLM 맥락으로 설명합니다.
