# 07. 더 나아가기

## 1. 실제 LLM 으로 확장할 때 바뀌는 것

| 이 플랫폼 | 실제 규모 |
|---|---|
| 문자 단위 토크나이저 | BPE 토크나이저 (`transformers`) |
| 24만 파라미터 TinyGPT | 0.5B~70B+ 모델, LoRA 로 일부만 학습하기도 함 |
| KV-cache 없는 생성 | vLLM / SGLang 같은 추론 엔진으로 롤아웃 (학습과 분리) |
| 규칙 보상 | 보상 모델, LLM-as-judge, 코드 실행 샌드박스, 수학 검증기 |
| 모든 모델이 한 프로세스 | 정책·참조·보상·critic 을 여러 GPU 에 분산 |

이 플랫폼의 알고리즘 코드는 TRL 의 `PPOTrainer`, `DPOTrainer`, `GRPOTrainer` 와 같은 구조를 따르므로,
여기서 이해한 내용을 그대로 라이브러리 코드 읽기에 적용할 수 있습니다.

추천 다음 단계:

1. **TRL** 로 GPT-2 / Qwen-0.5B 에 같은 실험 반복하기 (IMDB 긍정 리뷰 생성 = `sentiment` 과제의 원본)
2. **GSM8K** 에 GRPO 적용해 보기 (= `arithmetic` 과제의 원본)
3. **OpenRLHF**, **verl** 로 분산 RL 학습 구조 살펴보기

## 2. 읽을거리 (이 플랫폼의 각 장과 대응)

| 장 | 논문 |
|---|---|
| 02 | Williams (1992), *Simple statistical gradient-following algorithms* (REINFORCE) |
| 02 | Ahmadian et al. (2024), *Back to Basics: Revisiting REINFORCE Style Optimization for Learning from Human Feedback in LLMs* (RLOO) |
| 03 | Schulman et al. (2017), *Proximal Policy Optimization Algorithms* |
| 03 | Schulman et al. (2016), *High-Dimensional Continuous Control Using GAE* |
| 03 | Ziegler et al. (2019), *Fine-Tuning Language Models from Human Preferences* |
| 03 | Ouyang et al. (2022), *Training language models to follow instructions with human feedback* (InstructGPT) |
| 03 | Huang et al. (2024), *The N+ Implementation Details of RLHF with PPO* |
| 04 | Gao et al. (2022), *Scaling Laws for Reward Model Overoptimization* |
| 05 | Rafailov et al. (2023), *Direct Preference Optimization* |
| 05 | Azar et al. (2023), *A General Theoretical Paradigm to Understand Learning from Human Preferences* (IPO) |
| 06 | Shao et al. (2024), *DeepSeekMath* (GRPO) |
| 06 | DeepSeek-AI (2025), *DeepSeek-R1* |
| 06 | Yu et al. (2025), *DAPO* / Liu et al. (2025), *Understanding R1-Zero-Like Training* (Dr. GRPO) |

## 3. 이 플랫폼에 기여할 만한 확장 아이디어

- 새 과제: 단어 뒤집기, 길이 제어, 간단한 JSON 형식 맞추기 (format reward)
- 새 알고리즘: RLOO, ReMax, KTO, ORPO, SimPO, Online DPO
- 다단계 추론: `arithmetic` 을 `3+4+5=` 처럼 늘리고 중간 풀이를 쓰게 한 뒤 정답에만 보상 주기
- Process reward: 중간 단계마다 보상을 주는 과제와 PRM
