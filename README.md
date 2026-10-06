# RLPlatform — NLP 강화학습을 바닥부터 공부하는 플랫폼

LLM 정렬과 추론 학습에 쓰이는 강화학습 알고리즘(**REINFORCE, PPO-RLHF, 보상 모델, DPO, GRPO**)을
라이브러리 없이 **PyTorch 만으로** 한 파일에 하나씩 구현하고, **노트북 CPU 에서 1분 안팎**으로 직접 돌려 보며 공부하는 플랫폼입니다.

- 처음부터 구현한 작은 GPT (약 24만 파라미터, 문자 단위)
- 두 가지 장난감 과제
  - `sentiment` — 긍정 리뷰 쓰기 (**RLHF** 의 축소판, reward hacking 관찰 가능)
  - `arithmetic` — 덧셈 정답 맞히기 (**RLVR / GRPO** 의 축소판)
- 한국어 커리큘럼 문서 ([`docs/`](docs/00_roadmap.md)): 이론 → 코드 → 실습 → 연습문제

## 설치

```bash
pip install -r requirements.txt     # torch, matplotlib, pytest
```

GPU 는 필요 없습니다.

## 빠른 시작

```bash
# 0) SFT: 과제의 언어를 배운 초기 정책 만들기
python -m nlprl pretrain --task sentiment

# 1) 온라인 RL
python -m nlprl train --task sentiment --algo reinforce
python -m nlprl train --task sentiment --algo ppo

# 2) RLHF: 선호 데이터 → 보상 모델 → PPO
python -m nlprl reward-model --task sentiment
python -m nlprl train --task sentiment --algo ppo --reward rm

# 3) DPO: 보상 모델 없이 선호 데이터로 직접 최적화
python -m nlprl dpo --task sentiment

# 4) RLVR: 정답 보상 + GRPO
python -m nlprl pretrain --task arithmetic
python -m nlprl train --task arithmetic --algo grpo

# 결과 보기
python -m nlprl sample --ckpt runs/sentiment/ppo/policy.pt
python -m nlprl plot runs/sentiment/*/history.json --metric eval/positive_rate --out sentiment.png
```

`bash scripts/quickstart.sh` 로 위 과정을 한 번에 실행할 수 있습니다 (CPU 약 10분).

## 이런 것을 직접 관찰할 수 있습니다

| 실험 | 관찰 |
|---|---|
| `train --algo ppo --kl-coef 0.05` | 보상은 7점인데 문장은 `the game was fun fun fun fun fun fun fun.` — **reward hacking** |
| `train --algo ppo --reward rm` | 보상 모델 점수 5.5, 실제 보상 1.7 — **보상 모델 과최적화** |
| `dpo --lr 1e-4 --epochs 3` | chosen 응답의 확률까지 떨어지며 문법 붕괴 — **오프라인 선호 학습의 함정** |
| `arithmetic`: SFT greedy vs 샘플링 | greedy 97% vs 샘플링 32% → GRPO 후 샘플링 73% — **RL 은 분포를 날카롭게 한다** |
| `train --algo grpo --group-size 2` | `zero_adv_groups` 증가 — **그룹 내 보상이 같으면 학습 신호가 없다** |

## 커리큘럼

| | 문서 | 코드 |
|---|---|---|
| 00 | [학습 로드맵](docs/00_roadmap.md) | |
| 01 | [텍스트 생성을 RL 로 보기](docs/01_text_generation_as_rl.md) | `model.py`, `generation.py`, `tasks.py` |
| 02 | [REINFORCE](docs/02_reinforce.md) | `algorithms/reinforce.py` |
| 03 | [PPO 와 RLHF](docs/03_ppo_rlhf.md) | `algorithms/ppo.py` |
| 04 | [보상 모델과 reward hacking](docs/04_reward_model_and_hacking.md) | `preference.py` |
| 05 | [DPO](docs/05_dpo.md) | `algorithms/dpo.py` |
| 06 | [GRPO 와 RLVR](docs/06_grpo_rlvr.md) | `algorithms/grpo.py` |
| 07 | [더 나아가기](docs/07_next_steps.md) | |

## 프로젝트 구조

```
nlprl/
├── tokenizer.py        문자 단위 토크나이저
├── model.py            TinyGPT (+ value head), RewardModel
├── tasks.py            sentiment / arithmetic 과제와 보상 함수
├── generation.py       롤아웃 생성, 토큰별 log-prob, 마스킹 유틸
├── pretrain.py         0단계: SFT
├── preference.py       선호 쌍 생성, 보상 모델 학습
├── algorithms/
│   ├── reinforce.py    REINFORCE + baseline + KL
│   ├── ppo.py          PPO (GAE, clipping, value head)
│   ├── dpo.py          DPO
│   └── grpo.py         GRPO (+ Dr.GRPO / DAPO 옵션)
└── cli.py              python -m nlprl ...
docs/                   한국어 학습 문서
tests/                  단위 테스트 + 스모크 테스트 (pytest, 약 15초)
runs/                   실행 결과 (체크포인트, history.json) — git 에서 제외
```

## 테스트

```bash
python -m pytest -q
```
