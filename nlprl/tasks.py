"""학습용 장난감 과제(task).

각 과제는 다음을 제공한다.
  - corpus()        : 사전학습(SFT)용 문장들 → "기본 언어 능력"을 가진 정책을 만든다
  - sample_prompts(): RL 단계에서 사용할 프롬프트
  - reward()        : (프롬프트, 응답) → 스칼라 보상
  - metrics()       : 학습 경과를 관찰할 지표

두 과제는 현대 LLM RL 의 두 갈래를 축소한 것이다.
  1. SentimentTask  : "긍정적인 리뷰를 쓰도록" 조정 → RLHF 의 축소판
                      (보상은 사람 선호의 대리(proxy)이며, reward hacking 이 일어날 수 있다)
  2. ArithmeticTask : "정답을 맞히도록" 조정 → RLVR(검증 가능한 보상) / GRPO 의 축소판
"""

from __future__ import annotations

import random
import re
from abc import ABC, abstractmethod


class Task(ABC):
    name: str
    max_new_tokens: int

    @abstractmethod
    def corpus(self, n: int, rng: random.Random) -> list[str]:
        """사전학습용 전체 문장 (프롬프트 + 응답)."""

    @abstractmethod
    def sample_prompts(self, n: int, rng: random.Random) -> list[str]:
        ...

    @abstractmethod
    def reward(self, prompt: str, response: str) -> float:
        ...

    def rewards(self, prompts: list[str], responses: list[str]) -> list[float]:
        return [self.reward(p, r) for p, r in zip(prompts, responses)]

    def metrics(self, prompts: list[str], responses: list[str]) -> dict[str, float]:
        rs = self.rewards(prompts, responses)
        return {"reward": sum(rs) / len(rs)}


# ---------------------------------------------------------------------------
# 1. 감성 리뷰 생성 (RLHF 축소판)
# ---------------------------------------------------------------------------
SUBJECTS = ["the movie", "the food", "the hotel", "the book", "the game", "the show"]
ADVERBS = ["", "very ", "really ", "so "]
POSITIVE = ["great", "good", "fun", "nice", "lovely", "amazing"]
NEGATIVE = ["bad", "awful", "boring", "sad", "poor", "terrible"]

_ADV = r"(?:very |really |so )?"
_ADJ = "(?:" + "|".join(POSITIVE + NEGATIVE) + ")"
_GRAMMAR = re.compile(rf"^ was {_ADV}{_ADJ}(?: and {_ADV}{_ADJ})?\.$")


class SentimentTask(Task):
    """프롬프트 "the movie" → 응답 " was really good and fun."

    사전학습 코퍼스에는 긍정/부정 리뷰가 반반 섞여 있다.
    보상 = (긍정 단어 수) - (부정 단어 수).  이 보상은 '문법'을 전혀 보지 않으므로
    KL 제약 없이 최적화하면 "great great great ..." 같은 reward hacking 이 일어날 수 있다.
    """

    name = "sentiment"
    max_new_tokens = 44

    def __init__(self, p_positive: float = 0.5, p_two_clauses: float = 0.5):
        self.p_positive = p_positive
        self.p_two_clauses = p_two_clauses

    def _sentence(self, rng: random.Random) -> str:
        positive = rng.random() < self.p_positive
        words = POSITIVE if positive else NEGATIVE
        s = f"{rng.choice(SUBJECTS)} was {rng.choice(ADVERBS)}{rng.choice(words)}"
        if rng.random() < self.p_two_clauses:
            # 두 번째 절은 80% 확률로 같은 감성
            words2 = words if rng.random() < 0.8 else (NEGATIVE if positive else POSITIVE)
            s += f" and {rng.choice(ADVERBS)}{rng.choice(words2)}"
        return s + "."

    def corpus(self, n, rng):
        return [self._sentence(rng) for _ in range(n)]

    def sample_prompts(self, n, rng):
        return [rng.choice(SUBJECTS) for _ in range(n)]

    @staticmethod
    def sentiment_score(text: str) -> int:
        words = re.findall(r"[a-z]+", text)
        return sum(w in POSITIVE for w in words) - sum(w in NEGATIVE for w in words)

    def reward(self, prompt, response):
        return float(self.sentiment_score(response))

    @staticmethod
    def is_well_formed(response: str) -> bool:
        return bool(_GRAMMAR.match(response))

    def metrics(self, prompts, responses):
        rs = self.rewards(prompts, responses)
        n = len(responses)
        return {
            "reward": sum(rs) / n,
            "positive_rate": sum(r > 0 for r in rs) / n,
            "well_formed": sum(self.is_well_formed(r) for r in responses) / n,
            "distinct": len(set(responses)) / n,
            "avg_len": sum(len(r) for r in responses) / n,
        }


# ---------------------------------------------------------------------------
# 2. 덧셈 (RLVR / GRPO 축소판)
# ---------------------------------------------------------------------------
class ArithmeticTask(Task):
    """프롬프트 "7+12=" → 응답 "19."

    사전학습 데이터의 정답률은 p_correct (기본 40%) 뿐이고 나머지는 무작위 오답이다.
    그래서 SFT 모델은 '정답을 알긴 알지만' 샘플링하면 자주 틀린다.
    정답 여부(0/1)라는 검증 가능한 보상으로 RL 을 하면 정답 확률이 크게 올라간다.
    → "RL 은 모델이 이미 가진 능력을 끌어올린다(sharpening)" 는 관찰을 직접 해볼 수 있다.
    """

    name = "arithmetic"
    max_new_tokens = 4

    def __init__(self, max_operand: int = 19, p_correct: float = 0.4):
        self.max_operand = max_operand
        self.p_correct = p_correct

    def _pair(self, rng):
        return rng.randint(0, self.max_operand), rng.randint(0, self.max_operand)

    def corpus(self, n, rng):
        out = []
        for _ in range(n):
            a, b = self._pair(rng)
            ans = a + b if rng.random() < self.p_correct else rng.randint(0, 2 * self.max_operand)
            out.append(f"{a}+{b}={ans}.")
        return out

    def sample_prompts(self, n, rng):
        return [f"{a}+{b}=" for a, b in (self._pair(rng) for _ in range(n))]

    def reward(self, prompt, response):
        a, b = map(int, prompt[:-1].split("+"))
        return 1.0 if response == f"{a + b}." else 0.0

    def metrics(self, prompts, responses):
        rs = self.rewards(prompts, responses)
        n = len(rs)
        return {
            "reward": sum(rs) / n,
            "accuracy": sum(rs) / n,
            "well_formed": sum(bool(re.fullmatch(r"\d+\.", r)) for r in responses) / n,
        }


TASKS = {"sentiment": SentimentTask, "arithmetic": ArithmeticTask}


def get_task(name: str) -> Task:
    return TASKS[name]()
