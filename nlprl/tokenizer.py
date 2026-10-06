"""문자 단위(character-level) 토크나이저.

실제 LLM은 BPE 같은 서브워드 토크나이저를 쓰지만, RL 알고리즘을 공부하는 데에는
"토큰 = 행동(action)" 이라는 사실만 중요하다. 문자 단위로 두면 어휘가 작아서
CPU에서도 빠르게 학습되고, 생성 결과를 눈으로 쉽게 확인할 수 있다.
"""

from __future__ import annotations

PAD, BOS, EOS = "<pad>", "<bos>", "<eos>"
DEFAULT_CHARS = "abcdefghijklmnopqrstuvwxyz0123456789 +=."


class CharTokenizer:
    pad_id = 0
    bos_id = 1
    eos_id = 2

    def __init__(self, chars: str = DEFAULT_CHARS):
        self.itos = [PAD, BOS, EOS] + list(chars)
        self.stoi = {s: i for i, s in enumerate(self.itos)}

    @property
    def vocab_size(self) -> int:
        return len(self.itos)

    def encode(self, text: str, add_bos: bool = False, add_eos: bool = False) -> list[int]:
        ids = [self.stoi[c] for c in text]
        if add_bos:
            ids = [self.bos_id] + ids
        if add_eos:
            ids = ids + [self.eos_id]
        return ids

    def decode(self, ids, skip_special: bool = True) -> str:
        out = []
        for i in ids:
            i = int(i)
            if i == self.eos_id and skip_special:
                break
            if skip_special and i in (self.pad_id, self.bos_id):
                continue
            out.append(self.itos[i])
        return "".join(out)
