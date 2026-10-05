BOE_ID = 0
BOS_ID = 1
EOS_ID = 2
BPE_ID = 3
OFFSET = 4
VOCAB_SIZE = 256 + OFFSET


def encode(text: str, *, bos: bool = True, eos: bool = False) -> list[int]:
    ids = [b + OFFSET for b in text.encode("utf-8")]
    return [BOS_ID] * bos + ids + [EOS_ID] * eos


def decode(ids: list[int]) -> str:
    return bytes(i - OFFSET for i in ids if i >= OFFSET).decode("utf-8", errors="replace")
