from blt import tokens


def test_bytes_are_offset_past_the_special_ids():
    assert tokens.encode("A", bos=False) == [65 + tokens.OFFSET]


def test_bos_and_eos_flags():
    assert tokens.encode("", bos=True, eos=True) == [tokens.BOS_ID, tokens.EOS_ID]
    assert tokens.encode("", bos=False, eos=False) == []


def test_every_byte_fits_the_vocab():
    ids = tokens.encode(bytes(range(256)).decode("latin-1"), bos=False)
    assert max(ids) < tokens.VOCAB_SIZE and min(ids) >= tokens.OFFSET


def test_roundtrip_unicode_drops_specials():
    text = "naïve 日本語 😀"
    assert tokens.decode(tokens.encode(text, eos=True)) == text
