"""Torch-free tests for data/checks.py (numpy arrays stand in for batches)."""
import numpy as np

from data.checks import check_batch, check_texts, check_tokenized, tokenizer_fertility


class CharTok:
    """Stub tokenizer: one id per character, id 0 = unk for non-ASCII."""
    unk_token_id = 0

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) if ord(c) < 128 else 0 for c in text]}


def test_check_texts_flags_empty_dupes_and_script_mixing():
    texts = ["இது தமிழ் உரை " * 10, "", "   ", "english only text " * 10, "இது தமிழ் உரை " * 10]
    r = check_texts(texts, "TAMIL")
    assert r["n_texts"] == 5 and r["n_empty"] == 2
    assert r["n_duplicates"] == 1
    assert r["n_below_script_threshold"] == 1            # the English passage in a Tamil client
    assert r["chars_min"] > 0


def test_check_texts_empty_input():
    r = check_texts([], "TAMIL")
    assert r["n_texts"] == 0 and r["mean_script_fraction"] is None


def test_check_tokenized_lengths_and_invalid_ids():
    seqs = [[1, 2, 3], [5] * 8, [7], [1, 99]]
    r = check_tokenized(seqs, max_len=8, vocab_size=50)
    assert r["n_sequences"] == 4 and r["len_max"] == 8 and r["n_at_max_len"] == 1
    assert r["n_over_max_len"] == 0 and r["n_shorter_than_2"] == 1
    assert r["n_invalid_token_ids"] == 1                 # id 99 >= vocab 50
    assert check_tokenized([[1, 2, 3, 4]], max_len=3, vocab_size=10)["n_over_max_len"] == 1


def test_fertility_counts_unk():
    r = tokenizer_fertility(CharTok(), ["ab cd", "தமிழ்"])
    assert r["unk_tokens"] == 5 and r["tokens_per_word"] == round(10 / 3, 3)


def test_check_batch_ok_and_problems():
    ids = np.array([[5, 6, 7, 8], [5, 6, 0, 0]])
    ok = np.array([[1, 1, 1, 1], [1, 1, 0, 0]])
    assert check_batch({"input_ids": ids, "attention_mask": ok}, vocab_size=10) == []
    left_pad = np.array([[1, 1, 1, 1], [0, 0, 1, 1]])
    assert any("right-aligned" in p for p in check_batch({"input_ids": ids, "attention_mask": left_pad}, 10))
    assert any("token id" in p for p in check_batch({"input_ids": ids, "attention_mask": ok}, vocab_size=7))
    short = np.array([[1, 1, 1, 1], [1, 0, 0, 0]])
    assert any("fewer than 2" in p for p in check_batch({"input_ids": ids, "attention_mask": short}, 10))
    assert any("differ" in p for p in check_batch({"input_ids": ids, "attention_mask": ok[:, :3]}, 10))
