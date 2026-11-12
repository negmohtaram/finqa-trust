import numpy as np

from finqa_trust.chunks import chunk_page
from finqa_trust.embed import embed_texts, load_cache, normalize, save_cache, text_key


class FakeEncoder:
    def __init__(self):
        self.seen: list[str] = []

    def __call__(self, texts: list[str]) -> np.ndarray:
        self.seen.extend(texts)
        return np.array([[len(text), 1.0] for text in texts])


def test_normalize_gives_unit_length_and_leaves_a_zero_vector_alone():
    out = normalize(np.array([[3.0, 4.0], [0.0, 0.0]]))
    assert np.allclose(out[0], [0.6, 0.8])
    assert np.allclose(out[1], [0.0, 0.0])


def test_rows_follow_input_order():
    out = embed_texts(["aaaa", "b"], FakeEncoder(), {})
    assert np.allclose(out[0], normalize(np.array([[4.0, 1.0]]))[0])
    assert np.allclose(out[1], normalize(np.array([[1.0, 1.0]]))[0])


def test_duplicate_text_is_encoded_once_but_returned_twice():
    encoder = FakeEncoder()
    out = embed_texts(["same", "same"], encoder, {})
    assert encoder.seen == ["same"]
    assert out.shape == (2, 2)


def test_second_run_encodes_nothing():
    encoder, cache = FakeEncoder(), {}
    embed_texts(["a", "b"], encoder, cache)
    encoder.seen.clear()
    embed_texts(["a", "b"], encoder, cache)
    assert encoder.seen == []


def test_only_changed_text_is_encoded():
    encoder, cache = FakeEncoder(), {}
    embed_texts(["a", "b"], encoder, cache)
    encoder.seen.clear()
    embed_texts(["a", "c"], encoder, cache)
    assert encoder.seen == ["c"]


def test_cache_survives_a_save_and_load(tmp_path):
    path = tmp_path / "cache.npz"
    cache = {text_key("a"): np.array([1.0, 0.0])}
    save_cache(cache, path)
    assert np.allclose(load_cache(path)[text_key("a")], [1.0, 0.0])


def test_missing_cache_file_is_an_empty_cache(tmp_path):
    assert load_cache(tmp_path / "none.npz") == {}


def report_page(sentences: int = 12, words: int = 25) -> dict:
    """A page of distinct sentences, long enough for several chunks under every chunking."""
    lines = [" ".join(f"s{i}w{j}" for j in range(words)) for i in range(sentences)]
    return {"filename": "X/2010/page_1.pdf", "pre_text": lines, "post_text": [], "table": []}


def insert_words(item: dict, line: int, extra: str = "in that year") -> dict:
    edited = {**item, "pre_text": list(item["pre_text"])}
    words = edited["pre_text"][line].split()
    edited["pre_text"][line] = " ".join(words[:5] + extra.split() + words[5:])
    return edited


def reembedded(before: dict, after: dict, strategy: str) -> tuple[list[str], set[str]]:
    """Texts the encoder saw on the second run, and the chunk texts that really are new."""
    encoder, cache = FakeEncoder(), {}
    old = [c.text for c in chunk_page(before, strategy)]
    embed_texts(old, encoder, cache)
    encoder.seen.clear()
    new = [c.text for c in chunk_page(after, strategy)]
    embed_texts(new, encoder, cache)
    return encoder.seen, set(new) - set(old)


def test_a_one_line_edit_re_embeds_only_the_chunk_that_holds_it():
    page = report_page(words=24)  # 6 sentences = 144 words, so 3 more still fit in the chunk
    seen, changed = reembedded(page, insert_words(page, 7), "sentence")
    assert len(seen) == 1 and set(seen) == changed


def test_a_word_inserted_early_shifts_every_later_fixed_window():
    page = report_page()
    seen, changed = reembedded(page, insert_words(page, 0), "fixed")
    assert set(seen) == changed
    assert len(seen) == len(chunk_page(page, "fixed"))  # no window keeps its old words


def test_a_word_inserted_late_leaves_earlier_fixed_windows_alone():
    page = report_page()
    seen, _ = reembedded(page, insert_words(page, 11), "fixed")
    assert 0 < len(seen) < len(chunk_page(page, "fixed"))


def test_an_edit_that_overfills_a_sentence_chunk_pushes_its_last_sentence_on():
    page = report_page(words=25)  # 6 sentences = exactly 150 words, the limit
    seen, changed = reembedded(page, insert_words(page, 7), "sentence")
    assert set(seen) == changed
    assert len(seen) == 2  # the edited chunk, and a new one holding the sentence that no longer fits
