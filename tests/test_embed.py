import numpy as np

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
