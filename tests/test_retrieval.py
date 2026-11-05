import numpy as np

from finqa_trust.chunks import Chunk
from finqa_trust.retrieval import BM25, fuse, rank, recall_at_k, reciprocal_rank, tokenize
from finqa_trust.stats import bootstrap_ci, paired_bootstrap_diff


def chunk(*units: str) -> Chunk:
    return Chunk(page="p", strategy="s", order=0, text="", units=list(units))


def test_tokenize_keeps_a_figure_whole():
    assert tokenize("Net revenue was $1,234.5 in 2008.") == ["net", "revenue", "was", "1,234.5", "in", "2008"]


def test_rare_word_outweighs_common_word():
    bm25 = BM25(["revenue grew", "revenue fell", "revenue flat", "goodwill impairment"])
    scores = bm25.scores("revenue goodwill")
    assert scores[3] > scores[0]


def test_repeats_saturate():
    bm25 = BM25(["cash", "cash cash cash cash cash cash", "other", "words"], b=0)
    scores = bm25.scores("cash")
    assert scores[0] < scores[1] < 2 * scores[0]


def test_longer_chunk_with_same_count_scores_lower():
    bm25 = BM25(["cash flow", "cash and many other words here", "other"])
    scores = bm25.scores("cash")
    assert scores[0] > scores[1]


def test_no_shared_word_scores_zero():
    assert BM25(["cash flow", "debt"]).scores("goodwill").tolist() == [0.0, 0.0]


def test_rank_skips_chunks_outside_the_scope():
    scores = np.array([0.9, 0.5, 0.7])
    assert rank(scores, np.array([False, True, True])) == [2, 1]


def test_fusion_prefers_agreement_over_one_first_place():
    # chunk 3 is 3rd in both lists: 2/63 beats chunk 1, first in one list only: 1/61
    assert fuse([[1, 2, 3], [4, 5, 3]])[0] == 3


def test_fusion_ignores_score_size():
    assert fuse([[7, 8], [8, 7]]) == [7, 8]  # a full tie, broken by chunk order


def test_recall_counts_the_share_of_gold_found():
    chunks = [chunk("text_1"), chunk("text_2"), chunk("text_3")]
    assert recall_at_k([0, 1, 2], chunks, ["text_1", "text_3", "text_9"], "p", k=2) == 1 / 3


def test_table_row_without_its_header_does_not_count():
    chunks = [chunk("table_4"), chunk("table_0", "table_4")]
    assert recall_at_k([0], chunks, ["table_4"], "p", k=1) == 0
    assert recall_at_k([1], chunks, ["table_4"], "p", k=1) == 1


def test_reciprocal_rank_uses_the_first_hit_inside_the_cutoff():
    chunks = [chunk("text_8"), chunk("text_1")]
    assert reciprocal_rank([0, 1], chunks, ["text_1"], "p") == 0.5
    assert reciprocal_rank([0, 1], chunks, ["text_1"], "p", cutoff=1) == 0


def test_same_unit_id_on_another_page_is_not_evidence():
    other = Chunk(page="q", strategy="s", order=0, text="", units=["text_1"])
    assert recall_at_k([0], [other], ["text_1"], "p", k=1) == 0
    assert reciprocal_rank([0], [other], ["text_1"], "p") == 0


def test_intervals_keep_fractions():
    # int(0.5) would be 0, so these would both read as zero
    assert bootstrap_ci([0.5, 0.5])[0] == 0.5
    assert paired_bootstrap_diff([0.0, 0.0], [0.5, 0.5])[0] == 0.5
