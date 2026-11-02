from finqa_trust.chunks import Chunk, by_sentence, chunk_page, fixed_size, holds_evidence, page_units, table_whole


def page(rows: int = 3, sentence_words: int = 10) -> dict:
    sentence = " ".join(["word"] * sentence_words)
    return {
        "filename": "V/2008/page_17.pdf",
        "pre_text": [sentence, ".", sentence],
        "post_text": [sentence],
        "table": [["item", "2016", "2017"]] + [[f"row{i}", "1", "2"] for i in range(rows - 1)],
    }


def chunk(units: list[str]) -> Chunk:
    return Chunk(page="p", strategy="s", order=0, text="", units=units)


def test_unit_ids_match_gold_indexes_across_pre_and_post_text():
    ids = [u.id for u in page_units(page())]
    # text_1 is the OCR full stop; text_3 is the first post_text line because indexing runs over pre + post
    assert ids == ["text_0", "text_2", "table_0", "table_1", "table_2", "text_3"]


def test_fixed_size_does_not_list_a_sentence_it_cuts():
    pieces = fixed_size(page_units(page(rows=1, sentence_words=10)), size=15, overlap=0)
    assert "text_2" not in pieces[0][1]  # only 5 of its 10 words fit
    assert "text_2" not in pieces[1][1]  # the other 5 are here


def test_fixed_size_overlap_can_hold_a_sentence_whole():
    pieces = fixed_size(page_units(page(rows=1, sentence_words=10)), size=15, overlap=10)
    assert any("text_2" in units for _, units in pieces)


def test_fixed_size_reaches_the_last_word_once():
    units = page_units(page())
    pieces = fixed_size(units, size=15, overlap=5)
    assert pieces[-1][0].split()[-1] == units[-1].text.split()[-1]
    assert len(pieces[-1][0].split()) > 5  # not a tail made only of overlap


def test_by_sentence_keeps_every_unit_whole():
    units = page_units(page())
    held = [unit for _, whole in by_sentence(units, max_words=15) for unit in whole]
    assert held == [u.id for u in units]


def test_by_sentence_can_separate_a_row_from_its_header():
    units = page_units(page(rows=8))
    pieces = by_sentence(units, max_words=15)
    assert any("table_7" in whole and "table_0" not in whole for _, whole in pieces)


def test_table_whole_puts_all_rows_in_one_chunk_with_nothing_else():
    units = page_units(page(rows=20))
    tables = [whole for _, whole in table_whole(units, max_words=15) if "table_0" in whole]
    assert len(tables) == 1
    assert tables[0] == [f"table_{i}" for i in range(20)]


def test_table_whole_handles_a_page_with_no_table():
    item = page()
    item["table"] = []
    assert table_whole(page_units(item), max_words=15)


def test_a_row_without_its_header_is_not_evidence():
    assert holds_evidence(chunk(["table_0", "table_3"]), "table_3")
    assert not holds_evidence(chunk(["table_3"]), "table_3")


def test_a_sentence_needs_no_header_but_must_be_listed():
    assert holds_evidence(chunk(["text_4"]), "text_4")
    assert not holds_evidence(chunk(["text_5"]), "text_4")


def test_chunk_page_records_page_and_order():
    chunks = chunk_page(page(), "sentence")
    assert [c.order for c in chunks] == list(range(len(chunks)))
    assert all(c.page == "V/2008/page_17.pdf" and c.strategy == "sentence" for c in chunks)
