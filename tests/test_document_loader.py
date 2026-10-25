from document_loader import clean_pdf_text, load_documents, load_text_file


def test_removes_page_numbers_and_timestamps():
    text = "Intro line\nMain notes Page 42\n12\n6:29 PM\nBody line"
    cleaned = clean_pdf_text(text)
    assert "Page 42" not in cleaned
    assert "6:29 PM" not in cleaned
    assert "\n12\n" not in f"\n{cleaned}\n"
    assert "Intro line" in cleaned and "Body line" in cleaned


def test_back_to_back_number_lines_all_removed():
    # Used to remove only every other line
    cleaned = clean_pdf_text("start text\n30\n40\n50\nend text")
    for n in ("30", "40", "50"):
        assert n not in cleaned.split()


def test_keeps_short_meaningful_lines():
    # Relation and attribute names in DBMS notes are often 2 characters
    cleaned = clean_pdf_text("Relations:\nR1\nAB\nX\nend")
    lines = cleaned.split("\n")
    assert "R1" in lines and "AB" in lines
    assert "X" not in lines  # single characters are diagram artifacts


def test_skips_diagram_labels():
    cleaned = clean_pdf_text("text\nE(1)\nE(2)\nmore text")
    assert "E(1)" not in cleaned and "E(2)" not in cleaned


def test_collapses_blank_lines_and_spaces():
    cleaned = clean_pdf_text("a line\n\n\n\n\nanother     line")
    assert "\n\n\n" not in cleaned
    assert "     " not in cleaned


def test_load_text_file(tmp_path):
    f = tmp_path / "notes.md"
    f.write_text("# Title\n\nSome content", encoding="utf-8")
    doc = load_text_file(str(f))
    assert doc["source"] == "notes.md"
    assert "Some content" in doc["text"]


def test_load_documents_filters_extensions(tmp_path):
    (tmp_path / "a.md").write_text("markdown notes", encoding="utf-8")
    (tmp_path / "b.txt").write_text("text notes", encoding="utf-8")
    (tmp_path / "c.csv").write_text("x,y", encoding="utf-8")
    sub = tmp_path / "sub"
    sub.mkdir()
    (sub / "d.md").write_text("nested notes", encoding="utf-8")

    sources = sorted(d["source"] for d in load_documents(str(tmp_path)))
    assert sources == ["a.md", "b.txt", "d.md"]
