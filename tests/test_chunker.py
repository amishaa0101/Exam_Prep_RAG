import pytest

from chunker import chunk_all_documents, chunk_document, chunk_question_paper, extract_sections, recursive_chunk

BODY = "body text that is definitely more than thirty characters long"


def headers_for(line):
    return [s["header"] for s in extract_sections(f"intro text that is long enough here\n{line}\n{BODY}")]


@pytest.mark.parametrize("line", [
    "NORMALIZATION", "ER MODEL", "## Merge Sort", "1.1 Sorting Algorithms Overview",
    "3.1. Attribute types :", "Chapter 4: Transactions", "Types of integrity constraints :",
])
def test_real_headers_detected(line):
    assert line.lstrip("#").strip() in headers_for(line)


@pytest.mark.parametrize("line", [
    "6:29 PM", "A = 1100", "B = 3000", "W(A) - W(B)", "[BC, BCD, BCE, BCDE]",
    "T0,T1,T2,T3", "AB AB", "2. Repeat V-1 times:", "1 Ritesh 9",
])
def test_formulas_and_list_items_not_headers(line):
    assert line not in headers_for(line)


def test_recursive_chunk_respects_size():
    text = ". ".join(f"Sentence number {i} about databases" for i in range(100))
    chunks = recursive_chunk(text, chunk_size=200, overlap=20)
    assert len(chunks) > 1
    assert all(len(c) <= 300 for c in chunks)  # 1.5x tolerance used by the splitter


def test_short_text_is_single_chunk():
    assert recursive_chunk("short text", chunk_size=500) == ["short text"]


def test_overlap_not_smaller_than_chunk_size_rejected():
    # Used to loop forever in the hard-split fallback
    with pytest.raises(ValueError):
        recursive_chunk("x" * 1000, chunk_size=100, overlap=100)


def test_chunk_document_prepends_header_and_filters_noise():
    doc = {
        "source": "notes.md",
        "text": "## Merge Sort\n\nMerge sort is a divide and conquer algorithm that splits the array.\n\n"
                "## Diagram\n\n|| -- || 12 34 56 78 90 12 34 56 78 90 12 34 56 ||",
    }
    chunks = chunk_document(doc)
    assert len(chunks) == 1, "the mostly non-alphabetic section should be filtered out"
    assert chunks[0]["text"].startswith("Merge Sort\n\n")
    assert chunks[0]["metadata"]["section"] == "Merge Sort"


def test_chunk_ids_unique_and_sequential():
    text = "\n\n".join(f"## Section {i}\n\n" + "Some words in this section body. " * 30 for i in range(5))
    chunks = chunk_document({"source": "s.md", "text": text}, chunk_size=200, overlap=20)
    ids = [c["id"] for c in chunks]
    assert len(ids) == len(set(ids))
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))


def test_question_paper_numbering_and_answers_stay_together():
    doc = {
        "source": "pyq_2020.txt",
        "text": "Q1. What is BCNF? Explain with an example relation and FDs please.\n"
                "Answer: Step 2. take the closure of the attribute set and compare it.\n"
                "Q2. Define 2PL and explain the growing and shrinking phases in detail.",
    }
    chunks = chunk_question_paper(doc)
    assert [c["metadata"]["section"] for c in chunks] == ["Question 1", "Question 2"]
    assert "closure" in chunks[0]["text"], "the answer belongs to question 1"
    assert chunks[0]["id"] == "pyq_2020.txt_chunk_0"


@pytest.mark.parametrize("name,expected", [
    ("pyq_2020.txt", "question paper"), ("gate_exam.pdf", "question paper"),
    ("mid-term-papers.md", "question paper"),
    ("latest_notes.md", "recursive"), ("contest_prep.md", "recursive"), ("protest.md", "recursive"),
])
def test_question_paper_routing(name, expected, capsys):
    chunk_all_documents([{"source": name, "text": "## Topic\n\n" + "A normal lecture notes sentence. " * 5}])
    assert expected in capsys.readouterr().out


def test_difficulty_definitions_basic_problems_advanced():
    from chunker import estimate_difficulty
    definition = ("Header\n\nA weak entity set is called weak because it does not have a key. "
                  "It is defined as an entity set which needs an owner. Types of attributes are used to describe it.")
    problem = ("Header\n\nQ. how many super keys? sol. R(ABCDEF) [A]+ = ABCDEF, total = 2^5 = 32, "
               "maximum = 2n-1 = 63, minimum = 1")
    assert estimate_difficulty(definition)[0] == "basic"
    assert estimate_difficulty(problem)[0] == "advanced"


def test_chunks_carry_difficulty_metadata():
    chunks = chunk_document({"source": "s.md", "text": "## Topic\n\n" + "Merge sort is called a stable sort. " * 10})
    assert chunks and all(c["metadata"]["difficulty"] in ("basic", "intermediate", "advanced") for c in chunks)
