from types import SimpleNamespace as NS

from rag import answer, config


def citation(file_id, index, filename="upload.pdf"):
    return NS(type="file_citation", file_id=file_id, index=index, filename=filename)


def result(file_id, text, score=0.8):
    return NS(file_id=file_id, text=text, score=score, filename="upload.pdf", attributes=None)


def response(text, annotations=(), results=()):
    return NS(output=[
        NS(type="file_search_call", results=list(results)),
        NS(type="message", content=[
            NS(type="output_text", text=text, annotations=list(annotations)),
        ]),
    ])


FILENAMES = {"file-a": "report.pdf", "file-b": "보고서.docx"}


def test_cited_answer_gets_markers_and_sources():
    text = "Revenue grew 12%. Costs fell."
    resp = response(
        text,
        annotations=[citation("file-a", 17), citation("file-b", 29)],
        results=[result("file-a", "Revenue grew 12% in Q3."),
                 result("file-b", "Costs fell by 3%.")],
    )
    got = answer.parse_response(resp, FILENAMES)
    assert got.text == "Revenue grew 12%.[1] Costs fell.[2]"
    assert got.sources == [
        {"n": 1, "filename": "report.pdf", "passages": ["Revenue grew 12% in Q3."]},
        {"n": 2, "filename": "보고서.docx", "passages": ["Costs fell by 3%."]},
    ]
    assert not got.is_no_info


def test_same_file_cited_twice_shares_one_number():
    resp = response(
        "A. B.",
        annotations=[citation("file-a", 2), citation("file-a", 5)],
        results=[result("file-a", "low", score=0.5), result("file-a", "high", score=0.9),
                 result("file-a", "lowest", score=0.4)],
    )
    got = answer.parse_response(resp, FILENAMES)
    assert got.text == "A.[1] B.[1]"
    assert len(got.sources) == 1
    # best-scoring passages first, capped
    assert got.sources[0]["passages"] == ["high", "low"]


def test_two_files_cited_at_same_position():
    resp = response("Fact.", annotations=[citation("file-b", 5), citation("file-a", 5)],
                    results=[result("file-a", "x"), result("file-b", "y")])
    got = answer.parse_response(resp, FILENAMES)
    assert got.text == "Fact.[1][2]"


def test_no_search_results_means_no_information():
    resp = response("Paris is sunny.", annotations=[citation("file-a", 5)], results=[])
    got = answer.parse_response(resp, FILENAMES)
    assert got.text == config.NO_INFO
    assert got.sources == [] and got.is_no_info


def test_uncited_answer_means_no_information():
    resp = response("Probably 42.", results=[result("file-a", "unrelated")])
    assert answer.parse_response(resp, FILENAMES).text == config.NO_INFO


def test_model_saying_no_information_drops_sources():
    resp = response("No information.", annotations=[citation("file-a", 14)],
                    results=[result("file-a", "unrelated")])
    got = answer.parse_response(resp, FILENAMES)
    assert got.text == config.NO_INFO and got.sources == []


def test_inline_citation_tokens_are_removed():
    resp = response("Fact【4:0†report.pdf】.", annotations=[citation("file-a", 4)],
                    results=[result("file-a", "x")])
    assert answer.parse_response(resp, FILENAMES).text == "Fact[1]."


def test_unknown_file_falls_back_to_annotation_filename():
    resp = response("Fact.", annotations=[citation("file-z", 5, filename="old.md")],
                    results=[result("file-z", "x")])
    assert answer.parse_response(resp, FILENAMES).sources[0]["filename"] == "old.md"


def test_ask_skips_api_when_no_documents():
    class Boom:
        def __getattr__(self, name):
            raise AssertionError("API should not be called")
    got = answer.ask(Boom(), "vs", "gpt-6-luna", [], "anything?", filenames={})
    assert got.text == config.NO_INFO


def test_build_input_keeps_recent_history_then_question():
    history = [{"role": "user" if i % 2 == 0 else "assistant", "text": f"m{i}"} for i in range(20)]
    items = answer.build_input(history, "follow-up?")
    assert len(items) == config.HISTORY_MESSAGES + 1
    assert items[0] == {"role": "user", "content": "m8"}
    assert items[-1] == {"role": "user", "content": "follow-up?"}
