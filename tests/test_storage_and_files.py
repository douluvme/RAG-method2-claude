import io

from pypdf import PdfWriter

from rag import ingest, ocr
from rag.db import DB


def test_db_documents_and_chats_roundtrip(tmp_path):
    db = DB(tmp_path / "app.db")
    doc = db.add_document("report.pdf")
    db.update_document(doc, openai_file_id="file-a", status="ready")
    failed = db.add_document("bad.pdf")
    db.update_document(failed, status="failed", error="boom")
    assert db.ready_filenames() == {"file-a": "report.pdf"}

    chat = db.create_chat("What grew?")
    db.add_message(chat, "user", "What grew?")
    db.add_message(chat, "assistant", "Revenue.[1]", model="gpt-6-luna",
                   sources=[{"n": 1, "filename": "보고서.pdf", "passages": ["매출 증가"]}])
    msgs = db.list_messages(chat)
    assert [m["role"] for m in msgs] == ["user", "assistant"]
    assert msgs[1]["sources"][0]["filename"] == "보고서.pdf"

    db.delete_chat(chat)
    assert db.list_messages(chat) == []

    # data survives reopening
    assert DB(tmp_path / "app.db").ready_filenames() == {"file-a": "report.pdf"}


def test_settings(tmp_path):
    db = DB(tmp_path / "app.db")
    assert db.get_setting("vector_store_id") is None
    db.set_setting("vector_store_id", "vs_1")
    db.set_setting("vector_store_id", "vs_2")
    assert db.get_setting("vector_store_id") == "vs_2"


def test_korean_cp949_text_is_converted_to_utf8():
    assert ingest.to_utf8("한국어 문서".encode("cp949")) == "한국어 문서".encode("utf-8")
    assert ingest.to_utf8(b"plain") == b"plain"


def test_pdf_without_text_needs_ocr():
    writer = PdfWriter()
    writer.add_blank_page(width=612, height=792)
    buf = io.BytesIO()
    writer.write(buf)
    assert ocr.pdf_needs_ocr(buf.getvalue())
    assert len(ocr.pdf_to_images(buf.getvalue())) == 1
