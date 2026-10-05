"""Upload documents to an OpenAI vector store and remove them again."""

from collections.abc import Callable
from pathlib import PurePath

from openai import NotFoundError, OpenAI

from rag import ocr
from rag.db import DB

VECTOR_STORE_KEY = "vector_store_id"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
TEXT_SUFFIXES = {".txt", ".md"}


def get_vector_store_id(client: OpenAI, db: DB) -> str:
    """Reuse the app's vector store, creating it on first run (or if it was deleted online)."""
    vs_id = db.get_setting(VECTOR_STORE_KEY)
    if vs_id:
        try:
            client.vector_stores.retrieve(vs_id)
            return vs_id
        except NotFoundError:
            pass
    vs_id = client.vector_stores.create(name="RAG-method2 documents").id
    db.set_setting(VECTOR_STORE_KEY, vs_id)
    return vs_id


def to_utf8(data: bytes) -> bytes:
    """File search requires UTF-8/UTF-16/ASCII text; Korean files are often CP949 (EUC-KR)."""
    for encoding in ("utf-8", "utf-16", "cp949"):
        try:
            return data.decode(encoding).encode("utf-8")
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace").encode("utf-8")


def prepare(client: OpenAI, filename: str, data: bytes,
            on_progress: Callable[[str], None]) -> tuple[str, bytes, bool]:
    """Return (upload filename, upload bytes, whether OCR was used)."""
    path = PurePath(filename)
    suffix = path.suffix.lower()
    if suffix in IMAGE_SUFFIXES:
        on_progress("Reading the image…")
        text = ocr.transcribe(client, [ocr.image_to_png(data)], on_progress)
        return f"{path.stem}.md", text.encode("utf-8"), True
    if suffix == ".pdf" and ocr.pdf_needs_ocr(data):
        on_progress("This looks like a scanned PDF — reading its pages…")
        text = ocr.transcribe(client, ocr.pdf_to_images(data), on_progress)
        return f"{path.stem}.md", text.encode("utf-8"), True
    if suffix in TEXT_SUFFIXES:
        return filename, to_utf8(data), False
    return filename, data, False


def ingest(client: OpenAI, db: DB, vs_id: str, filename: str, data: bytes,
           on_progress: Callable[[str], None] = lambda _: None) -> tuple[bool, str | None]:
    """Upload one document and wait until it is searchable. Returns (ok, error message)."""
    doc_id = db.add_document(filename)
    file_id = None
    try:
        upload_name, upload_data, used_ocr = prepare(client, filename, data, on_progress)
        on_progress("Uploading to OpenAI…")
        file_id = client.files.create(file=(upload_name, upload_data), purpose="assistants").id
        db.update_document(doc_id, openai_file_id=file_id, ocr=int(used_ocr))
        on_progress("Indexing for search…")
        vs_file = client.vector_stores.files.create_and_poll(file_id, vector_store_id=vs_id)
        if vs_file.status != "completed":
            reason = vs_file.last_error.message if vs_file.last_error else vs_file.status
            raise RuntimeError(reason)
    except Exception as exc:  # report any failure in the UI instead of crashing the app
        error = str(exc) or type(exc).__name__
        db.update_document(doc_id, status="failed", error=error)
        if file_id:
            _delete_remote(client, vs_id, file_id)
        return False, error
    db.update_document(doc_id, status="ready")
    return True, None


def delete(client: OpenAI, db: DB, vs_id: str, doc) -> None:
    if doc["openai_file_id"]:
        _delete_remote(client, vs_id, doc["openai_file_id"])
    db.delete_document(doc["id"])


def _delete_remote(client: OpenAI, vs_id: str, file_id: str) -> None:
    try:
        client.vector_stores.files.delete(file_id, vector_store_id=vs_id)
    except NotFoundError:
        pass
    try:
        client.files.delete(file_id)
    except NotFoundError:
        pass
