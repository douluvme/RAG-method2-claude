"""Turn scanned PDFs and images into text by having an OpenAI model read each page."""

import base64
import io
from collections.abc import Callable

import pypdfium2 as pdfium
from openai import OpenAI
from PIL import Image
from pypdf import PdfReader

from rag import config

# A PDF averaging fewer extractable characters per page than this is treated as scanned.
MIN_CHARS_PER_PAGE = 20
# Longest image side sent to the model; larger scans are shrunk to this.
MAX_IMAGE_SIDE = 2000

TRANSCRIBE_PROMPT = (
    "Transcribe all text on this page exactly as written, keeping the original language "
    "(Korean, English, or both). Keep the reading order. Use Markdown for headings, lists "
    "and tables. Output only the transcription, with no commentary. "
    "If the page has no readable text, output exactly: [blank page]"
)


def pdf_needs_ocr(data: bytes) -> bool:
    reader = PdfReader(io.BytesIO(data))
    if not reader.pages:
        return False
    chars = sum(len((page.extract_text() or "").strip()) for page in reader.pages)
    return chars / len(reader.pages) < MIN_CHARS_PER_PAGE


def pdf_to_images(data: bytes) -> list[bytes]:
    pdf = pdfium.PdfDocument(data)
    images = []
    for page in pdf:
        pil = page.render(scale=2).to_pil()  # ~144 dpi
        images.append(_to_png(pil))
    return images


def image_to_png(data: bytes) -> bytes:
    return _to_png(Image.open(io.BytesIO(data)))


def _to_png(img: Image.Image) -> bytes:
    img = img.convert("RGB")
    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def transcribe_page(client: OpenAI, png: bytes) -> str:
    data_url = "data:image/png;base64," + base64.b64encode(png).decode()
    response = client.responses.create(
        model=config.OCR_MODEL,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_text", "text": TRANSCRIBE_PROMPT},
                {"type": "input_image", "image_url": data_url, "detail": "high"},
            ],
        }],
    )
    return response.output_text.strip()


def transcribe(client: OpenAI, pages: list[bytes],
               on_progress: Callable[[str], None] = lambda _: None) -> str:
    """Return Markdown with one '## Page N' section per page."""
    parts = []
    for n, png in enumerate(pages, start=1):
        on_progress(f"Reading page {n} of {len(pages)}…")
        parts.append(f"## Page {n}\n\n{transcribe_page(client, png)}")
    return "\n\n".join(parts) + "\n"
