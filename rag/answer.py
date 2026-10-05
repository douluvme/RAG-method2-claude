"""Answer a question from the uploaded documents, with numbered sources."""

import re
from dataclasses import dataclass, field

from openai import OpenAI

from rag import config

INSTRUCTIONS = f"""You answer questions using ONLY the user's uploaded documents, which you reach through the file_search tool.

Rules:
- Always search the documents before answering.
- Use only information found in the search results. Never use outside knowledge, and never guess.
- Cite the documents you used.
- Answer in the same language as the user's latest question (Korean or English), even if the documents are in another language.
- If the search results do not contain the answer, reply with exactly: {config.NO_INFO}
  Reply with nothing else in that case — no apology, no explanation, no suggestions.
- Earlier messages in the conversation are only context for follow-up questions; they are not a source of facts."""

# Citation tokens some models write into the text itself; the real citations are annotations.
_INLINE_CITATION = re.compile(r"【[^】]*】|.*?")
PASSAGES_PER_SOURCE = 2


@dataclass
class Answer:
    text: str
    sources: list[dict] = field(default_factory=list)  # [{"n", "filename", "passages": [str]}]

    @property
    def is_no_info(self) -> bool:
        return not self.sources


def no_info() -> Answer:
    return Answer(config.NO_INFO)


def build_input(history: list[dict], question: str) -> list[dict]:
    recent = history[-config.HISTORY_MESSAGES:] if config.HISTORY_MESSAGES else []
    items = [{"role": m["role"], "content": m["text"]} for m in recent]
    items.append({"role": "user", "content": question})
    return items


def ask(client: OpenAI, vs_id: str, model: str, history: list[dict], question: str,
        filenames: dict[str, str]) -> Answer:
    if not filenames:  # nothing searchable yet: don't spend an API call
        return no_info()
    response = client.responses.create(
        model=model,
        instructions=INSTRUCTIONS,
        input=build_input(history, question),
        tools=[{
            "type": "file_search",
            "vector_store_ids": [vs_id],
            "max_num_results": config.MAX_SEARCH_RESULTS,
            "ranking_options": {"score_threshold": config.SCORE_THRESHOLD},
        }],
        include=["file_search_call.results"],
    )
    return parse_response(response, filenames)


def _looks_like_no_info(text: str) -> bool:
    normalized = text.strip().strip(".!。 ").lower()
    return normalized == config.NO_INFO.lower()


def parse_response(response, filenames: dict[str, str]) -> Answer:
    """Turn a Responses API result into an Answer, enforcing the "No information" rule.

    The answer is "No information" unless file search found passages AND the model's
    reply cites at least one of them.
    """
    results = []
    text, annotations = "", []
    for item in response.output:
        if item.type == "file_search_call":
            results.extend(item.results or [])
        elif item.type == "message":
            for part in item.content:
                if part.type == "output_text":
                    # Annotation indexes are relative to this part, so shift them.
                    annotations.extend(
                        (len(text) + a.index, a) for a in part.annotations
                        if a.type == "file_citation"
                    )
                    text += part.text

    if not results or not annotations or _looks_like_no_info(text):
        return no_info()

    # Number cited files in order of first appearance.
    numbers: dict[str, int] = {}
    for _, a in sorted(annotations, key=lambda pair: pair[0]):
        numbers.setdefault(a.file_id, len(numbers) + 1)

    # Insert [n] markers at the citation positions, working backwards so indexes stay valid.
    markers: dict[int, list[int]] = {}
    for index, a in annotations:
        at = markers.setdefault(index, [])
        if numbers[a.file_id] not in at:
            at.append(numbers[a.file_id])
    for index in sorted(markers, reverse=True):
        tag = "".join(f"[{n}]" for n in sorted(markers[index]))
        text = text[:index] + tag + text[index:]
    text = _INLINE_CITATION.sub("", text).strip()

    sources = []
    for file_id, n in sorted(numbers.items(), key=lambda kv: kv[1]):
        hits = sorted((r for r in results if r.file_id == file_id),
                      key=lambda r: r.score or 0, reverse=True)
        fallback = next((a.filename for _, a in annotations if a.file_id == file_id), file_id)
        sources.append({
            "n": n,
            "filename": filenames.get(file_id, fallback),
            "passages": [r.text.strip() for r in hits[:PASSAGES_PER_SOURCE] if r.text],
        })
    return Answer(text, sources)
