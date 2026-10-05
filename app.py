"""Document Q&A: upload documents, ask questions, get answers with sources.

Start with:  uv run streamlit run app.py
"""

import openai
import streamlit as st

from rag import answer, config, ingest
from rag.db import DB

st.set_page_config(page_title="Document Q&A", page_icon="📄", layout="wide")

if not config.api_key():
    st.error("**OpenAI API key missing.**")
    st.markdown(
        "1. Copy `.env.example` to a new file named `.env` in the project folder.\n"
        "2. Put your key on the line `OPENAI_API_KEY=sk-...` "
        "(create one at https://platform.openai.com/api-keys).\n"
        "3. Stop the app (Ctrl+C in the terminal) and start it again."
    )
    st.stop()


@st.cache_resource
def get_db() -> DB:
    return DB()


@st.cache_resource
def get_client() -> openai.OpenAI:
    return openai.OpenAI(api_key=config.api_key())


@st.cache_resource
def get_vector_store_id() -> str:
    return ingest.get_vector_store_id(get_client(), get_db())


db, client = get_db(), get_client()
try:
    vs_id = get_vector_store_id()
except openai.AuthenticationError:
    st.error("OpenAI rejected the API key in `.env`. Check it and restart the app.")
    st.stop()
except openai.APIConnectionError:
    st.error("Can't reach OpenAI. Check your internet connection, then reload this page.")
    st.stop()

state = st.session_state
state.setdefault("chat_id", None)
state.setdefault("uploader_key", 0)
state.setdefault("notices", [])


def render_sources(sources: list[dict]) -> None:
    if not sources:
        return
    st.caption("Sources")
    for s in sources:
        with st.expander(f"[{s['n']}] {s['filename']}"):
            if s["passages"]:
                for passage in s["passages"]:
                    st.markdown("\n".join(f"> {line}" for line in passage.splitlines()))
            else:
                st.caption("No passage text was returned for this source.")


def render_message(m: dict) -> None:
    with st.chat_message(m["role"]):
        st.markdown(m["text"])
        if m["role"] == "assistant":
            render_sources(m["sources"])
            if m["model"]:
                st.caption(config.MODELS.get(m["model"], m["model"]).split(" — ")[0])


# ---------------- Sidebar ----------------
with st.sidebar:
    st.header("⚙️ Model")
    model = st.selectbox("Answer with", list(config.MODELS), format_func=config.MODELS.get,
                         index=list(config.MODELS).index(config.DEFAULT_MODEL),
                         label_visibility="collapsed")

    st.header("📤 Upload documents")
    files = st.file_uploader("PDF, Word, text, Markdown, or scanned images",
                             type=config.UPLOAD_TYPES, accept_multiple_files=True,
                             key=f"uploader_{state.uploader_key}")
    if files:
        state.notices = []
        for f in files:
            with st.status(f"Processing {f.name}…", expanded=True) as status:
                ok, error = ingest.ingest(client, db, vs_id, f.name, f.getvalue(),
                                          on_progress=status.write)
                if ok:
                    status.update(label=f"✅ {f.name} is ready", state="complete", expanded=False)
                    state.notices.append(("success", f"✅ **{f.name}** is uploaded and ready to search."))
                else:
                    status.update(label=f"❌ {f.name} failed", state="error")
                    state.notices.append(("error", f"❌ **{f.name}** failed: {error}"))
        state.uploader_key += 1  # clear the uploader so files aren't processed twice
        st.rerun()
    for kind, text in state.notices:
        (st.success if kind == "success" else st.error)(text)

    st.header("📚 Documents")
    docs = db.list_documents()
    if not docs:
        st.caption("No documents yet.")
    icons = {"ready": "✅", "processing": "⏳", "failed": "❌"}
    for doc in docs:
        name_col, del_col = st.columns([5, 1])
        label = f"{icons.get(doc['status'], '')} {doc['filename']}"
        if doc["ocr"]:
            label += " · scanned"
        name_col.markdown(label, help=doc["error"] or None)
        if del_col.button("🗑", key=f"del_doc_{doc['id']}", help="Delete this document"):
            with st.spinner(f"Deleting {doc['filename']}…"):
                ingest.delete(client, db, vs_id, doc)
            state.notices = [("success", f"🗑 Deleted **{doc['filename']}**.")]
            st.rerun()

    st.header("💬 Chats")
    if st.button("➕ New chat", width="stretch"):
        state.chat_id = None
        st.rerun()
    for chat in db.list_chats():
        title_col, del_col = st.columns([5, 1])
        current = chat["id"] == state.chat_id
        if title_col.button(chat["title"], key=f"chat_{chat['id']}", width="stretch",
                            type="primary" if current else "secondary"):
            state.chat_id = chat["id"]
            st.rerun()
        if del_col.button("🗑", key=f"del_chat_{chat['id']}", help="Delete this chat"):
            db.delete_chat(chat["id"])
            if current:
                state.chat_id = None
            st.rerun()

# ---------------- Chat ----------------
st.title("📄 Ask your documents")
if not db.ready_filenames():
    st.info("Upload documents in the sidebar, then ask questions about them here.")

history = db.list_messages(state.chat_id) if state.chat_id else []
for m in history:
    render_message(m)

if question := st.chat_input("Ask a question about your documents…"):
    render_message({"role": "user", "text": question})
    with st.chat_message("assistant"):
        try:
            with st.spinner("Searching your documents…"):
                result = answer.ask(client, vs_id, model, history, question, db.ready_filenames())
        except openai.APIError as exc:
            st.error(f"OpenAI request failed: {exc}")
            st.stop()
    if state.chat_id is None:
        title = question if len(question) <= 40 else question[:40] + "…"
        state.chat_id = db.create_chat(title)
    db.add_message(state.chat_id, "user", question)
    db.add_message(state.chat_id, "assistant", result.text, model=model, sources=result.sources)
    st.rerun()
