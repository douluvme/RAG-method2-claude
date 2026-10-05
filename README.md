# RAG-method2-claude

A personal web app for asking questions about your own documents. You upload PDFs, Word files, text/Markdown files or scanned images, then ask questions in a chat. Answers come **only** from your documents, with numbered sources and the quoted passages they came from. If the documents don't contain the answer, the app replies **"No information"**.

It runs on your Mac and opens in your web browser. Korean and English both work: the app answers in the language of your question.

## How it works

| Piece | What it does |
|---|---|
| **Streamlit** | The web page: the upload button, documents list, chat window and chat history. |
| **OpenAI File Search** | OpenAI indexes your documents, finds the passages relevant to each question, and returns `file_citation` markers that point to the documents it used. |
| **GPT-6 Luna / GPT-6.1 Sol** | Writes the answer. You choose the model in the sidebar. Luna (the default) is fast and cheap; Sol gives better answers. |
| **GPT-6 Luna (vision)** | Reads scanned PDFs and images page by page and turns them into searchable text. |
| **SQLite** (`data/app.db`) | Saves your documents list and chat history on your Mac. |

The app shows "No information" whenever **any** of these is true:
- the search found no passage above the relevance threshold,
- the answer cites no document,
- the model itself replied "No information".

## Setup (once)

1. Install **uv**, the tool that installs Python and the app's libraries, if you don't have it:
   ```
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```
2. Create an OpenAI API key at https://platform.openai.com/api-keys.
3. In this folder, copy `.env.example` to a file named `.env` and paste your key:
   ```
   OPENAI_API_KEY=sk-...
   ```
   `.env` is excluded from git, so your key is never uploaded to GitHub.

## Start the app

```
uv run streamlit run app.py
```

The first run installs everything automatically. Your browser opens at http://localhost:8501. To stop the app, press **Ctrl+C** in the terminal.

## Using it

- **Upload:** pick one or more files in the sidebar. Each file shows its progress and then **"✅ … is ready"** once it can be searched. Scanned files take longer because each page is read by the model.
- **Ask:** type a question in the chat box. Follow-up questions ("what about the second one?") work within the same chat.
- **Sources:** each answer has markers like [1] and [2]. Expand a source under the answer to see the file name and the quoted passage.
- **Manage:** delete documents (🗑) in the **Documents** list. Start or reopen chats in the **Chats** list.

## Costs (OpenAI pricing, Oct 2026)

- Storing documents: free up to 1 GB.
- Searching: $2.50 per 1,000 questions, about ¼ cent each.
- Answers: a small fraction of a cent with Luna, about 1–2 cents with Sol.
- Scanned pages: a fraction of a cent per page, charged once at upload.

## Tuning

Settings are in `rag/config.py`:
- `SCORE_THRESHOLD`: raise it if irrelevant passages show up as sources. Lower it if you get "No information" for things the documents do cover.
- `MAX_SEARCH_RESULTS`: how many passages are retrieved for each question.
- `MODELS` / `DEFAULT_MODEL`: the model choices in the sidebar.

## For developers

```
uv run pytest        # run the tests
```

The code:
- `app.py`: the user interface.
- `rag/ingest.py`: uploading and deleting documents.
- `rag/ocr.py`: reading scanned pages.
- `rag/answer.py`: answering, citations and the "No information" rule.
- `rag/db.py`: local storage.
