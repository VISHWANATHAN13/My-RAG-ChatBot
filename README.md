### WELCOME TO MY RAG BASED AI CHATBOT
# College Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions
about college policies, courses, facilities, and FAQs using a college
knowledge base. The project uses LangChain for retrieval and LLM
orchestration, and Gradio for the chat interface.

> **Project status:** RAG pipeline with **hybrid retrieval
> (dense + BM25, fused with Reciprocal Rank Fusion)** and a Gradio UI.
> Cross-encoder reranking and formal evaluation are still planned.

## Features

-   Ask natural-language questions about college information.
-   **Hybrid retrieval:** dense vector search (Chroma +
    `text-embedding-3-small`) combined with sparse keyword search
    (BM25 Okapi), merged by Reciprocal Rank Fusion.
-   Generate answers using retrieved context.
-   Display source metadata for retrieved documents.
-   Chat-style UI with both an **Enter** button and keyboard submission.
-   A self-contained diagnostic script that proves the hybrid retriever
    is actually fusing, not just concatenating.

## Architecture

``` text
User question
     |
     v
Gradio Blocks UI  (ui.py)
     |
     v
ask_question(question)                          (rag.py)
     |
     v
hybrid_search(query, bm25, stored_data, top_k=5)
     |
     +--> semantic_retrive(query, k=3)
     |       Chroma similarity_search over OpenAI embeddings
     |
     +--> bm25_retrive(query, bm25, stored_data, k=20)
     |       BM25Okapi scores over the same chunk corpus
     |
     +--> reciprocal_rank_fusion([dense, bm25], k=60)
     |       score(doc) = sum over lists of 1 / (60 + rank + 1)
     |
     +--> top_k fused chunks
     |
     +--> Retrieved page content is combined as context
     |
     +--> Prompt receives context + question
     |
     +--> LLM generates an answer (gpt-4o-mini, temperature=0)
     |
     v
Answer + fused documents
     |
     +--> Chatbot displays answer
     +--> Sources section displays document source metadata
```

## Tech Stack

-   Python
-   LangChain
-   Gradio
-   ChromaDB (dense retrieval)
-   `rank_bm25` / BM25Okapi (sparse retrieval)
-   Reciprocal Rank Fusion (custom implementation in `rag.py`)
-   OpenAIEmbeddings (`text-embedding-3-small`)
-   ChatOpenAI (`gpt-4o-mini`)
-   RecursiveCharacterTextSplitter

## Project Structure

``` text
My-RAG-ChatBot/
├── app/
│   ├── ui.py              # Gradio UI and chat callback
│   ├── rag.py             # Hybrid search (dense + BM25 + RRF) and answer generation
│   ├── bm25_retrival.py   # BM25 index build and sparse retrieval
│   ├── vectorstore.py     # Embeddings, Chroma create/load, dense retrieval
│   ├── ingest.py          # Loaders (txt/csv/html/json) - splitting and chunking
│   └── check_hybrid.py    # Diagnostic for the hybrid retriever
├── chroma_db/             # Persisted Chroma collection "college_knowledge"
├── data/
│   ├── policies/          # college_handbook.txt
│   ├── courses/           # course_details.csv
│   ├── facilities/        # facilities.html
│   └── faqs/              # college_faqs.json
├── .env                   # create a .env file and paste your API key here
├── requirements.txt       # Python dependencies
└── README.md
```


## Prerequisites

-   Python installed (use a version supported by your dependencies).
-   A virtual environment.
-   API credentials for the LLM provider used by `ui.py`, if required.
-   Your knowledge-base files and vector index, if the project expects
    them to exist before launch.

## Setup

### 1. Open the project directory

``` bash
cd path/to/My-RAG-ChatBot
```

### 2. Create and activate a virtual environment

**Windows PowerShell:**

``` powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, you can use Command Prompt instead:

``` bat
.venv\Scripts\activate.bat
```

### 3. Install dependencies

If `requirements.txt` already exists:

``` bash
python -m pip install -r requirements.txt
```

Otherwise, install the packages your code imports, then save the
environment:

``` bash
python -m pip freeze > requirements.txt
```

For a reproducible project, review the generated file and keep only the
dependencies your application actually needs.

Hybrid search adds one dependency beyond the baseline RAG stack:

``` bash
python -m pip install rank_bm25
```

### 4. Configure environment variables

Create a `.env` file in the project root if your code loads credentials
from it. For example:

``` dotenv
OPENAI_API_KEY=your_api_key_here
```

Use the variable name expected by your code and provider. Do not commit
`.env` or paste real API keys into source code.

Add this to `.gitignore`:

``` gitignore
.env
.venv/
__pycache__/
*.py[cod]
```

### 5. Prepare the knowledge base

Place your college policies, course information, facilities documents,
and FAQs in the location expected by your ingestion code.

Inspect the loaders and chunking:

``` bash
python app/ingest.py
```

To build the Chroma collection, un-comment the `create_vector_store()`
call in `app/vectorstore.py` and run it:

``` bash
python app/vectorstore.py
```

The persisted index lives in `chroma_db/` (collection
`college_knowledge`). If it is already built, you do not need to ingest
again. The BM25 index needs no separate build step — it is derived from
the Chroma collection at import time.

## Run the application

From the project root, activate the virtual environment and run the
Gradio entry point:

``` bash
python app/ui.py
```

Gradio will print a local URL in the terminal, commonly:

``` text
http://127.0.0.1:7860
```

Open the URL in your browser. If your entry-point file is named
`ui.py` or something else, run that filename instead.

## How to use

1.  Open the Gradio URL shown in the terminal.
2.  Type a question in the message box.
3.  Click **Enter** or press the keyboard Enter key.
4.  Read the generated answer in the chat.
5.  Check the **Sources** section for the source metadata returned by
    retrieval.

Example questions:

-   What is the attendance requirement for students?
-   What courses are available in the Computer Science department?
-   What facilities are available in the college library?
-   What is the attendance requirement, and what happens if I do not
    meet it?

The assistant can only reliably answer from information available to its
knowledge base. If a topic is not covered, it should indicate that the
information is unavailable rather than inventing college policy.

## Current RAG flow

The current `ask_question()` function follows this pattern:

1.  Run `hybrid_search()` to retrieve fused documents for the question.
2.  Join each retrieved document's `content` into a context string.
3.  Pass `context` and `question` into the prompt.
4.  Invoke the configured chat model.
5.  Return the answer text and retrieved documents.

The UI callback unpacks both values:

``` python
answer, documents = ask_question(question)
```

Use `answer` as the chatbot response. Retrieval returns plain dicts of
the shape `{"content": ..., "metadata": ..., "score": ...}`, so the UI
reads `doc["metadata"]["source"]` rather than LangChain `Document`
attributes.

## Hybrid search

### Why hybrid

Dense and sparse retrieval fail in opposite ways, and the knowledge base
contains both kinds of query:

-   **Dense search alone** blurs rare exact tokens. A query for a course
    code such as `AIML201` embeds close to every other course row, so the
    right chunk may not surface.
-   **BM25 alone** is blind to paraphrase. "I need time off because I am
    unwell" shares almost no tokens with a leave-policy chunk, so lexical
    scoring never ranks it.

Running both and fusing the rankings recovers chunks that either
retriever alone would miss.

### Shared corpus

Both retrievers must index the same chunks, otherwise fused ranks are
meaningless. `rag.py` guarantees this by building the BM25 index from the
chunks already stored in Chroma, rather than re-reading `data/`:

``` python
vector_store = load_vectore_store()
stored_data  = vector_store.get()          # ids, documents, metadatas
bm25, bm25_documents = format_docs(stored_data)
```

`format_docs()` (in `bm25_retrival.py`) lowercases and whitespace-splits
each chunk into a tokenized corpus and hands it to `BM25Okapi`.

Because the index is built once at import time, **re-ingesting data
requires restarting the app** — otherwise BM25 keeps scoring a stale
corpus while Chroma serves the new one.

### The two retrievers

| Retriever | Function | Where | Default `k` | Scoring |
|---|---|---|---|---|
| Dense | `semantic_retrive()` | `rag.py` | 3 | Chroma `similarity_search` over `text-embedding-3-small` |
| Sparse | `bm25_retrive()` | `bm25_retrival.py` | 20 | `BM25Okapi.get_scores()` on lowercased tokens |

Both normalise their output to the same dict shape so the fusion step
does not need to know which retriever produced a candidate.

### Reciprocal Rank Fusion

`reciprocal_rank_fusion()` merges the two ranked lists using rank
position only — never the raw scores, which are on incomparable scales
(cosine distance vs. unbounded BM25 weights):

``` python
fused_scores[doc_id] += 1 / (k + rank + 1)     # k = 60
```

A chunk that appears in **both** lists accumulates two contributions, so
agreement between the retrievers pushes it to the top. Candidates are
then sorted by fused score and truncated to `top_k` (default 5).

### The fusion key

RRF needs a stable identity per chunk. The stored chunks carry no
`chunk_id`, and Chroma's `similarity_search` does not return its internal
ids, so `doc_key()` uses the **chunk text itself**:

``` python
def doc_key(doc):
    return doc["content"]
```

This is the one fragile point in the design: if the dense and sparse
paths ever normalise text differently (whitespace, casing, truncation),
keys stop colliding and RRF silently degrades into plain concatenation —
every chunk gets a single-list score and no agreement bonus is ever
awarded. There is no exception; the only symptom is slightly worse
answers. The diagnostic below exists to catch exactly that.

### Tuning knobs

| Knob | Location | Effect |
|---|---|---|
| `semantic_retrive(k=3)` | `rag.py` | Dense candidate depth |
| `bm25_retrive(k=20)` | `rag.py` `hybrid_search()` | Sparse candidate depth |
| `reciprocal_rank_fusion(k=60)` | `rag.py` | RRF damping; lower `k` favours top ranks more sharply |
| `hybrid_search(top_k=5)` | `rag.py` | Chunks sent to the LLM as context |

Note that the current corpus is small (~18 chunks). At `k=20` BM25
effectively returns the entire corpus, which makes fusion behave like
"dense ranking with a lexical tiebreaker". Lower the BM25 `k` as the
knowledge base grows.

## Verifying hybrid retrieval

`app/check_hybrid.py` asserts the three properties that fail silently in
a hybrid retriever:

``` bash
python app/check_hybrid.py        # uses the app's BM25 k (20)
python app/check_hybrid.py 3      # discriminating k, real signal
```

1.  **Corpus parity** — the BM25 index size matches the live Chroma
    collection size, catching a stale index after re-ingestion.
2.  **Fusion key match** — the dense and BM25 key sets intersect, and the
    top fused chunk scores above the single-list ceiling
    `1 / (60 + 1)`, proving it was credited by *both* retrievers.
3.  **Unique contribution** — BM25 recovers a rare exact token
    (`AIML201`) that dense search blurs, dense search recovers a
    paraphrase (`I need time off because I am unwell`) that BM25 is blind
    to, and both survive into the fused top-5.

The script exits non-zero if any check fails, so it can be wired into CI.
Pass the second argument when the corpus is small — at the default `k`,
BM25 returns everything and checks 2 and 3 pass regardless of whether
fusion works.

You can also inspect the two retrievers side by side:

``` bash
python app/bm25_retrival.py    # BM25 results vs. Chroma results for one query
python app/rag.py              # end-to-end answer + fused context with RRF scores
```

## Testing

Use a small set of questions with known answers from your source
documents.

  -----------------------------------------------------------------------
  Test case               Example query           What to check
  ----------------------- ----------------------- -----------------------
  TC01                    What is the attendance  Correct policy answer
                          requirement for         and relevant source
                          students?               

  TC02                    What courses are        Relevant course
                          available in the        information, without
                          Computer Science        mixing departments
                          department?             

  TC03                    What facilities are     Correct facility
                          available in the        details and source
                          college library?        

  TC04                    What should I do if my  Safe handling of
                          question is not covered missing information
                          by college policies?    

  TC05                    What is the attendance  Covers both parts of a
                          requirement, and what   multi-part question
                          happens if I fail to    
                          meet it?                
  -----------------------------------------------------------------------

Two additional cases specifically exercise hybrid retrieval:

  -----------------------------------------------------------------------
  Test case               Example query           What to check
  ----------------------- ----------------------- -----------------------
  TC06                    Tell me about AIML201   Exact course code is
                                                  retrieved - this is the
                                                  BM25 leg working

  TC07                    I need time off because Leave policy is
                          I am unwell             retrieved despite no
                                                  shared tokens - this is
                                                  the dense leg working
  -----------------------------------------------------------------------

Both are automated in `app/check_hybrid.py`.

Also test an out-of-scope question, such as:

> What is the hostel mess menu for tomorrow?

If the knowledge base does not contain this information, the assistant
should not fabricate an answer.

## Troubleshooting

### `ModuleNotFoundError`

Make sure the project virtual environment is activated and dependencies
are installed:

``` bash
python -m pip install -r requirements.txt
```

### API key or authentication error

-   Confirm the expected environment variable is present.
-   Check that `.env` is loaded by the application if you use one.
-   Ensure the key belongs to the provider configured in your code.
-   Never print or commit the key.

### Vector store or index not found

Run the project's ingestion/index-building script, or verify that the
configured persisted index path exists and matches the path used by the
retriever.

### Gradio callback argument error

Make sure the callback's function signature matches the components
listed in `inputs`. For example, if the callback is declared as
`chat(question, history)`, it needs two corresponding inputs. If your
RAG function accepts only a question, call it from the UI callback as
`ask_question(question)`.

### `ModuleNotFoundError: No module named 'rank_bm25'`

Install the sparse-retrieval dependency:

``` bash
python -m pip install rank_bm25
```

### `ModuleNotFoundError: No module named 'rag'` / `'ingest'`

The modules in `app/` import each other as siblings. Run scripts as
`python app/ui.py` from the project root (Python puts the script's own
directory on `sys.path`), not as `python -m app.ui`.

### Answers ignore obvious keyword matches

BM25 may be scoring a stale corpus. The index is built once at import
from the Chroma collection, so restart the app after re-ingesting, then
confirm parity:

``` bash
python app/check_hybrid.py 3
```

A failing "BM25 index size == live Chroma collection size" check
confirms the stale-index case. A failing "dense and BM25 key sets
intersect" check means `doc_key()` no longer collides across the two
retrievers, so RRF is concatenating instead of fusing.

### Chatbot message format error

Keep the data returned by the callback consistent with the format
expected by the installed Gradio version and the configured chatbot
component. The chatbot output must be valid conversation history, while
the sources output must target a separate Gradio component.

## Planned enhancements

Completed:

-   ~~Hybrid retrieval (keyword search + vector search) with Reciprocal
    Rank Fusion.~~ Implemented in `rag.py`, verified by
    `app/check_hybrid.py`.

Still open:

-   Cross-encoder reranking of the fused candidates before sending
    context to the LLM.
-   Stable `chunk_id` metadata at ingestion time, so RRF can key on an id
    instead of the chunk text.
-   Rebuild the BM25 index on ingestion instead of only at import, so a
    re-ingest does not require a restart.
-   Evaluation using a fixed question set and retrieval/answer quality
    metrics.
-   Improved source presentation, such as document names and relevant
    excerpts.

Only mark an enhancement as completed after it is implemented and
tested.

## Security

-   Keep API keys in environment variables or a secret manager.
-   Do not commit `.env`, virtual environments, private college
    documents, or generated indexes unless you are authorized to publish
    them.
-   Confirm you have permission to use and share the source material.

## License

Add the license applicable to your project here. Do not claim a license
unless you have selected and included one.
