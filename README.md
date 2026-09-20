### WELCOME TO MY RAG BASED AI CHATBOT
# College Knowledge Assistant

A Retrieval-Augmented Generation (RAG) chatbot that answers questions
about college policies, courses, facilities, and FAQs using a college
knowledge base. The project uses LangChain for retrieval and LLM
orchestration, and Gradio for the chat interface.

> **Project status:** Baseline RAG pipeline with Gradio UI. Hybrid
> retrieval and reranking are planned enhancements unless they have been
> implemented in your code.

## Features

-   Ask natural-language questions about college information.
-   Retrieve relevant documents from the configured knowledge base.
-   Generate answers using retrieved context.
-   Display source metadata for retrieved documents.
-   Chat-style UI with both an **Enter** button and keyboard submission.

## Architecture

``` text
User question
     |
     v
Gradio Blocks UI
     |
     v
ask_question(question)
     |
     +--> Retriever retrieves relevant documents
     |
     +--> Retrieved page content is combined as context
     |
     +--> Prompt receives context + question
     |
     +--> LLM generates an answer
     |
     v
Answer + retrieved documents
     |
     +--> Chatbot displays answer
     +--> Sources section displays document source metadata
```

## Tech Stack

-   Python
-   LangChain
-   Gradio
-   ChromaDB
-   OpenAIEmbeddings
-   RecursiveCharacterTextSplitter

## Project Structure

Example structure (adjust names to match your repository):

``` text
college-knowledge-assistant/
├── ui.py                  # Gradio UI and chat callback
├── rag.py                 # Retrieval and answer-generation pipeline
├── ingest.py              # Ingestion process - splitting and chunking
├── vectorstore.py         # Embedding process
├── data/                  # Knowledge-base files - local
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
cd path/to/college-knowledge-assistant
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

If your project uses a separate ingestion script, run it before
launching the chatbot. For example:

``` bash
python ingest.py
```

Use the actual ingestion filename and command from your repository. If
your vector index is already built and persisted, you may not need to
ingest again.

## Run the application

From the project root, activate the virtual environment and run the file
containing your Gradio `Blocks` interface.

For example, if it is named `ui.py`:

``` bash
python ui.py
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

1.  Retrieve documents for the question.
2.  Join each retrieved document's `page_content` into a context string.
3.  Pass `context` and `question` into the prompt.
4.  Invoke the configured chat model.
5.  Return the answer text and retrieved documents.

The UI callback should unpack both values:

``` python
answer, documents = ask_question(question)
```

Use `answer` as the chatbot response. Convert the documents' useful
metadata into a display string or other Gradio component output; do not
pass raw LangChain `Document` objects directly to a text component.

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

### Chatbot message format error

Keep the data returned by the callback consistent with the format
expected by the installed Gradio version and the configured chatbot
component. The chatbot output must be valid conversation history, while
the sources output must target a separate Gradio component.

## Planned enhancements

-   Hybrid retrieval (keyword search + vector search), potentially with
    Reciprocal Rank Fusion.
-   Reranking retrieved candidates before sending context to the LLM.
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
