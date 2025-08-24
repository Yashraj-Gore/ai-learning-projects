# RAG Implementation with RAGAS

Production-oriented Retrieval-Augmented Generation (RAG) pipeline using LangChain, FAISS, and RAGAS for evaluation. The notebook builds a simple, testable path from raw PDF to an answerable retriever with quantitative evals.

- Repo: <https://github.com/Yashraj-Gore/ai-learning-projects/tree/main/rag-implementation-with-ragas>
- Entry point: [`./rag_implementation_with_ragas.ipynb`](./rag_implementation_with_ragas.ipynb)

## What this does

- Loads a local PDF knowledge source (BABOK v3).
- Chunks and embeds the content with OpenAI embeddings.
- Indexes chunks in a FAISS vector store and exposes a retriever.
- Defines a strict answer-only-from-context prompt.
- Answers questions via `ChatOpenAI`.
- Evaluates the pipeline with **RAGAS** metrics (faithfulness, relevance, etc.).
- Produces a DataFrame of metric scores for quick inspection/export.

## Techniques worth noting

- **LangChain Expression Language (LCEL) composition** using the `|` operator to chain components (prompt → model → parser) for concise, testable graphs.  
  See LangChain’s expression language guide: <https://python.langchain.com/docs/expression_language/>

- **Prompt templating** with `PromptTemplate` to enforce “answer only from context” behavior and predictable failure modes when context is insufficient.  
  <https://python.langchain.com/docs/modules/model_io/prompts/>

- **Structured output parsing** via `StrOutputParser` to normalize model responses for downstream scoring.

- **PDF ingestion** with `PyPDFLoader` and **recursive chunking** via `RecursiveCharacterTextSplitter` for stable retrieval recall.  
  Chunking concept: <https://python.langchain.com/docs/modules/data_connection/document_transformers/recursive_text_splitter/>

- **FAISS vector search** (`FAISS.from_documents`) for fast approximate nearest neighbor retrieval.  
  FAISS project: <https://github.com/facebookresearch/faiss>

- **OpenAI embeddings + chat** with `OpenAIEmbeddings` and `ChatOpenAI` to power retrieval and generation.  
  <https://platform.openai.com/docs/guides/embeddings>  
  <https://python.langchain.com/docs/integrations/llms/openai/>

- **Context window control**: a small utility that caps concatenated context length (`max_docs`, `max_chars`) before passing to the model—useful for predictable latency and cost.

- **RAG evaluation with RAGAS**:  
  Metrics used include `answer_correctness`, `faithfulness`, `answer_relevancy`, `context_recall`, and `answer_similarity`.  
  <https://github.com/explodinggradients/ragas>

- **Lightweight judging LLM wrapper** using `LangchainLLMWrapper` so RAGAS can call a LangChain-managed model for metric computation.

- **Reproducible mini-datasets** with `datasets.Dataset.from_dict` for question/ground-truth pairs.  
  <https://huggingface.co/docs/datasets/python/how_to>

## Libraries and tools (with links)

- LangChain: <https://python.langchain.com/>
- langchain-openai: <https://python.langchain.com/docs/integrations/llms/openai/>
- RAGAS: <https://github.com/explodinggradients/ragas>
- FAISS (CPU): <https://github.com/facebookresearch/faiss>
- PyPDF: <https://pypdf.readthedocs.io/>
- Hugging Face Datasets: <https://huggingface.co/docs/datasets>
- python-dotenv: <https://pypi.org/project/python-dotenv/>
- OpenAI API: <https://platform.openai.com/docs/overview>

> Fonts: none used by this project.

## Project structure

```text
.
├─ README.md
├─ rag_implementation_with_ragas.ipynb
├─ .env.example
├─ data/
├─ eval/
└─ assets/
```

**Directory notes**

- `data/` — place source documents (e.g., your PDF corpus).  
- `eval/` — optional output location for RAGAS evaluation artifacts (CSV/Parquet you may export from the notebook).  
- `assets/` — any images/diagrams you might add for docs later.

> The notebook currently references a PDF file name directly (`BABOK-Guide-v3-Member.pdf`). Keep it in repo root (as used in the notebook) or move it into `data/` and update the path in [`./rag_implementation_with_ragas.ipynb`](./rag_implementation_with_ragas.ipynb).

## Setup

1. **Python**: 3.10+ recommended.
2. **Environment variables**: copy `.env.example` to `.env` and set:
   - `OPENAI_API_KEY`
3. **Install deps** (from the notebook cells or your shell):

   ```bash
   pip install python-dotenv
   pip install langchain langchain-community pypdf faiss-cpu langchain-openai datasets ragas
   ```

4. **Place documents**: put your PDF(s) in repo root (or `./data`) and ensure the path in the notebook matches.
5. **Run the notebook**: open [`./rag_implementation_with_ragas.ipynb`](./rag_implementation_with_ragas.ipynb) and execute cells top-to-bottom.

## Pipeline workflow

1. **Load**  
   - Read PDF with `PyPDFLoader`.
2. **Chunk**  
   - `RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=100)`.
3. **Embed & index**  
   - `OpenAIEmbeddings(model="text-embedding-3-small")` → `FAISS.from_documents(...)`.
4. **Retrieve**  
   - `retriever = vectorstore.as_retriever(...)`.
5. **Constrain context**  
   - Build bounded context (`max_docs`, `max_chars`) before generation.
6. **Generate**  
   - `PromptTemplate` → `ChatOpenAI` → `StrOutputParser` (LCEL `|` composition).
7. **Evaluate (RAGAS)**  
   - Build `datasets.Dataset` of Q&A pairs.  
   - Run `ragas.evaluate` with metrics: `answer_correctness`, `faithfulness`, `answer_relevancy`, `context_recall`, `answer_similarity`.  
   - Collect a `pandas` DataFrame for scores (optionally export to `./eval`).

## Configuration

- **Model**: default `gpt-5-mini` (update in the notebook if needed).  
- **Embeddings**: `text-embedding-3-small` by default; swap as needed.  
- **Context limits**: tune `CTX_MAX_DOCS` and `CTX_MAX_CHARS` to balance latency, quality, and cost.

## Notes for productionization

- Persist FAISS index to disk (or migrate to a hosted vector DB) for multi-session usage.
- Externalize data paths and knobs via a config file or environment variables.
- Add tracing (e.g., LangSmith/OpenTelemetry) and basic guardrails for empty contexts.
