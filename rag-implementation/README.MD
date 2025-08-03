# 🧠 Retrieval-Augmented Generation (RAG) with LangChain + TinyLlama

This project demonstrates a complete RAG pipeline using **LangChain** and **FAISS**, powered by local open-source models like **TinyLlama** via **Ollama**. The system loads a PDF, chunks and embeds its content, stores it in a vector database, retrieves relevant chunks based on a user query, and uses a local LLM to generate answers.

---

## 📁 Project Structure

- `rag_implementation.ipynb` — Main Jupyter notebook implementing the RAG pipeline.
- `Data/Challenges in ESG Reporting.pdf` — Input document for question answering.
- *(Optional)* `requirements.txt` — To store all required packages for setup.

---

## ⚙️ Technologies Used

| Component                | Description                                                                 |
|--------------------------|-----------------------------------------------------------------------------|
| **LangChain**            | For PDF loading, text splitting, retrieval, and LLM orchestration           |
| **FAISS**                | Vector similarity search for text embeddings                                |
| **Ollama**               | Lightweight local LLM backend (used to run TinyLlama)                       |
| **sentence-transformers**| Hugging Face embeddings for converting text to vectors                      |
| **Jupyter Notebook**     | Development environment for the project                                     |

---

## 🔧 Setup Instructions

### 1. Install Ollama

➡️ [https://ollama.com](https://ollama.com)

### 2. Run the Model via Terminal

```
ollama run tinyllama
```

### 3. Install Dependencies

Run the following in the **first cell** of your Jupyter Notebook:

```
!pip install langchain langchain-community pypdf faiss-cpu sentence-transformers langchain-ollama
```
### 4. Launch the Notebook
Open Jupyter Notebook using Anaconda Navigator or from the terminal:

```
jupyter notebook
```
Then navigate to and open `rag_implementation.ipynb.` 
Execute each cell sequentially.

### 🧪 Pipeline Workflow
- Load PDF 
Using PyPDFLoader from langchain_community.document_loaders.

- Split Text into Chunks
Using RecursiveCharacterTextSplitter to create overlapping text segments for embedding.

- Generate Embeddings
Using sentence-transformers/all-MiniLM-L6-v2 via HuggingFace to encode each chunk.

- Store in FAISS Vectorstore
The embeddings are indexed using FAISS for fast similarity search.

- Retrieve Chunks Based on Query
A retriever fetches the most relevant chunks for a user-supplied question.

- Generate Answer Using LLM
The retrieved chunks are passed into ChatOllama, which calls the tinyllama model hosted locally.
