# LangChain RAG System: Querying Scientific Literature

> This project demonstrates a Retrieval-Augmented Generation (RAG) system built with Python and the LangChain framework. The system uses a specific scientific paper as its knowledge base, allowing users to ask questions and receive context-aware answers.

---

## 📜 Reference Paper

The knowledge base for this RAG system is the paper "A Cassini 4th pamphlet against Delambre on lunar libration and other issues" by Pasquale Tucci.

- **Title:** _A Cassini 4th pamphlet against Delambre on lunar libration and other issues_
- **Author:** Pasquale Tucci
- **Source:** [arXiv:2506.18449](https://arxiv.org/pdf/2506.18449)

---

## ✨ Features

- **Document Loading:** Ingests and processes PDF documents.
- **Vector Storage:** Creates and stores vector embeddings for efficient retrieval.
- **Question Answering:** Accepts user queries and retrieves relevant document chunks.
- **Content Generation:** Generates human-like answers based on the retrieved context.

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Pip

### Installation

1.  **Activate your virtual environment:**

    ```bash
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

2.  **Install the required packages:**
    We have already installed `langchain_community` and `pypdf`. For a full RAG system, you will likely need additional packages. A typical setup might include:
    ```bash
    pip install -qU langchain langchain-openai faiss-cpu tiktoken
    ```
    _(Note: You will need to configure an API key for your chosen LLM provider, e.g., OpenAI)._

---

## Usage

While the main application script is not yet created, the typical usage would be to run a Python script from your terminal:

```bash
python your_rag_script.py
```

You would then be prompted to enter your questions about the loaded document.
