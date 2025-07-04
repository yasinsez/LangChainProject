# LangChain RAG System: Academic Paper Analysis

> This project implements a Retrieval-Augmented Generation (RAG) system built with Python and the LangChain framework. The system processes academic papers and provides intelligent responses to queries using a combination of vector similarity search and large language models.

## 🛠 Technical Stack

- **LangChain**: Core framework for building the RAG pipeline
- **HuggingFace Embeddings**: Using `sentence-transformers/all-MiniLM-L6-v2` for text embeddings
- **FAISS**: Vector store for efficient similarity search
- **OpenRouter**: API access to advanced language models (currently using Google's Gemini)
- **PyPDF**: PDF document processing

## ✨ Features

- **Smart Document Loading**: Uses `PyPDFLoader` to process PDF academic papers
- **Intelligent Text Splitting**: Implements `RecursiveCharacterTextSplitter` with 1000-character chunks and 200-character overlap
- **Vector Embeddings**: Creates and stores document embeddings using HuggingFace's sentence transformers
- **Persistent Storage**: Saves and loads FAISS indexes for quick retrieval
- **Context-Aware Responses**: Uses a custom prompt template designed for academic paper analysis
- **Professional Output**: Generates concise, academic-style responses with proper citations

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- Virtual environment (recommended)

### Installation

1. **Clone the repository and create a virtual environment:**

   ```bash
   git clone <repository-url>
   cd LangChainProject
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

2. **Install required packages:**

   ```bash
   pip install langchain langchain-openai langchain-community langchain-huggingface
   pip install python-dotenv faiss-cpu sentence-transformers pypdf
   ```

3. **Set up environment variables:**
   Create a `.env` file in the project root with:
   ```
   OPENROUTER_API_KEY=your_api_key_here
   ```

## 💡 Usage

The project provides several key functions:

1. **Loading Documents**

   ```python
   docs = load_document("your_paper.pdf")
   ```

2. **Creating Vector Database**

   ```python
   db = create_faiss_db_from_document("your_paper.pdf")
   ```

3. **Querying the System**
   ```python
   db = load_faiss_index()
   retrieval_chain(db, "Your question about the paper?")
   ```

### Example Query

```python
python test.py
```

This will load the saved FAISS index and ask a sample question about the paper.

## 📝 Implementation Details

The system uses a specialized prompt template for academic analysis that:

- Provides clear, direct answers (3-4 sentences)
- Starts with main points followed by supporting details
- References specific sections when possible
- Acknowledges information gaps
- Uses professional academic language
- Considers multiple perspectives when relevant

## 🗂 Project Structure

```
LangChainProject/
├── test.py           # Main implementation file
├── .env              # Environment variables
├── faiss_index/      # Stored vector embeddings
├── Paper.pdf         # Source academic paper
└── README.md         # Documentation
```

## 🔒 Security Note

- The project uses environment variables for API keys
- FAISS index deserialization is handled safely
- Virtual environment is properly excluded from version control
