import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
# HuggingFaceEmbeddings is no longer needed at runtime
# from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from fastapi import FastAPI
from mangum import Mangum

load_dotenv()
app = FastAPI()


#Document loading
def load_document(file_path):
    loader = PyPDFLoader(file_path)
    docs = loader.load()
    return docs

#Text splitting
def split_document(docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(docs)
    return texts

# This function is for building the index offline, not for runtime.
# #Creating embeddings
# def create_embeddings(texts):
#     embeddings = HuggingFaceEmbeddings(
#         model_name="sentence-transformers/all-MiniLM-L6-v2"
#     )
#     #Creating FAISS vector store
#     db = FAISS.from_documents(texts, embeddings)
#     db.save_local("faiss_index")
#     return db

#Loading FAISS vector store
def load_faiss_index():
    # The embedding model is only needed for CREATING the index.
    # For loading and querying, we can pass None, but we need to provide a dummy embedder
    # that matches the expected interface for LangChain's FAISS.load_local.
    # A simple mock or a non-operational embedder would work here if needed,
    # but FAISS itself doesn't need it for lookups.
    # However, LangChain's wrapper *does* require an embedding object.
    # Let's try loading without it first, and if that fails, we'll use a placeholder.
    # The allow_dangerous_deserialization flag is important here.
    
    # We need an embedding object to satisfy the LangChain FAISS loader's signature,
    # but the actual embedding model isn't used for retrieval.
    # We will use a placeholder or a mock embedder from LangChain if available.
    # Let's check for a "Fake" or "Mock" embedder. A quick search suggests
    # langchain_community.embeddings.fake.FakeEmbeddings
    from langchain_community.embeddings.fake import FakeEmbeddings

    embeddings = FakeEmbeddings(size=384) # The size must match the original model (all-MiniLM-L6-v2 has 384 dimensions)

    db = FAISS.load_local(
        "faiss_index", 
        embeddings, 
        allow_dangerous_deserialization=True
    )
    return db

#Retrieval chain
def retrieval_chain(db, question):
    prompt_template = """You are an expert research assistant analyzing academic papers. Answer the question based on the provided context.

Instructions:
- Provide a clear, direct answer in 3-4 sentences maximum
- Start with the main point, then add supporting details
- Reference specific sections or evidence when possible
- If information is insufficient, explain what's missing and provide a partial answer
- Use professional, academic language
- If multiple perspectives exist, acknowledge them briefly

Context: {context}
Question: {input}

Answer:
"""
    retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(prompt_template)
    retriever = db.as_retriever()
    llm = ChatOpenAI(
        model="google/gemini-flash-1.5", 
        base_url="https://openrouter.ai/api/v1",
        api_key=os.getenv("OPENROUTER_API_KEY")
    )

    combine_docs_chain = create_stuff_documents_chain(
        llm, retrieval_qa_chat_prompt
    )
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    response = retrieval_chain.invoke({"input": question})
    return response["answer"]

def create_faiss_db_from_document(Paper_path):
    # This function should not be part of the Lambda runtime.
    # It's an offline build step.
    # docs = load_document(Paper_path)
    # split_docs = split_document(docs)
    # db = create_embeddings(split_docs)
    # return db
    pass

#Create a function to load the document and create the FAISS index
@app.post("/create_faiss_db_from_document")
def create_faiss_db_from_document(Paper_path):
    docs = load_document(Paper_path)
    split_docs = split_document(docs)
    db = create_embeddings(split_docs)
    return db

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.get("/ask/{question}")
def ask(question: str):
    db = load_faiss_index()
    return {"answer": retrieval_chain(db, question=question)}



handler = Mangum(app, api_gateway_base_path="/default")

