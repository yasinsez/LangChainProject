import os
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

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

#Creating embeddings
def create_embeddings(texts):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    #Creating FAISS vector store
    db = FAISS.from_documents(texts, embeddings)
    db.save_local("faiss_index")
    return db

#Loading FAISS vector store
def load_faiss_index():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
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
    print(response["answer"])

def create_faiss_db_from_document(Paper_path):
    docs = load_document(Paper_path)
    split_docs = split_document(docs)
    db = create_embeddings(split_docs)
    return db

def main():
    #db = create_faiss_db_from_document("math.pdf")
    db = load_faiss_index()
    print(retrieval_chain(db, "Can you paraphrase an idea from the paper? Please just one sentence."))

if __name__ == "__main__":
    main()
