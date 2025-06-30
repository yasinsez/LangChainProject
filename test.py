import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

#Document loading
from langchain_community.document_loaders import PyPDFLoader

file_path = "Paper.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

#Text splitting
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
texts = text_splitter.split_documents(docs)

#For printing the first 3 chunks
#print(texts[:3])

#Creating embeddings
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

#Creating FAISS vector store
db = FAISS.from_documents(texts, embeddings)
db.save_local("faiss_index")

#Similarity search
query = "What is the main topic of the paper?"
result_docs = db.similarity_search(query)

for i, doc in enumerate(result_docs):
    print(f"Result {i+1}:")
    print(doc.page_content)
    print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    print("-" * 20)