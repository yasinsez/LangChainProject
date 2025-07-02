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
#db = FAISS.from_documents(texts, embeddings)
#db.save_local("faiss_index")

#Loading FAISS vector store
db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)

#Similarity search
''''
query = "Within the Histoire, in truth, there were unjustifiable preferences, hazardous elucidations and a few mistakes"
result_docs = db.similarity_search(query)

for i, doc in enumerate(result_docs):
    print(f"Result {i+1}:")
    print(doc.page_content)
    print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    print("-" * 20)
'''

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain import hub
from langchain_core.prompts import ChatPromptTemplate


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

llm = ChatOpenAI(
    model="google/gemini-flash-1.5", 
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)
retriever = db.as_retriever()
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt
)
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

response = retrieval_chain.invoke({"input": "What is the main idea of the paper?"})
print(response["answer"])
