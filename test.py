import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

#Document loading
from langchain_community.document_loaders import PyPDFLoader

file_path = "Paper.pdf"
loader = PyPDFLoader(file_path)
docs = loader.load()

print(docs)