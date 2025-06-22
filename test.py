import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv

load_dotenv()

llm = ChatOpenAI(
    model="google/gemini-flash-1.5", 
    base_url="https://openrouter.ai/api/v1",
    api_key=os.getenv("OPENROUTER_API_KEY")
)

messages = [
    HumanMessage(content="Explain the importance of using environment variables for API keys in a short paragraph.")
]

response = llm.invoke(messages)

print(response.content)