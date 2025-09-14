from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate


model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in answering questions about Laptops.

Here are some relevant reviews: {reviews}
Based on the reviews, answer the following question: {question} 

"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

result = chain.invoke({"reviews": "1. Great performance and battery life. 2. Sleek design but gets warm quickly. 3. Excellent display quality.", "question": "What are the pros and cons of this laptop?",
              "questions": "What are the pros and cons of this laptop?"})

print(result)
