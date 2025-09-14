from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in answering questions about Laptops.

Here are some relevant reviews: {reviews}
Based on the reviews, answer the following question: {question} 

"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True: 
    print("\n\n=============================\n")
    question = input("Ask your question (q to quit): ")
    print("\n\n=============================\n")
    if question == "q":
        break 

    reviews = retriever.invoke(question)
    results = chain.invoke({"reviews": reviews, "question": question})
    print(results)
