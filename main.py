from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a laptop shopping assistant.
Use ONLY this context to answer:
{reviews}

If the answer is not in the context, reply exactly: "I don't know".

Question: {question}
Respond in two lines:
Answer: <one short sentence>
Citations: <model name or row index seen in the context>
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def fmt_ctx(docs):
    # Turn Documents into readable context with lightweight citations
    return "\n\n".join(
        f"[row={d.metadata.get('row')} name={d.metadata.get('name')}] {d.page_content[:500]}"
        for d in docs
    )

while True:
    print("\n\n=============================\n")
    question = input("Ask your question (q to quit): ")
    print("\n\n=============================\n")
    if question.lower() == "q":
        break

    docs = retriever.invoke(question)

    if not docs:
        print('Answer: I don\'t know\nCitations: (none)')
        continue

    reviews = fmt_ctx(docs)
    print("Retrieved top docs.\n")
    results = chain.invoke({"reviews": reviews, "question": question})
    print(results)
