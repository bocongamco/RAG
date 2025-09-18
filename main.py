from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model="llama3.2")

template = """
You are a laptop shopping assistant.
Use ONLY the given CONTEXT. Do not invent facts.

Return the answer in EXACTLY this format:
Answer: <one short sentence with the exact MODEL and PRICE from context>
Citations: row=<ROW>, name="<MODEL>"

If the answer is not in the context, reply exactly:
Answer: I don't know
Citations: (none)

CONTEXT:
{ctx}

QUESTION: {q}
"""
prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

def fmt_ctx(docs):
    # Show first 350 chars to keep context focused
    return "\n".join(d.page_content[:350] for d in docs)

while True:
    q = input("\nAsk your question (q to quit): ").strip()
    if q.lower() == "q": break

    docs = retriever.invoke(q)
    if not docs:
        print("Answer: I don't know\nCitations: (none)")
        continue

    ctx = fmt_ctx(docs)
    resp = chain.invoke({"ctx": ctx, "q": q})
    print(resp)
