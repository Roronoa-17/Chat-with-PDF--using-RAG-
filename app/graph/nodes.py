from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers import StrOutputParser
from app.graph.state import GraphState
from app.services.vectorstore import get_vectorstore

llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")
output_parser = StrOutputParser()

def retrieve_node(state: GraphState):
    vectorstore = get_vectorstore()
    docs = vectorstore.similarity_search(state["question"], k=3)
    
    context = "\n\n".join([
        f"[Page {doc.metadata.get('page', 0) + 1}]: {doc.page_content}"
        for doc in docs
    ])
    return {"context": context, "sources": docs}

def generate_node(state: GraphState):
    prompt = (
        f"Answer the question using only the context.\n"
        f"Context: {state['context']}\n"
        f"Question: {state['question']}"
    )
    chain = llm | output_parser
    answer = chain.invoke(prompt)
    return {"answer": answer}