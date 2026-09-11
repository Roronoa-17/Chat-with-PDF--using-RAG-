from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from typing import TypedDict
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import StateGraph, END
from langchain_core.output_parsers import StrOutputParser

# Initializing loader
loader = PyPDFLoader("test.pdf")
documents = loader.load()

# Verify extraction
print(f"Loaded {len(documents)} pages.")
print(f"Sample content: {documents[0].page_content[:150]}...")

# Text splitting (chunking)
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=150
)

chunks = text_splitter.split_documents(documents)
print(f"Split document into {len(chunks)} chunks")

os.environ["PINECONE_API_KEY"] = ""
os.environ["GOOGLE_API_KEY"] = ""

embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2")

index_name = "enterprise-rag-index"

print("Vectorizing and uploading chunks to Pinecone...")

# Upsert vectors directly into the Pinecone index
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings,
    index_name=index_name
)

print("Successfully vectorized and uploaded chunks to Pinecone!")

class GraphState(TypedDict):
    question:str
    context:str
    answer: str

# Initialize the Gemini model
llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

def retrieve_node(state: GraphState):
    question = state["question"]
    
    # Query your existing vectorDB for the top 3 matching chunks
    docs = vectorstore.similarity_search(question, k=3)
    
    # Combine the matched chunks into a single string
    context = "\n".join([doc.page_content for doc in docs])
    
    return {"context": context}

output_parser = StrOutputParser()

def generate_node(state: GraphState):
    prompt = (
        f"Answer the question using only the context.\n"
        f"Context: {state['context']}\n"
        f"Question: {state['question']}"
    )
    
    # Invoke Gemini and extract the raw text content
    chain = llm | output_parser
    answer = chain.invoke(prompt)
    return {"answer": answer}

# Initialize the graph with the state schema
workflow = StateGraph(GraphState)

# Add the independent nodes
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)

# Define the execution sequence
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_edge("generate", END)

# Compile into a runnable application
app = workflow.compile()

# Execute the worflow
test_query = "Summarize the document into 10 sentence"
result = app.invoke({"question": test_query})

print("\n Final Answer")
print(result["answer"])