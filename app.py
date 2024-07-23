import streamlit as st 
import os 
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings, ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

from dotenv import load_dotenv
import tempfile
import time

load_dotenv()

# load the Nvidia API key 
os.environ['NVIDIA_API_KEY'] = os.getenv('NVIDIA_API_KEY')

llm = ChatNVIDIA(model="meta/llama3-70b-instruct")

def vector_embedding(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name

    st.session_state.embeddings = NVIDIAEmbeddings()
    st.session_state.loader = PyPDFLoader(tmp_file_path)
    st.session_state.docs = st.session_state.loader.load()
    st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
    st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)

    os.unlink(tmp_file_path)

st.title("NVIDIA NIM Demo")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate response based on the question
    <context>
    {context}
    </context>
    Question: {input}
    """
)

uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

if uploaded_file is not None:
    if st.button("Process PDF"):
        with st.spinner("Processing PDF..."):
            vector_embedding(uploaded_file)
        st.success("FAISS Vector Store DB is ready using NvidiaEmbedding")

prompt1 = st.text_input("Enter your question about the uploaded document")

if prompt1 and 'vectors' in st.session_state:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    
    with st.spinner("Generating answer..."):
        start = time.process_time()
        response = retrieval_chain.invoke({'input': prompt1})
        end = time.process_time()
    
    st.write("Answer:", response['answer'])
    st.write(f"Response time: {end - start:.2f} seconds")
    
    with st.expander("Document Similarity Search"):
        for i, doc in enumerate(response["context"]):
            st.write(f"Chunk {i + 1}:")
            st.write(doc.page_content)
            st.write("------------------------------------------")
else:
    if prompt1:
        st.warning("Please upload and process a PDF document first.")