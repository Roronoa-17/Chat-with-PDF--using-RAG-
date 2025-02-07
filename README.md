 # ChatWithPDF using RAG (Retrival Augmented Generation)

This web application allows users to chat with their uploaded PDF documents using LLAMA 3 through NVIDIA NIM and Retrieval Augmented Generation (RAG).

## Try it on huggingface :- https://huggingface.co/spaces/priyesh17/ChatWithPDF
## Features

- PDF upload and processing
- Question answering based on PDF content
- Utilizes LLAMA 2 model via NVIDIA NIM for natural language processing
- Implements Retrieval Augmented Generation for accurate responses
- Built with Streamlit for a user-friendly interface

## Prerequisites

- Python 3.7+
- NVIDIA API key

## Installation

1. Clone the repository:

2. Create a virtual environment and activate it

3. Install the required packages: command - `pip install -r requirements.txt`

4. Create a `.env` file in the project root and add your NVIDIA API key

## Usage

1. Run the Streamlit app:

2. Open your web browser and go to the URL provided by Streamlit (usually `http://localhost:8501`).

3. Upload a PDF file using the file uploader.

4. Click the "Process PDF" button to analyze the document.

5. Once processing is complete, enter your questions in the text input field.

6. View the AI-generated answers and relevant document chunks.

## How It Works

1. The app uses NVIDIA Embeddings to create vector representations of the PDF content.
2. The document is split into chunks and stored in a FAISS vector database.
3. When a question is asked, the app retrieves relevant document chunks using the vector database.
4. The LLAMA 2 model, accessed through NVIDIA NIM, generates an answer based on the retrieved context and the question.
5. The answer is displayed to the user along with the most relevant document chunks.

## Project Structure

- `app.py`: Main application file containing the Streamlit interface and core logic
- `requirements.txt`: List of Python dependencies
- `.env`: Environment file for storing the NVIDIA API key (not tracked in version control)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- NVIDIA for providing access to the LLAMA 2 model through NIM
- Streamlit for the web application framework
- LangChain for the document processing and RAG implementation
