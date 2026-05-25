Smart Document Assistant is an intelligent document-based chatbot that allows users to upload PDF files and interact with them using natural language questions. The application uses a Retrieval-Augmented Generation (RAG) pipeline to extract relevant information from uploaded documents and generate context-aware responses using Google's Gemini model.

The workflow of the application includes:

* Uploading PDF documents
* Extracting and splitting text into chunks
* Creating vector embeddings
* Storing embeddings using Chroma vector database
* Retrieving relevant document sections
* Generating accurate responses with Gemini AI

The system helps users quickly understand and search through lengthy documents without manually reading every page.

**Features:**

* Upload PDF documents
* Automatic text extraction
* Text chunking for better retrieval
* Vector embeddings using Hugging Face
* Chroma vector database integration
* Context-aware question answering
* Interactive Gradio interface
* Gemini-powered response generation

**Tech Stack:**

* Python
* Gradio
* Google Gemini API
* LangChain
* ChromaDB
* Hugging Face Embeddings
* PyPDFLoader

**For resume/project section (short version):**

Built an AI-powered PDF chatbot using RAG architecture with Gemini, LangChain, ChromaDB, and Hugging Face embeddings for context-aware document querying.

One small thing in your code before deployment: never hardcode this:

```python
GEMINI_API_KEY = "YOUR_API_KEY_HERE"
```

Replace it with:

```python
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
```

Then set the environment variable in your deployment platform (Firebase, Hugging Face Spaces, Render, etc.). That prevents accidentally exposing your API key on GitHub.
