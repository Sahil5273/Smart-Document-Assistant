import os
import gradio as gr
import google.generativeai as genai
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# --- CONFIGURATION ---
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY") # Replace with your real key
genai.configure(api_key=GEMINI_API_KEY)
model = genai.GenerativeModel('gemini-1.5-flash')
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Global variable to store our database
vectorstore = None

# --- CORE FUNCTIONS ---

def ingest_pdf(file):
    global vectorstore
    if file is None:
        return "Please upload a PDF file first."
    
    # 1. Load and Split
    loader = PyPDFLoader(file.name)
    documents = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    
    # 2. Build Vector Store
    # We create it in-memory for the web app to keep it fast
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    
    return f"✅ Successfully processed {len(chunks)} text chunks. You can now ask questions!"

def chat_with_pdf(message, history):
    global vectorstore
    if vectorstore is None:
        return "Please upload a PDF and wait for processing before asking questions."
    
    # 1. Retrieval
    docs = vectorstore.similarity_search(message, k=3)
    context = "\n\n".join([d.page_content for d in docs])
    
    # 2. Augmented Generation
    prompt = f"""Answer the question based ONLY on the following context. 
    If the answer isn't there, say you don't know.
    
    Context: {context}
    Question: {message}"""
    
    response = model.generate_content(prompt)
    return response.text

# --- GRADIO INTERFACE ---

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 📄 Smart Document Assistant")
    gr.Markdown("### Upload a PDF and chat with its contents using RAG + Gemini")
    
    with gr.Row():
        with gr.Column(scale=1):
            file_input = gr.File(label="Step 1: Upload PDF", file_types=[".pdf"])
            process_btn = gr.Button("Step 2: Process Document", variant="primary")
            status_output = gr.Textbox(label="Status", interactive=False)
            
        with gr.Column(scale=2):
            chatbot = gr.ChatInterface(
                fn=chat_with_pdf,
                type="messages",
                examples=["What is the summary of this document?", "List the key findings."],
            )

    # Link the button to the ingestion function
    process_btn.click(ingest_pdf, inputs=[file_input], outputs=[status_output])

if __name__ == "__main__":
    demo.launch()