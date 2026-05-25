import os
from google import genai
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# 1. Setup Gemini (Paste your actual API key here for testing)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# We use Gemini 1.5 Flash because it is incredibly fast and free
model = genai.GenerativeModel('gemini-1.5-flash') 

print("1. Connecting to the database...")
# We must use the exact same embedding model we used to save the text
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Point Chroma to the folder we created in the last step
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=embeddings)

def ask_question(question):
    print(f"\nQuestion: {question}")
    
    print("2. Searching the database for answers...")
    # k=3 means we grab the top 3 most relevant chunks of text
    search_results = vectorstore.similarity_search(question, k=3) 
    
    # Glue those 3 chunks together into one big paragraph of context
    context = "\n\n".join([doc.page_content for doc in search_results])
    
    print("3. Asking Gemini to read the chunks and answer...")
    # This prompt is the magic of RAG. It forces the AI to stick to the facts.
    prompt = f"""
    You are a helpful AI assistant. Answer the user's question using ONLY the context provided below. 
    If the context does not contain the answer, simply state: "I don't know based on the provided document." Do not guess.
    
    Context:
    {context}
    
    Question:
    {question}
    """
    
    # Send the prompt to Gemini
    response = model.generate_content(prompt)
    
    print("\n--- Final Answer ---")
    print(response.text)
    print("--------------------")

# Let's test it!
if __name__ == "__main__":
    # Change this question to match something inside your sample.pdf
    user_question = "What is the main topic of this document?"
    ask_question(user_question)