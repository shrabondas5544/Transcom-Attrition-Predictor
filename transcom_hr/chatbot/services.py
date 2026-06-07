import os
from django.conf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate

# Thread-safe in-memory vector store cache
_vector_store = None

def get_vector_store():
    global _vector_store
    if _vector_store is not None:
        return _vector_store
        
    policies_path = os.path.join(settings.BASE_DIR, 'chatbot', 'data', 'transcom_retention_policies.txt')
    if not os.path.exists(policies_path):
        raise FileNotFoundError(f"Retention policies document not found at {policies_path}")
        
    with open(policies_path, 'r', encoding='utf-8') as f:
        text = f.read()
        
    # Split text into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    docs = text_splitter.split_text(text)
    
    # Initialize embeddings (using exact class name supported by installed package)
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001",
        google_api_key=settings.GEMINI_API_KEY
    )
    
    # Load into in-memory FAISS database
    _vector_store = FAISS.from_texts(docs, embeddings)
    return _vector_store

def generate_retention_response(user_query):
    """
    RAG pipeline: similarity search on FAISS + ChatGoogleGenerativeAI response generation.
    """
    # Defensive check for missing API Key
    if not settings.GEMINI_API_KEY or settings.GEMINI_API_KEY == 'your_gemini_api_key_here' or settings.GEMINI_API_KEY == '':
        return (
            "System Notice: The GEMINI_API_KEY is currently not configured or using the default placeholder. "
            "Please add a valid Google Gemini API key to your `.env` file in the root directory and restart the server to activate the AI Chatbot."
        )
        
    try:
        # Get or build vector store
        db = get_vector_store()
        
        # Retrieve top 3 blocks
        docs = db.similarity_search(user_query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Initialize LLM
        llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3
        )
        
        # Prompt construction
        template = "You are an expert HR Advisor for Transcom Electronics Limited. Use the following policy context to give a precise, concise, and actionable retention strategy answer to the manager's problem. Context: {context} \n Question: {question}"
        prompt_template = PromptTemplate(template=template, input_variables=["context", "question"])
        prompt = prompt_template.format(context=context, question=user_query)
        
        # Invoke LLM
        response = llm.invoke(prompt)
        return response.content
    except Exception as e:
        return f"Advisor Error: Failed to generate retention response. Detail: {str(e)}"
