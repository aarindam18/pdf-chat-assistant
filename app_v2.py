import streamlit as st
import os
# Replace these imports at the top:
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.llms import LlamaCpp
from langchain_community.document_loaders import PyPDFLoader


# Initialize session state variables
if 'vector_store' not in st.session_state:
    st.session_state.vector_store = None
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'current_pdf' not in st.session_state:
    st.session_state.current_pdf = None

# Page configuration
st.set_page_config(
    page_title="PDF Chat Assistant",
    page_icon="🤓☝️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling with improved visibility
st.markdown("""
    <style>
    /* Main styling */
    .main {
        background-color: #f8f9fa;
    }
    .sidebar .sidebar-content {
        background-color: #e9ecef;
    }
    .block-container {
        padding-top: 2rem;
    }
    
    /* Improved text input visibility */
    .stTextInput>div>div>input {
        background-color: #ffffff;
        color: #212529 !important;
        border: 1px solid #ced4da;
        border-radius: 4px;
        padding: 8px 12px;
    }
    
    /* Button styling */
    .stButton>button {
        background-color: #0d6efd;
        color: white;
        border-radius: 4px;
        padding: 0.5rem 1rem;
        border: none;
        font-weight: 500;
    }
    .stButton>button:hover {
        background-color: #0b5ed7;
    }

    /* History items - better contrast */
    .history-item {
        border-left: 3px solid #0d6efd;
        padding: 0.75rem;
        margin-bottom: 0.75rem;
        background-color: #f1f3f5;
        border-radius: 0 6px 6px 0;
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        color: #212529;
    }

    /* Chat containers */
    .chat-container {
        background-color: #ffffff;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 2px 10px rgba(0,0,0,0.05);
    }
    
    /* Question box - more subtle styling */
    .question-box {
        background-color: #f8f9fa;
        border-left: 4px solid #0d6efd;
        padding: 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.25rem;
        color: #212529;
    }
    
    /* Answer box - more subtle styling */
    .answer-box {
        background-color: #f8f9fa;
        border-left: 4px solid #198754;
        padding: 1.25rem;
        border-radius: 6px;
        margin-bottom: 1.25rem;
        color: #212529;
    }
    
    /* Better header styling */
    .header {
        color: #212529;
        font-weight: 600;
    }
    
    /* Improved spacing */
    .st-emotion-cache-1y4p8pa {
        padding: 1.5rem 1.5rem 3rem;
    }
    </style>
    """, unsafe_allow_html=True)

# Sidebar - PDF Upload and History
with st.sidebar:
    st.header("📁 Document Upload")
    uploaded_file = st.file_uploader("Upload a PDF", type=["pdf"], label_visibility="collapsed")
    
    new_pdf_uploaded = False
    if uploaded_file is not None:
        # Check if this is a new PDF
        if uploaded_file.name != st.session_state.current_pdf:
            new_pdf_uploaded = True
            st.session_state.current_pdf = uploaded_file.name
            
        # Process PDF
        temp_pdf_path = f"temp_{uploaded_file.name}"
        with open(temp_pdf_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        loader = PyPDFLoader(temp_pdf_path)
        pages = loader.load()
        
        chunks = [page.page_content for page in pages]
        metadata_list = [page.metadata for page in pages]
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)
        split_docs = splitter.create_documents(chunks)
        
        chunks = [doc.page_content for doc in split_docs]
        metadata_list = [doc.metadata for doc in split_docs]
        
        embedding = HuggingFaceEmbeddings(model_name="BAAI/bge-small-en")
        vector_store = Chroma.from_documents(
            [Document(page_content=ch, metadata=md) for ch, md in zip(chunks, metadata_list)],
            embedding
        )
        
        st.session_state.vector_store = vector_store
        os.remove(temp_pdf_path)
        
        if new_pdf_uploaded:
            # Reset conversation when new document is uploaded
            st.session_state.conversation = []
            st.success("New document processed successfully!")
        else:
            st.success("Document refreshed successfully!")

    st.divider()
    st.header("💬 Conversation History")
    
    # Display conversation history
    if st.session_state.conversation:
        for i, (query, answer) in enumerate(st.session_state.conversation):
            with st.container():
                st.markdown(f"<div class='history-item'>"
                            f"<strong>Q{i+1}:</strong> {query}<br><br>"
                            f"<strong>A{i+1}:</strong> {answer}"
                            f"</div>", unsafe_allow_html=True)
        
        if st.button("Clear History", use_container_width=True):
            st.session_state.conversation = []
            st.rerun()
    else:
        st.info("No conversation history yet")

# Main Content Area
st.title("📚 PDF Chat Assistant")
st.markdown("Ask questions about your uploaded PDF document")
if not os.path.exists("model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
    os.makedirs("model", exist_ok=True)
    with st.spinner("Downloading AI model (4.5GB)..."):
        import urllib.request
        urllib.request.urlretrieve(
            "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
            "model/mistral-7b-instruct-v0.2.Q4_K_M.gguf"
        )
# Initialize LLM
if 'llm' not in st.session_state:
    st.session_state.llm = LlamaCpp(
        model_path="model/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        n_gpu_layers=40,
        n_ctx=2048,
        max_tokens=512,
        verbose=False,
    )

# Define the prompt
prompt_template = """
You are an AI assistant. Answer the question using ONLY the relevant part of the context below.

Instructions:
- Only answer the current user question.
- Be clear, accurate, and directly based on the context.
- If the answer isn't in the context, reply: "Sorry, I don't have information about that."

Context:
{context}

User Question:
{query}

Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=prompt_template
)

# Create the chain
if 'llm_chain' not in st.session_state:
    st.session_state.llm_chain = LLMChain(
        llm=st.session_state.llm,
        prompt=prompt,
        verbose=False
    )

# Main chat area
if st.session_state.vector_store is None:
    st.info("Please upload a PDF document to get started")
else:
    # Show current document name
    st.markdown(f"**Current Document:** `{st.session_state.current_pdf}`")
    
    with st.container():
        # Add placeholder with dark text for better visibility
        query = st.text_input(
            "Enter your question:", 
            key="query_input", 
            label_visibility="collapsed",
            placeholder="Type your question about the document here..."
        )
        
        if query:
            with st.spinner("Analyzing document and generating response..."):
                # Retrieve relevant context
                docs = st.session_state.vector_store.similarity_search(query, k=5)
                context = "\n".join([doc.page_content for doc in docs])
                
                # Generate response
                answer = st.session_state.llm_chain.run({
                    "context": context,
                    "query": query
                })
                
                # Store conversation
                st.session_state.conversation.append((query, answer))
                
                # Display response with improved styling
                st.markdown("<div class='question-box'>"
                            f"<strong>QUESTION:</strong><br>{query}"
                            "</div>", unsafe_allow_html=True)
                st.markdown("<div class='answer-box'>"
                            f"<strong>ANSWER:</strong><br>{answer}"
                            "</div>", unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Aarindam 😎")