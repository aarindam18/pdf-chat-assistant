import streamlit as st
import os
from langchain.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain_community.document_loaders import PyPDFLoader
from langchain.chat_models import ChatOpenAI

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

# Custom CSS for styling
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
        
        splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        split_docs = splitter.create_documents(chunks)
        
        chunks = [doc.page_content for doc in split_docs]
        metadata_list = [doc.metadata for doc in split_docs]
        api_key = st.secrets["OPENAI_API_KEY"]
        embedding = OpenAIEmbeddings(openai_api_key = api_key)
        vector_store = FAISS.from_documents(
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

# Initialize LLM with your exact parameters
if 'llm' not in st.session_state:
    st.session_state.llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.7,
        max_tokens=16000,
        verbose=True,
        openai_api_key=api_key
    )

# Define the prompt exactly as in your Jupyter notebook
# Define the prompt
prompt_template = """
You are an expert document analysis assistant. Your task is to provide comprehensive answers using EXCLUSIVELY the information contained in the provided context. You must NEVER use any external knowledge or information beyond what is in the context.

### Strict Rules:
1. **Context-Only Policy**:
   - Every part of your answer MUST be derived solely from the context
   - Never add information, examples, or concepts not present in the context
   - If context lacks details, do not supplement with your own knowledge

2. **Answer Construction Process**:
   a) Extract all relevant facts from context
   b) Synthesize these facts into a base answer
   c) Elaborate using ONLY:
      - Additional context details
      - Logical connections within the context
      - Examples explicitly mentioned in context

3. **Handling Unknowns**:
   - If context contains NO relevant information: 
        → "The document doesn't contain information to answer this question"
   - If context has partial information: 
        → Answer only with what exists (no extrapolation)

### Context:
{context}

### User Question:
{query}

### Response Structure:
   - Concise response using only context facts
   - Key supporting points from context
   - Relevant examples (only if present in context)
   - Relationships between context concepts
   - Use bullet points for clarity
   - Brief restatement using only context terms]

### Answer:
"""
prompt = PromptTemplate(
    input_variables=["context", "query"],
    template=prompt_template
)

# Create the chain using the pipe operator as in your Jupyter notebook
if 'chain' not in st.session_state:
    st.session_state.chain = prompt | st.session_state.llm

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
                try:
                    docs = st.session_state.vector_store.similarity_search(query, k=7)  # Get more context
                    final_chunk = "\n\n".join([f"SOURCE {i+1}:\n{doc.page_content}" for i, doc in enumerate(docs)])
                    # Generate response exactly as in your Jupyter notebook
                    answer = st.session_state.chain.invoke({
                        "context": final_chunk,
                        "query": query
                    })
                    
                    # Extract content from response
                    if isinstance(answer, dict):
                        content = answer.get("content", "not found")
                    else:
                        content = getattr(answer, "content", "not found")
                    
                    # Store conversation
                    st.session_state.conversation.append((query, content))
                    
                    # Display response with styling
                    st.markdown("<div class='question-box'>"
                                f"<strong>QUESTION:</strong><br>{query}"
                                "</div>", unsafe_allow_html=True)
                    st.markdown("<div class='answer-box'>"
                                f"<strong>ANSWER:</strong><br>{content}"
                                "</div>", unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
                    st.session_state.conversation.append((query, "Error generating response"))

# Footer
st.divider()
st.caption("Aarindam 😎")
