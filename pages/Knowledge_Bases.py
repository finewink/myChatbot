import streamlit as st
import os
from datetime import datetime
import json
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import shutil
from dotenv import load_dotenv

load_dotenv()

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Using OpenAI embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

st.set_page_config("Create Knowledge Base", "📚", layout="wide")
st.title("Create Knowledge Base")

def create_knowledge_base(name, files, description=""):
    """Create a new knowledge base with the proper structure"""
    # Validate the name (no spaces, special chars, etc.)
    import re
    name = re.sub(r'[^\w]', '_', name)
    
    # Create directories
    kb_path = os.path.join("db", name)
    original_docs_path = os.path.join(kb_path, "original_docs")
    
    if os.path.exists(kb_path):
        return False, "A knowledge base with this name already exists"
    
    os.makedirs(kb_path)
    os.makedirs(original_docs_path)
    
    # Save original files
    file_names = []
    for file in files:
        file_path = os.path.join(original_docs_path, file.name)
        with open(file_path, "wb") as f:
            f.write(file.getbuffer())
        file_names.append(file.name)
    
    # Process the documents
    loader = DirectoryLoader(original_docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
    documents = loader.load()
    
    if not documents:
        shutil.rmtree(kb_path)
        return False, "No valid documents found in the uploaded files"
    
    # Split documents
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    # Create vector store
    vector_store = FAISS.from_documents(texts, embeddings)
    vector_store.save_local(kb_path)
    
    # Save metadata
    desc_data = {
        "file_names": file_names,
        "description": description,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(kb_path, "desc.json"), "w") as f:
        json.dump(desc_data, f)
    
    # Save embedding info
    embedding_info = {
        "model": "text-embedding-3-small",  # Your current model
        "dimensions": len(embeddings.embed_query("test")),
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    with open(os.path.join(kb_path, "embedding_info.json"), "w") as f:
        json.dump(embedding_info, f)
    
    return True, f"Knowledge base '{name}' created successfully"

# Create form for knowledge base creation
with st.form("kb_creation_form"):
    kb_name = st.text_input("Knowledge Base Name", help="Enter a name for your knowledge base (no spaces or special characters)")
    kb_description = st.text_area("Description", help="Optional description of this knowledge base")
    uploaded_files = st.file_uploader("Upload PDF Documents", accept_multiple_files=True, type=["pdf"])
    
    submit_button = st.form_submit_button("Create Knowledge Base")
    
    if submit_button:
        if not kb_name:
            st.error("Please enter a name for the knowledge base")
        elif not uploaded_files:
            st.error("Please upload at least one PDF document")
        else:
            with st.spinner("Creating knowledge base..."):
                success, message = create_knowledge_base(kb_name, uploaded_files, kb_description)
                if success:
                    st.success(message)
                    st.info("You can now use this knowledge base in the chatbot")
                else:
                    st.error(message)

# Show existing knowledge bases
st.markdown("---")
st.header("Existing Knowledge Bases")

if os.path.exists("db"):
    existing_indices = [name for name in os.listdir("db") if os.path.isdir(os.path.join("db", name))]
    
    if not existing_indices:
        st.info("No knowledge bases found")
    else:
        for index in existing_indices:
            with st.expander(f"Knowledge Base: {index}"):
                try:
                    with open(f"db/{index}/desc.json", "r") as openfile:
                        description = json.load(openfile)
                        st.write(f"**Files:** {', '.join(description['file_names'])}")
                        if 'description' in description:
                            st.write(f"**Description:** {description['description']}")
                        if 'created_at' in description:
                            st.write(f"**Created:** {description['created_at']}")
                except:
                    st.write("No additional information available")
                
                # Check if it has original documents
                original_docs_path = os.path.join("db", index, "original_docs")
                if os.path.exists(original_docs_path) and len(os.listdir(original_docs_path)) > 0:
                    st.success("✅ Has original documents (can be rebuilt)")
                else:
                    st.warning("⚠️ No original documents (cannot be rebuilt)")
else:
    st.info("No knowledge bases found")

# Add a link back to the main page
st.markdown("---")
st.page_link("Chatbot.py", label="Return to Chatbot", icon="🏠")