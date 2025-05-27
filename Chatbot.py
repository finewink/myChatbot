from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain_community.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
import json
import PyPDF2
import streamlit as st
import os
import shutil
from datetime import datetime
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import pickle
from langchain.chains import LLMChain
from langchain.prompts import ChatPromptTemplate

os.environ["PYDANTIC_V2_FORCE"] = "1"

load_dotenv()
st.set_page_config("My RAG Chatbot", "💬", layout="wide")

# API Keys
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

# Using OpenAI embeddings
embeddings = OpenAIEmbeddings(
    openai_api_key=OPENAI_API_KEY,
    model="text-embedding-3-small"
)

# For OpenAI's GPT-4 llm
llm = ChatOpenAI(
    temperature=1,
    openai_api_key=OPENAI_API_KEY,
    model="gpt-4"
)

# Initialize session state for chat history if it doesn't exist
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "conversation_id" not in st.session_state:
    st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
if "chat_mode" not in st.session_state:
    st.session_state.chat_mode = "kb"  # Default to knowledge base mode
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# For reading PDFs and returning text string
def read_pdf(files):
    file_content=""
    for file in files:
        # Create a PDF file reader object
        pdf_reader = PyPDF2.PdfReader(file)
        # Get the total number of pages in the PDF
        num_pages = len(pdf_reader.pages)
        # Iterate through each page and extract text
        for page_num in range(num_pages):
            # Get the page object
            page = pdf_reader.pages[page_num]
            file_content += page.extract_text()
    return file_content


def migrate_knowledge_bases():
    """Migrate existing knowledge bases to the new structure"""
    path = "db"
    if not os.path.exists(path):
        os.makedirs(path)
        st.sidebar.info("Created new db directory")
        return
        
    # Get all existing indices
    existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
    
    migrated_count = 0
    for index_name in existing_indices:
        index_path = os.path.join(path, index_name)
        original_docs_path = os.path.join(index_path, "original_docs")
        
        # Check if this index already has the new structure
        if not os.path.exists(original_docs_path):
            os.makedirs(original_docs_path)
            
            # Create embedding info file if it doesn't exist
            embedding_info_path = os.path.join(index_path, "embedding_info.json")
            if not os.path.exists(embedding_info_path):
                embedding_info = {
                    "model": "text-embedding-3-small",  # Assuming this was the model used
                    "migrated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": "This is a migrated knowledge base. Original documents may be missing."
                }
                with open(embedding_info_path, "w") as f:
                    json.dump(embedding_info, f)
            
            migrated_count += 1
    
    if migrated_count > 0:
        st.sidebar.success(f"Migrated {migrated_count} knowledge bases to the new structure")


def rebuild_index(index_name):
    """Rebuild the index using current embedding model"""
    try:
        # Get the original documents path
        docs_path = os.path.join("db", index_name, "original_docs")
        if not os.path.exists(docs_path) or len(os.listdir(docs_path)) == 0:
            st.error(f"No original documents found for {index_name}")
            st.info("This knowledge base was created before the migration. You'll need to re-upload the documents.")
            return False
        
        # Load documents
        loader = DirectoryLoader(docs_path, glob="**/*.pdf", loader_cls=PyPDFLoader)
        documents = loader.load()
        
        if not documents:
            st.error("No valid documents found in the knowledge base")
            return False
        
        # Split documents
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        # Create new vector store
        kb_path = os.path.join("db", index_name)
        new_db = FAISS.from_documents(texts, embeddings)
        
        # Save embedding info
        embedding_info = {
            "model": "text-embedding-3-small",
            "dimensions": len(embeddings.embed_query("test")),
            "rebuilt_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(os.path.join(kb_path, "embedding_info.json"), "w") as f:
            json.dump(embedding_info, f)
            
        # Save the new index
        new_db.save_local(kb_path)
        
        return True
    except Exception as e:
        st.error(f"Error rebuilding index: {e}")
        return False


def can_rebuild_kb(index_name):
    """Check if a knowledge base has original documents and can be rebuilt"""
    original_docs_path = os.path.join("db", index_name, "original_docs")
    return os.path.exists(original_docs_path) and len(os.listdir(original_docs_path)) > 0


def debug_embedding_dimensions():
    """Debug function to check embedding dimensions"""
    # Check current embedding dimensions
    test_text = "This is a test sentence."
    current_embedding = embeddings.embed_query(test_text)
    st.sidebar.write(f"Current embedding dimensions: {len(current_embedding)}")
    
    # Try to check FAISS index dimensions
    if st.session_state.book_docsearch and hasattr(st.session_state.book_docsearch, 'index'):
        try:
            st.sidebar.write(f"FAISS index dimensions: {st.session_state.book_docsearch.index.d}")
        except:
            st.sidebar.write("Could not determine FAISS index dimensions")

# Save chat history to file
def save_chat_history():
    if not os.path.exists("chat_histories"):
        os.makedirs("chat_histories")
    
    history_path = os.path.join("chat_histories", f"chat_{st.session_state.conversation_id}.pkl")
    
    # Convert conversation_chatbot format to chat_history format if needed
    if st.session_state.conversation_chatbot and not st.session_state.chat_history:
        for item in st.session_state.conversation_chatbot:
            st.session_state.chat_history.append({"role": "user", "content": item[0]})
            st.session_state.chat_history.append({"role": "assistant", "content": item[1]})
    
    with open(history_path, "wb") as f:
        pickle.dump(st.session_state.chat_history, f)
    
    return history_path

# Update memory from chat history
def update_memory_from_history():
    # Clear current memory
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )
    
    # Add previous messages to memory
    for message in st.session_state.chat_history:
        if message["role"] == "user":
            st.session_state.memory.chat_memory.add_user_message(message["content"])
        elif message["role"] == "assistant":
            st.session_state.memory.chat_memory.add_ai_message(message["content"])

#-----------------------------------------------------------#
#------------------------💬 CHATBOT -----------------------#
#----------------------------------------------------------#
def chatbot():
    if st.session_state.chat_mode == "kb":
        st.subheader("Ask questions from the PDFs")
    elif st.session_state.chat_mode == "hybrid":
        st.subheader("Ask questions using both documents and general knowledge")
    else:
        st.subheader("General Conversation Mode")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Display chat messages from history
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Knowledge Base Mode
    if st.session_state.chat_mode == "kb":
        # Check if it is empty
        if st.session_state.book_docsearch:
            # Get the toggle value for strict KB mode
            strict_kb = st.session_state.get("strict_kb", True)
            
            prompt = st.chat_input("Ask a question about your documents")
            
            if prompt:
                # Add user message to chat history
                st.session_state.chat_history.append({"role": "user", "content": prompt})
                
                # Display user message
                with st.chat_message("user"):
                    st.markdown(prompt)
                
                with st.spinner("Getting Answer..."):
                    try:
                        # Create a list of chat history messages for context
                        chat_history_tuples = []
                        for i in range(0, len(st.session_state.chat_history)-1, 2):
                            if i+1 < len(st.session_state.chat_history):
                                user_msg = st.session_state.chat_history[i]["content"]
                                ai_msg = st.session_state.chat_history[i+1]["content"]
                                chat_history_tuples.append((user_msg, ai_msg))
                        
                        # No of chunks the search should retrieve from the db
                        chunks_to_retrieve = 5
                        retriever = st.session_state.book_docsearch.as_retriever(
                            search_type="similarity", 
                            search_kwargs={"k": chunks_to_retrieve}
                        )

                        if strict_kb:
                            # Use standard ConversationalRetrievalChain for strict KB mode
                            qa = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=retriever,
                                verbose=True,
                                return_source_documents=True
                            )
                        else:
                            # Create a custom prompt that allows general knowledge
                            custom_template = """Answer the question based on the context below. If the question cannot be answered using the information provided, feel free to use your general knowledge but clearly indicate when you are doing so.

                            Context: {context}
                            
                            Question: {question}
                            
                            Answer:"""
                            
                            qa_prompt = ChatPromptTemplate.from_template(custom_template)
                            
                            qa = ConversationalRetrievalChain.from_llm(
                                llm=llm,
                                retriever=retriever,
                                combine_docs_chain_kwargs={"prompt": qa_prompt},
                                verbose=True,
                                return_source_documents=True
                            )
                        
                        # Pass the chat history as a separate parameter
                        response = qa.invoke({
                            "question": prompt,
                            "chat_history": chat_history_tuples
                        })
                        
                        answer = response["answer"]
                        
                        # Display assistant response
                        with st.chat_message("assistant"):
                            st.markdown(answer)
                        
                        # Add assistant response to chat history
                        st.session_state.chat_history.append({"role": "assistant", "content": answer})
                        
                        # Also update the old conversation_chatbot format for backward compatibility
                        metadata = {"kb_used": st.session_state.selected_option, "strict_kb": strict_kb}
                        st.session_state.conversation_chatbot.append((prompt, answer, metadata))
                        
                        # Showing source documents
                        with st.expander("View source documents"):
                            for i, doc in enumerate(response.get("source_documents", [])):
                                st.write(f"**Source {i+1}:**")
                                st.write(doc.page_content)
                                st.write("---")
                        
                        # Display which KB was used
                        kb_mode = "strict" if strict_kb else "flexible"
                        st.caption(f"*Knowledge base used: {st.session_state.selected_option} ({kb_mode} mode)*")
                        
                    except Exception as e:
                        st.error(f"Error processing your query: {e}")
                        print(e)
                        if "assert d == self.d" in str(e):
                            st.warning("Dimension mismatch detected. Please rebuild the knowledge base from the sidebar.")
        else:
            st.warning("Please select a knowledge base from the sidebar")
    
    # Hybrid Mode (KB + General Knowledge)
    elif st.session_state.chat_mode == "hybrid":
        prompt = st.chat_input("Ask a question (I'll use both documents and general knowledge)")
        
        if prompt:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Getting Answer..."):
                try:
                    # Get relevant documents from the knowledge base
                    if st.session_state.book_docsearch:
                        chunks_to_retrieve = 3  # Fewer chunks in hybrid mode
                        retriever = st.session_state.book_docsearch.as_retriever(
                            search_type="similarity", 
                            search_kwargs={"k": chunks_to_retrieve}
                        )
                        docs = retriever.get_relevant_documents(prompt)
                        
                        # Extract content from documents
                        doc_content = "\n\n".join([doc.page_content for doc in docs])
                        
                        # Create hybrid prompt template
                        hybrid_template = """You are a helpful assistant that can use both document information and general knowledge.
                        
                        CONTEXT INFORMATION FROM DOCUMENTS:
                        {context}
                        
                        USER QUESTION: {question}
                        
                        Instructions:
                        1. If the question can be answered using the document context, use that information first.
                        2. You may supplement with your general knowledge when appropriate.
                        3. If the document context doesn't contain relevant information, you should rely on your general knowledge.
                        4. Always make it clear when you're using general knowledge versus document information.
                        
                        ANSWER:"""
                        
                        # Create a prompt from the template
                        prompt_template = ChatPromptTemplate.from_template(hybrid_template)
                        
                        # Create a chain
                        hybrid_chain = LLMChain(
                            llm=llm,
                            prompt=prompt_template
                        )
                        
                        # Get response
                        response = hybrid_chain.invoke({
                            "context": doc_content,
                            "question": prompt
                        })
                        
                        answer = response["text"]
                        
                        # Display source documents
                        with st.expander("View source documents"):
                            for i, doc in enumerate(docs):
                                st.write(f"**Source {i+1}:**")
                                st.write(doc.page_content)
                                st.write("---")
                    else:
                        # Fall back to general conversation if no knowledge base
                        messages = []
                        for message in st.session_state.chat_history:
                            messages.append({"role": message["role"], "content": message["content"]})
                        
                        response = llm.invoke(messages)
                        answer = response.content
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Also update the old conversation_chatbot format for backward compatibility
                    metadata = {"mode": "hybrid", "kb_used": st.session_state.selected_option if st.session_state.book_docsearch else None}
                    st.session_state.conversation_chatbot.append((prompt, answer, metadata))
                    
                    # Display which mode was used
                    st.caption("*Hybrid mode (documents + general knowledge)*")
                    
                except Exception as e:
                    st.error(f"Error processing your query: {e}")
                    print(e)
    
    # General Conversation Mode (without knowledge base)
    else:
        prompt = st.chat_input("Chat with the AI assistant")
        
        if prompt:
            # Add user message to chat history
            st.session_state.chat_history.append({"role": "user", "content": prompt})
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(prompt)
            
            with st.spinner("Thinking..."):
                try:
                    # Create a conversation history to pass to the model
                    messages = []
                    for message in st.session_state.chat_history:
                        if message["role"] == "user":
                            messages.append({"role": "user", "content": message["content"]})
                        elif message["role"] == "assistant":
                            messages.append({"role": "assistant", "content": message["content"]})
                    
                    # Get response from the model
                    response = llm.invoke(messages)
                    answer = response.content
                    
                    # Display assistant response
                    with st.chat_message("assistant"):
                        st.markdown(answer)
                    
                    # Add assistant response to chat history
                    st.session_state.chat_history.append({"role": "assistant", "content": answer})
                    
                    # Also update the old conversation_chatbot format for backward compatibility
                    metadata = {"mode": "general_conversation"}
                    st.session_state.conversation_chatbot.append((prompt, answer, metadata))
                    
                    # Update memory
                    st.session_state.memory.chat_memory.add_user_message(prompt)
                    st.session_state.memory.chat_memory.add_ai_message(answer)
                    
                    # Display which mode was used
                    st.caption("*General conversation mode (no knowledge base)*")
                    
                except Exception as e:
                    st.error(f"Error processing your query: {e}")
                    print(e)
            
# For initialization of session variables
def initial(flag=False):
    path="db"
    if 'existing_indices' not in st.session_state or flag:
        if os.path.exists(path):
            st.session_state.existing_indices = [name for name in os.listdir(path) if os.path.isdir(os.path.join(path, name))]
        else:
            st.session_state.existing_indices = []
            
    if ('selected_option' not in st.session_state) or flag:
        try:
            st.session_state.selected_option = st.session_state.existing_indices[0]
        except:
            st.session_state.selected_option = None
    
    if 'conversation_chatbot' not in st.session_state:
        st.session_state.conversation_chatbot =  []
    if 'book_docsearch' not in st.session_state:
        st.session_state.book_docsearch = None

def load_kb():
    # Load the selected index from local storage
    if st.session_state.selected_option:
        try:
            kb_path = os.path.join("db", st.session_state.selected_option)
            
            # Check if this knowledge base has been migrated
            embedding_info_path = os.path.join(kb_path, "embedding_info.json")
            if os.path.exists(embedding_info_path):
                with open(embedding_info_path, "r") as f:
                    embedding_info = json.load(f)
                    
                # Check if the embedding model matches
                if embedding_info.get("model") != "text-embedding-3-small":
                    st.sidebar.warning(f"⚠️ This knowledge base was created with {embedding_info.get('model')} but you're using text-embedding-3-small.")
            
            st.session_state.book_docsearch = FAISS.load_local(
                kb_path, 
                embeddings, 
                allow_dangerous_deserialization=True
            )
            st.sidebar.success(f"Loaded knowledge base: {st.session_state.selected_option}")
        except Exception as e:
            error_msg = str(e)
            st.sidebar.error(f"Error loading knowledge base: {error_msg}")
            
            # Check if it's a dimension mismatch error
            if "assert d == self.d" in error_msg:
                st.sidebar.warning("⚠️ This appears to be a dimension mismatch error. The knowledge base was created with a different embedding model.")
                
                # Check if we have original documents to rebuild
                original_docs_path = os.path.join("db", st.session_state.selected_option, "original_docs")
                if os.path.exists(original_docs_path) and len(os.listdir(original_docs_path)) > 0:
                    st.sidebar.info("Use the 'Rebuild Knowledge Base' button to fix this issue.")
                else:
                    st.sidebar.info("This knowledge base needs to be recreated. Please upload the documents again.")
            
            st.session_state.book_docsearch = None
    else:
        st.session_state.book_docsearch = None

# Load chat history from file
def load_chat_history(history_path):
    try:
        with open(history_path, "rb") as f:
            loaded_history = pickle.load(f)
            st.session_state.chat_history = loaded_history
            
            # Update memory from loaded history
            update_memory_from_history()
            
            # Also update conversation_chatbot for backward compatibility
            st.session_state.conversation_chatbot = []
            for i in range(0, len(loaded_history), 2):
                if i+1 < len(loaded_history):
                    user_msg = loaded_history[i]["content"]
                    ai_msg = loaded_history[i+1]["content"]
                    st.session_state.conversation_chatbot.append((user_msg, ai_msg, {}))
            
            return True
    except Exception as e:
        st.error(f"Error loading chat history: {e}")
        return False

def main():
    initial(True)
    
    # Run migration on startup
    migrate_knowledge_bases()
    
    # Sidebar for knowledge base selection
    with st.sidebar:
        st.title("Chat Mode Selection")
        st.markdown("---")
        
        # Chat mode selection
        chat_mode = st.radio(
            "Select chat mode:",
            ["Knowledge Base Chat", "Hybrid Chat", "General Conversation"],
            index=0 if st.session_state.chat_mode == "kb" else 
                   1 if st.session_state.chat_mode == "hybrid" else 2,
            key="chat_mode_selection"
        )
        
        # Update session state based on selection
        if chat_mode == "Knowledge Base Chat" and st.session_state.chat_mode != "kb":
            st.session_state.chat_mode = "kb"
            st.rerun()
        elif chat_mode == "Hybrid Chat" and st.session_state.chat_mode != "hybrid":
            st.session_state.chat_mode = "hybrid"
            st.rerun()
        elif chat_mode == "General Conversation" and st.session_state.chat_mode != "general":
            st.session_state.chat_mode = "general"
            st.rerun()
        
        # Add KB mode toggle if in KB mode
        if st.session_state.chat_mode == "kb":
            st.markdown("### Knowledge Base Settings")
            st.session_state.strict_kb = st.toggle(
                "Strict KB mode (answers only from documents)", 
                value=st.session_state.get("strict_kb", True)
            )
        
        # Debug button
        if st.button("Debug Embedding Dimensions"):
            debug_embedding_dimensions()
        
        # Only show knowledge base selection if in KB mode or hybrid mode
        if st.session_state.chat_mode in ["kb", "hybrid"]:
            st.markdown("---")
            st.title("Knowledge Base Selection")
            
            if not st.session_state.existing_indices:
                st.warning("⚠️ No knowledge bases found. Please add a new index.")
                st.page_link("pages/Knowledge_Bases.py", label="Upload Files", icon="⬆️")
            else:
                file_list = []
                for index in st.session_state.existing_indices:
                    try:
                        with open(f"db/{index}/desc.json", "r") as openfile:
                            description = json.load(openfile)
                            file_list.append(",".join(description["file_names"]))
                    except:
                        file_list.append("No description available")
                
                st.write("### Select a knowledge base")
                selected_index = st.radio(
                    "Available knowledge bases:", 
                    st.session_state.existing_indices, 
                    captions=file_list, 
                    index=0 if st.session_state.selected_option in st.session_state.existing_indices else 0,
                    key="kb_selection"
                )
                
                if selected_index != st.session_state.selected_option:
                    st.session_state.selected_option = selected_index
                    load_kb()
                    st.info("Knowledge base changed - conversation history is maintained")
                
                st.markdown("---")
                st.write("### Knowledge Base Info")
                if st.session_state.selected_option:
                    try:
                        with open(f"db/{st.session_state.selected_option}/desc.json", "r") as openfile:
                            description = json.load(openfile)
                            st.write(f"**Files:** {', '.join(description['file_names'])}")
                            if 'description' in description:
                                st.write(f"**Description:** {description['description']}")
                            if 'created_at' in description:
                                st.write(f"**Created:** {description['created_at']}")
                    except:
                        st.write("No additional information available.")
                
                # Maintenance options for KB mode
                st.markdown("---")
                st.write("### Maintenance Options")
                
                col1, col2 = st.columns(2)
                
                # Add a rebuild button if the knowledge base has original documents
                with col1:
                    if st.session_state.selected_option and can_rebuild_kb(st.session_state.selected_option):
                        if st.button("Rebuild Knowledge Base"):
                            with st.spinner("Rebuilding index with current embedding model..."):
                                if rebuild_index(st.session_state.selected_option):
                                    st.success("Knowledge base rebuilt successfully!")
                                    # Reload the knowledge base
                                    load_kb()
                                    st.rerun()
                                else:
                                    st.error("Failed to rebuild knowledge base.")
                    else:
                        st.button("Rebuil Knowledge Base", disabled=True)
                        if st.session_state.selected_option:
                            st.info("No original documents available for rebuilding")
                
                # Button to refresh the knowledge base list
                with col2:
                    if st.button("Refresh Knowledge Bases"):
                        initial(True)
                        load_kb()
                
                # Link to upload page
                st.markdown("---")
                st.page_link("pages/Knowledge_Bases.py", label="Upload New Files", icon="⬆️")
        
        # Chat history options (available in both modes)
        st.markdown("---")
        st.write("### Chat History Options")
        
        col1, col2 = st.columns(2)
        
        # Clear chat history button
        with col1:
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.conversation_chatbot = []
                st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
                # Reset memory
                st.session_state.memory = ConversationBufferMemory(
                    memory_key="chat_history",
                    return_messages=True
                )
                st.rerun()
        
        # Save chat history button
        with col2:
            if st.button("Save Chat History") and (st.session_state.chat_history or st.session_state.conversation_chatbot):
                saved_path = save_chat_history()
                st.success(f"Chat history saved!")
        
        # Load chat history section
        if os.path.exists("chat_histories"):
            st.markdown("---")
            st.write("### Load Chat History")
            history_files = [f for f in os.listdir("chat_histories") if f.endswith(".pkl")]
            
            if history_files:
                selected_history = st.selectbox(
                    "Select a saved chat history:",
                    history_files,
                    format_func=lambda x: x.replace("chat_", "").replace(".pkl", "")
                )
                
                if st.button("Load Selected History"):
                    history_path = os.path.join("chat_histories", selected_history)
                    if load_chat_history(history_path):
                        st.success("Chat history loaded successfully!")
                        st.rerun()
            else:
                st.info("No saved chat histories found")
        
        # Link to chat history page if it exists
        if os.path.exists("pages/Chat_History.py"):
            st.page_link("pages/Chat_History.py", label="View Saved Chat Histories", icon="📜")
    
    # Main content area
    st.title("💰 Mutual Fund Chatbot")

    # Show different content based on chat mode
    if st.session_state.chat_mode == "kb":
        if st.session_state.selected_option:
            kb_mode = "strict" if st.session_state.get("strict_kb", True) else "flexible"
            st.info(f"Currently using knowledge base: **{st.session_state.selected_option}** in {kb_mode} mode")
            chatbot()
        else:
            st.warning("⚠️ No knowledge base selected. Please select one from the sidebar.")
    elif st.session_state.chat_mode == "hybrid":
        if st.session_state.selected_option:
            st.info(f"Hybrid mode: Using knowledge base **{st.session_state.selected_option}** + general knowledge")
            chatbot()
        else:
            st.warning("⚠️ No knowledge base selected. Hybrid mode will use only general knowledge.")
            chatbot()
    else:
        st.info("You are in general conversation mode. Ask any questions without using a knowledge base.")
        chatbot()
            
if __name__ == "__main__":
    main()