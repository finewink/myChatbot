import streamlit as st
import os
import pickle
import pandas as pd
from datetime import datetime

# Page configuration
st.set_page_config(page_title="Chat History", page_icon="📜", layout="wide")
st.title("📜 Chat History")

# Create chat_histories directory if it doesn't exist
if not os.path.exists("chat_histories"):
    os.makedirs("chat_histories")

# Get list of saved chat histories
history_files = [f for f in os.listdir("chat_histories") if f.endswith(".pkl")]

if not history_files:
    st.info("No saved chat histories found.")
else:
    # Display available chat histories
    st.header("Available Chat Histories")
    
    # Create a dataframe of chat histories with metadata
    history_data = []
    for file in history_files:
        # Extract timestamp from filename (assuming format chat_YYYYMMDD_HHMMSS.pkl)
        try:
            timestamp_str = file.replace("chat_", "").replace(".pkl", "")
            timestamp = datetime.strptime(timestamp_str, "%Y%m%d_%H%M%S")
            formatted_date = timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Get file size
            file_path = os.path.join("chat_histories", file)
            file_size = os.path.getsize(file_path) / 1024  # Size in KB
            
            # Load the file to count messages
            with open(file_path, "rb") as f:
                chat_data = pickle.load(f)
            message_count = len(chat_data)
            
            history_data.append({
                "File": file,
                "Date": formatted_date,
                "Messages": message_count,
                "Size (KB)": f"{file_size:.2f}"
            })
        except Exception as e:
            history_data.append({
                "File": file,
                "Date": "Unknown",
                "Messages": "Error",
                "Size (KB)": "Error"
            })
    
    # Display as dataframe
    history_df = pd.DataFrame(history_data)
    st.dataframe(history_df, use_container_width=True)
    
    # Select a chat history to view
    selected_history = st.selectbox("Select a chat history to view", history_files)
    
    if selected_history:
        st.header(f"Viewing: {selected_history}")
        
        # Load the selected chat history
        history_path = os.path.join("chat_histories",  selected_history)
        with open(history_path, "rb") as f:
            chat_data = pickle.load(f)
        
        # Display the chat
        st.subheader("Chat Messages")
        for message in chat_data:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])
        
        # Option to load this chat history into the current session
        if st.button("Load this chat history into current chat"):
            st.session_state.chat_history = chat_data
            st.session_state.conversation_id = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Also update the old conversation_chatbot format for backward compatibility
            st.session_state.conversation_chatbot = []
            for i in range(0, len(chat_data), 2):
                if i+1 < len(chat_data):  # Make sure we have a pair
                    user_msg = chat_data[i]["content"]
                    assistant_msg = chat_data[i+1]["content"]
                    metadata = {"kb_used": st.session_state.get("selected_option", "Unknown")}
                    st.session_state.conversation_chatbot.append((user_msg, assistant_msg, metadata))
            
            st.success("Chat history loaded! Go to the main chat page to continue this conversation.")
        
        # Option to delete this chat history
        if st.button("Delete this chat history", type="primary", use_container_width=True):
            os.remove(history_path)
            st.success("Chat history deleted!")
            st.rerun()

# Option to delete all chat histories
with st.expander("Danger Zone"):
    if st.button("Delete ALL Chat Histories", type="primary", use_container_width=True):
        for file in history_files:
            os.remove(os.path.join("chat_histories", file))
        st.success("All chat histories deleted!")
        st.rerun()