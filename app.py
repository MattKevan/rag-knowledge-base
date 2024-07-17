import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.supabase import SupabaseVectorStore

# Load environment variables
load_dotenv()

# Set environment variables
os.environ['TOKENIZERS_PARALLELISM'] = 'False'
DB_CONNECTION = os.getenv('DB_CONNECTION')

# Initialize components
@st.cache_resource
def initialize_components():
    vector_store = SupabaseVectorStore(
        postgres_connection_string=DB_CONNECTION, 
        collection_name='knowledgebase'
    )
    
    embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    llm = OpenAI(model="gpt-3.5-turbo", api_key=os.getenv('OPENAI_API_KEY'))
    
    index = VectorStoreIndex.from_vector_store(
        vector_store
    )
    
    return index
# Custom CSS for sticky chat input
def local_css():
    st.markdown("""
    <style>
        .stApp {
            height: 100vh;
        }
        .main {
            padding-bottom: 100px;
        }
        .sticky-chat-input {
            position: fixed;
            bottom: 0;
            left: 0;
            right: 0;
            padding: 20px;
            background-color: white;
            z-index: 100;
        }
    </style>
    """, unsafe_allow_html=True)
    
# Streamlit app
def main():
    st.set_page_config(page_title="HST knowledge hub", page_icon="🔎", layout="centered", initial_sidebar_state="auto", menu_items=None)

    st.title("Knowledge hub")

    # Initialize components with the 'site' collection
    index = initialize_components()

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["Standard", "Citations", "Chat"])

    with tab1:
        st.header("Standard search")
        query = st.text_input("Enter your question:", key="standard_query")
        if query:
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            st.markdown("### Answer")
            st.write(response.response)

    with tab2:
        st.header("Search with citations")
        query = st.text_input("Enter your question:", key="citations_query")
        if query:
            citation_query_engine = CitationQueryEngine.from_args(
                index,
                similarity_top_k=3,
                citation_chunk_size=512,
            )
            response = citation_query_engine.query(query)
            st.markdown("### Answer")
            st.write(response.response)
            
            st.markdown("### Sources")
            for i, source_node in enumerate(response.source_nodes):
                st.markdown(f"**Source {i + 1}:**")
                st.markdown(source_node.node.get_content())
                st.markdown("---")

    with tab3:
        st.header("Chat interface")
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "How can I help you today?"}
            ]

        if "chat_engine" not in st.session_state.keys():
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="condense_question", verbose=True, streaming=True
            )

        if prompt := st.chat_input(
            "Ask a question"
        ):  # Prompt for user input and save to chat history
            st.session_state.messages.append({"role": "user", "content": prompt})


        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        # If last message is not from assistant, generate a new response
        if st.session_state.messages[-1]["role"] != "assistant":
            with st.chat_message("assistant"):
                response_stream = st.session_state.chat_engine.stream_chat(prompt)
                st.write_stream(response_stream.response_gen)
                message = {"role": "assistant", "content": response_stream.response}
                # Add response to message history
                st.session_state.messages.append(message)

if __name__ == "__main__":
    main()