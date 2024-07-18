import streamlit as st
import os
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex, StorageContext
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai import OpenAI
from llama_index.core.query_engine import CitationQueryEngine
from llama_index.vector_stores.supabase import SupabaseVectorStore
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.core.query_engine import SubQuestionQueryEngine
from llama_index.core.callbacks import CallbackManager, LlamaDebugHandler
from llama_index.core import Settings
from llama_index.core.callbacks import CBEventType, EventPayload
import json

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
    
    llama_debug = LlamaDebugHandler(print_trace_on_end=True)
    callback_manager = CallbackManager([llama_debug])
    
    Settings.callback_manager = callback_manager
    
    index = VectorStoreIndex.from_vector_store(
        vector_store
    )
    
    return index, llama_debug

def extract_url_from_metadata(metadata):
    try:
        node_content = json.loads(metadata.get('_node_content', '{}'))
        url = node_content.get('metadata', {}).get('URL')
        return url if url else 'No URL available'
    except json.JSONDecodeError:
        return 'Error parsing metadata'
    
def clear_chat_history():
    st.session_state.messages = [
        {"role": "assistant", "content": "How can I help you today?"}
    ]
    st.session_state.chat_engine = None

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
    st.markdown("Answer questions based on public HST site content, including the course catalogue, hub content and help & FAQs.")

    # Initialize components
    index, llama_debug = initialize_components()

    # Create tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Standard", "Complex", "Citations", "Chat"])

    with tab1:
        st.header("Standard query")
        st.markdown("Get an answer to your question based on site content. Specific questions will give better results than more general ones.")

        query = st.text_input("What would you like to know?", key="standard_query")
        if query:
            query_engine = index.as_query_engine()
            response = query_engine.query(query)
            st.markdown("### Answer")
            st.write(response.response)

    with tab2:
        st.header("Complex query")
        st.markdown("Breaks your question into one or more sub-questions to generate a deeper and more nuanced answer. Good for comparisons or asking more than one question at once.")
        query = st.text_input("What would you like to know?", key="subquestion_query")
        if query:
            # Create the base query engine
            base_query_engine = index.as_query_engine()

            # Setup base query engine as tool
            query_engine_tools = [
                QueryEngineTool(
                    query_engine=base_query_engine,
                    metadata=ToolMetadata(
                        name="knowledgebase",
                        description="HST knowledgebase",
                    ),
                ),
            ]
            
            sub_question_engine = SubQuestionQueryEngine.from_defaults(
                query_engine_tools=query_engine_tools,
                use_async=True,
            )
            
            response = sub_question_engine.query(query)
            
            st.markdown("### Response")
            st.write(response.response)
            
            st.markdown("### Sub-Questions and Answers")
            for i, (start_event, end_event) in enumerate(
                llama_debug.get_event_pairs(CBEventType.SUB_QUESTION)
            ):
                qa_pair = end_event.payload[EventPayload.SUB_QUESTION]
                st.markdown(f"**Sub Question {i + 1}:** {qa_pair.sub_q.sub_question.strip()}")
                st.markdown(f"**Answer:** {qa_pair.answer.strip()}")
                st.markdown("---")

    with tab3:
        st.header("Search with citations")
        st.markdown("Shows the sources it uses to generate the answer, linking to the specific pages if possible. Good for finding further information or verifying responses.")

        query = st.text_input("What would you like to know?", key="citations_query")
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
                # Extract and display the URL from metadata
                url = source_node.node.metadata.get('URL', 'No URL available')
                if url != 'No link available':
                    st.markdown(f"[{url}]({url})")
                else:
                    st.markdown(url)
                st.markdown("---")

    

    with tab4:
        st.header("Chat interface")
        st.markdown("Remembers the context of previous messages. Good for going deeper or expanding on subjects.")
        if "messages" not in st.session_state:
            st.session_state.messages = [
                {"role": "assistant", "content": "How can I help you today?"}
            ]

        if "chat_engine" not in st.session_state.keys():
            st.session_state.chat_engine = index.as_chat_engine(
                chat_mode="condense_question", verbose=True, streaming=True
            )

        if prompt := st.chat_input("Ask a question"):
            st.session_state.messages.append({"role": "user", "content": prompt})

        # Display chat messages
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if st.button("Clear Chat"):
                clear_chat_history()

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