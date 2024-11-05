import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
import replicate
import os
from dotenv import load_dotenv
import tempfile
from uuid import uuid4

load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your documents 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, history):
    prompt_template = "<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\nYou are a helpful assistant.<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
    prompt = prompt_template.format(prompt=query)

    response = ""
    for event in replicate.stream(
        "meta/meta-llama-3-8b-instruct",
        input={
            "top_k": 0,
            "top_p": 0.95,
            "prompt": prompt,
            "max_tokens": 512,
            "temperature": 0.7,
            "length_penalty": 1,
            "max_new_tokens": 512,
            "stop_sequences": "<|end_of_text|>,<|eot_id|>",
            "presence_penalty": 0,
            "log_performance_metrics": False
        },
    ):
        response += str(event)

    history.append((query, response))
    return response

def display_chat_history():
    reply_container = st.container()
    container = st.container()

    with container:
        with st.form(key='my_form', clear_on_submit=True):
            user_input = st.text_input("Question:", placeholder="Ask about your Documents", key='input')
            submit_button = st.form_submit_button(label='Send')

        if submit_button and user_input:
            with st.spinner('Generating response...'):
                output = conversation_chat(user_input, st.session_state['history'])

            st.session_state['past'].append(user_input)
            st.session_state['generated'].append(output)

    if st.session_state['generated']:
        with reply_container:
            for i in range(len(st.session_state['generated'])):
                message(st.session_state["past"][i], is_user=True, key=str(i) + '_user', avatar_style="thumbs")
                message(st.session_state['generated'][i], key=str(i), avatar_style="fun-emoji")

def create_vector_store(documents):
    # Initialize the embedding model using Hugging Face API
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"  # Use the proper attribute name
    )

    # Create Chroma vector store (persistence enabled for keeping embeddings even after reloading)
    vector_store = Chroma(
        collection_name="example_collection",
        embedding_function=embedding_model,
        persist_directory="./chroma_langchain_db"  # Chroma persists data here
    )

    # Add documents to the vector store
    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)

    return vector_store

def main():
    load_dotenv()
    initialize_session_state()
    st.title("Kwantu Chatbot")
    st.sidebar.title("Document Processing")
    uploaded_files = st.sidebar.file_uploader("Upload files", accept_multiple_files=True)

    if uploaded_files:
        text = []
        for file in uploaded_files:
            file_extension = os.path.splitext(file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_file.write(file.read())
                temp_file_path = temp_file.name

            loader = None
            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
            elif file_extension == ".docx" or file_extension == ".doc":
                loader = Docx2txtLoader(temp_file_path)
            elif file_extension == ".txt":
                loader = TextLoader(temp_file_path)

            if loader:
                text.extend(loader.load())
                os.remove(temp_file_path)

        text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=100, length_function=len)
        text_chunks = text_splitter.split_documents(text)

        documents = [Document(page_content=chunk.page_content, metadata={"source": "uploaded_file"}) for chunk in text_chunks]

        if documents:
            vector_store = create_vector_store(documents)
            display_chat_history()
        else:
            st.warning("No content available to create vector store. Please check your document.")

if __name__ == "__main__":
    main()
