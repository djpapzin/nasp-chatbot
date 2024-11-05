import streamlit as st
from streamlit_chat import message
from langchain.chains import ConversationalRetrievalChain
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader, TextLoader, Docx2txtLoader
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
import requests
import os
from dotenv import load_dotenv
import tempfile
from uuid import uuid4
import replicate
import json

load_dotenv()

def initialize_session_state():
    if 'history' not in st.session_state:
        st.session_state['history'] = []

    if 'generated' not in st.session_state:
        st.session_state['generated'] = ["Hello! Ask me anything about your documents 🤗"]

    if 'past' not in st.session_state:
        st.session_state['past'] = ["Hey! 👋"]

def conversation_chat(query, history):
    # Use a ChatPromptTemplate to generate a consistent prompt for the LLM
    prompt_template = ChatPromptTemplate([
        ("system", "You are a helpful assistant that provides information based on given documents."),
        ("user", "{query}")
    ])
    
    # Create the prompt with the user's query
    prompt = prompt_template.invoke({"query": query})

    # Debug: Print the generated prompt
    st.write("Generated Prompt: ", prompt)

    # Convert prompt to string
    prompt_string = "\n".join([f"{message.type}: {message.content}" for message in prompt.to_messages()])

    # Use replicate API to get the LLM response
    api_url = "https://api.replicate.com/v1/predictions"
    headers = {
        "Authorization": f"Token {os.getenv('REPLICATE_API_TOKEN')}",
        "Content-Type": "application/json"
    }
    data = {
        "version": "meta/meta-llama-3-8b-instruct",
        "input": {
            "top_k": 0,
            "top_p": 0.95,
            "prompt": prompt_string,
            "max_tokens": 512,
            "temperature": 0.7,
            "length_penalty": 1,
            "max_new_tokens": 512,
            "stop_sequences": ["<|end_of_text|>", "<|eot_id|>"],
            "presence_penalty": 0,
            "log_performance_metrics": False
        }
    }

    # Debug: Print data to be sent to API
    st.write("Request Data: ", data)

    response = requests.post(api_url, headers=headers, data=json.dumps(data))
    response_json = response.json()
    
    # Debug: Print response from API
    st.write("API Response: ", response_json)

    result = response_json.get("output", "Sorry, I couldn't generate a response.")
    
    history.append((query, result))
    return result

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
    # Use Hugging Face Inference API for embedding generation
    api_url = "https://api-inference.huggingface.co/models/sentence-transformers/all-mpnet-base-v2"
    headers = {
        "Authorization": f"Bearer {os.getenv('HUGGINGFACE_API_TOKEN')}"
    }

    embeddings = []
    for document in documents:
        response = requests.post(api_url, headers=headers, json={"inputs": document.page_content})
        response_json = response.json()
        embedding = response_json[0] if isinstance(response_json, list) else None
        if embedding:
            embeddings.append((document, embedding))

    # Create a simple in-memory dictionary to act as a vector store
    vector_store = {str(uuid4()): (doc, emb) for doc, emb in embeddings}
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
3de87a317e73cdf43eb0af1c72d0dda23c3f88f7