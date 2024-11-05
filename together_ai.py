import os
import requests
from together import Together
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor, as_completed
import time
from requests.exceptions import RequestException
import numpy as np
import json
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from uuid import uuid4
from langchain_core.documents import Document
from langchain_together import TogetherEmbeddings

# Load environment variables from .env file
load_dotenv()

# Set Together API key
api_key = os.getenv("TOGETHER_API_KEY")
os.environ["TOGETHER_API_KEY"] = api_key

# Initialize Together client
client = Together()

# Load PDF files from the 'docs' directory
pdf_files = [os.path.join('docs', file) for file in os.listdir('docs') if file.endswith('.pdf')]
print(f"PDF files found: {len(pdf_files)}")

# Initialize a list to hold the loaded documents
docs = []

# Load each PDF file individually
for pdf_file in pdf_files:
    loader = PyPDFLoader(pdf_file)  # Adjusted to pass a single file
    loaded_docs = loader.load()  # Assuming load() returns a list of documents
    docs.extend(loaded_docs)
    print(f"Loaded {len(loaded_docs)} documents from {pdf_file}")

# Check if any documents were loaded
if not docs:
    print("No documents were loaded. Please check your file paths or file formats.")
    exit("Exiting: No documents found.")

# Use an embedding model with a large context length to avoid splitting where possible
max_context_length = 8192  # M2-BERT-Retrieval-8k can handle up to 8192 tokens

# Avoid splitting unless the document exceeds the model's context length
text_splitter = RecursiveCharacterTextSplitter(chunk_size=max_context_length, chunk_overlap=0)
splits = text_splitter.split_documents(docs)

# Check if any text chunks were created
if not splits:
    print("No text chunks were created. Please check the text splitter settings.")
    exit("Exiting: No text chunks found.")
else:
    print(f"Number of text chunks created: {len(splits)}")

# Initialize TogetherEmbeddings for embedding generation
embeddings_model = TogetherEmbeddings(model="togethercomputer/m2-bert-80M-8k-retrieval")

# Generate embeddings for each split
documents = []
embeddings = []

for idx, doc in enumerate(splits):
    print(f"Generating embedding for document {idx + 1}/{len(splits)}")
    truncated_content = doc.page_content[:2000].replace('\n', ' ')
    try:
        embedding = embeddings_model.embed_documents([truncated_content])[0]
        if embedding:
            documents.append(Document(page_content=doc.page_content, metadata=doc.metadata))
            embeddings.append(embedding)
        else:
            print(f"Failed to get embedding for document {idx + 1}")
    except Exception as e:
        print(f"Failed to generate embedding for document {idx + 1}: {str(e)}")

# Check if any embeddings were generated
if not embeddings:
    print("No embeddings were generated. Please check the embedding API and input content.")
    exit("Exiting: No embeddings generated.")
else:
    print(f"Number of embeddings generated: {len(embeddings)}")

# Initialize FAISS index
index = faiss.IndexFlatL2(len(embeddings[0]))

# Create FAISS vector store
vector_store = FAISS(
    embedding_function=embeddings_model,  # Set to TogetherEmbeddings to use for adding new documents
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={}
)

# Add documents to the FAISS vector store
uuids = [str(uuid4()) for _ in range(len(documents))]
added_ids = vector_store.add_embeddings(text_embeddings=list(zip(uuids, embeddings)), metadatas=[doc.metadata for doc in documents])
print(f"Successfully added the following UUIDs to the vector store: {added_ids}")

# Retrieve and generate using the relevant snippets of the PDF content
retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})

# Adding a querying mechanism
while True:
    user_query = input("Enter your query (or 'exit' to quit): ")
    if user_query.lower() == 'exit':
        break
    # Retrieve relevant documents from the vector store
    context = retriever.invoke(user_query)
    formatted_context = "\n\n".join([f"Document {i+1} Content:\n{doc.page_content}" for i, doc in enumerate(context)])
    
    # Print the retrieved context
    print("\nRetrieved Context:\n")
    if not formatted_context.strip():
        print("No relevant documents found for your query.")
        continue
    print(formatted_context)
    
    # Optionally, generate a response using Together AI
    response = client.chat.completions.create(
        model="meta-llama/Llama-3.2-3B-Instruct-Turbo",
        messages=[{"role": "user", "content": f"User Query: {user_query}\n\nContext:\n{formatted_context}"}],
        max_tokens=512,
        temperature=0.7,
        top_p=0.7,
        top_k=50,
        repetition_penalty=1,
        stop=["<|eot_id|>","<|eom_id|>"],
        stream=True
    )
    
    # Print the response
    if not response:
        print("No response received from Together AI.")
    else:
        print("\nGenerated response:")
        for token in response:
            if hasattr(token, 'choices'):
                print(token.choices[0].delta.content, end='', flush=True)
            else:
                print(f"Unexpected token format: {token}")
    print("\n")

# Cleanup: Delete the documents using only successfully added IDs
try:
    if added_ids:
        vector_store.delete(ids=added_ids)  # Specify the exact IDs that were added to delete them
        print("FAISS vector store deleted successfully.")
    else:
        print("No documents were added, so none to delete.")
except ValueError as e:
    print(f"Error during deletion: {e}")
