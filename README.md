# NASP Chatbot

A RAG-powered chatbot for the National Agency of Social Protection, built with LangChain, Together AI, and FAISS vector storage.

## Features

- 📚 Document Processing: Supports PDF, DOCX, and TXT files
- 🔍 RAG Implementation: Using FAISS vector store and Together AI embeddings
- 💬 Multi-Platform: Available via Streamlit UI and Telegram bot
- 🌐 Pre-loaded Documents: Comes with key social protection documents
- 🔄 Real-time Processing: Dynamic document updates and vector store management
- 🌍 Translation Support: Planned support for multiple languages including Uzbek

## Tech Stack

- **LLM**: Mixtral-8x7B-Instruct-v0.1 via Together AI
- **Embeddings**: m2-bert-80M-8k-retrieval via Together AI
- **Vector Store**: FAISS (Facebook AI Similarity Search)
- **Framework**: LangChain
- **UI**: Streamlit and Telegram Bot
- **Document Processing**: Support for PDF, DOCX, and TXT files

## Prerequisites

- Python 3.8+
- Together AI API key
- Telegram Bot Token (for bot functionality)
- Virtual environment (recommended)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/djpapzin/kwantu-rag
cd kwantu-rag
```

2. Create and activate virtual environment:
```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# OR
.venv\Scripts\activate  # Windows
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

4. Create a `.env` file in the project root:
```plaintext
TOGETHER_API_KEY=your_together_api_key
TELEGRAM_BOT_TOKEN=your_telegram_bot_token  # Optional, for Telegram bot
```

## Project Structure

```
kwantu-rag/
├── src/
│   ├── default_docs/     # Default document storage
│   ├── document_manager.py
│   ├── vector_search.py
│   ├── llm.py
│   ├── ui.py
│   └── telegram_bot.py
├── faiss_index/         # Vector store location
├── setup_vector_store.py
├── main.py
└── requirements.txt
```

## Usage

1. Initialize the vector store with default documents:
```bash
python setup_vector_store.py
```

2. Run the Streamlit interface:
```bash
streamlit run main.py
```

3. (Optional) Run the Telegram bot:
```bash
python -m src.telegram_bot
```

## Features in Detail

### Document Processing
- Automatic processing of PDF, DOCX, and TXT files
- Chunk size: 1000 characters with 200 character overlap
- Metadata preservation (filename, page numbers)

### Vector Store
- FAISS-based vector storage
- Together AI embeddings (m2-bert-80M-8k-retrieval model)
- Persistent storage in `faiss_index` directory

### LLM Integration
- Model: Mixtral-8x7B-Instruct-v0.1
- Context-aware responses with document references
- Conversation memory management

## Contributing

Please read [CONTRIBUTING.md](CONTRIBUTING.md) for details on our code of conduct and the process for submitting pull requests.

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details.

## Acknowledgments

- Together AI for providing the embedding and LLM capabilities
- LangChain for the RAG framework
- FAISS for vector storage capabilities
