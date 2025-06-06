# Local ChatCSV with DeepSeek R1 and Ollama

**ChatCSV** is a Retrieval-Augmented Generation (RAG) application that allows users to upload CSV documents and interact with them through a chatbot interface. The system uses advanced embedding models and a local vector store for efficient and accurate question-answering.

## Features

- **CSV Upload**: Upload one or multiple CSV documents to enable question-answering across their combined content.
- **RAG Workflow**: Combines retrieval and generation for high-quality responses.
- **Customizable Retrieval**: Adjust the number of retrieved results (`k`) and similarity threshold to fine-tune performance.
- **Memory Management**: Easily clear vector store and retrievers to reset the system.
- **Streamlit Interface**: A user-friendly web application for seamless interaction.

---

## Installation

NOTE:All the steps(commands) are for MAC/linux
Follow the steps below to set up and run the application:
Before Cloning the repo, please make sure yo have installed ollama and docker

### 1. Run locally running Qdrant db

```bash
docker pull qdrant/qdrant
docker run -p 6333:6333 -p 6334:6334 \
    -v $(pwd)/qdrant_storage:/qdrant/storage:z \
    qdrant/qdrant
```

### 2. Clone the Repository

```bash
git clone https://github.com/Muhiddin02/rag-deepseek-r1-ollama.git
cd rag-deepseek-r1-ollama
```

### 3. Create a Virtual Environment

```bash
python3 -m venv venv
source venv/bin/activate
```

### 4. Install Dependencies

Install the required Python packages:

```bash
pip3 install -r requirements.txt
```

Make sure to include the following packages in your `requirements.txt`:

```
langchain
langchain_ollama
langchain_community
streamlit-chat
qdrant-client
```

### 5. Pull Required Models for Ollama

To use the specified embedding and LLM models (`mxbai-embed-large` and `deepseek-r1`), download them via the `ollama` CLI:

```bash
ollama pull mxbai-embed-large
ollama pull deepseek-r1:latest
```

---

## Usage

### 1. Start the Application

Run the Streamlit app:

```bash
streamlit run app.py
```

### 2. Upload Documents

- Navigate to the **Upload a Document** section in the web interface.
- Upload one or multiple CSV files to process their content.
- Each file will be ingested automatically and confirmation messages will show processing time.

### 3. Ask Questions

- Type your question in the chat input box and press Enter.
- Adjust retrieval settings (`k` and similarity threshold) in the **Settings** section for better responses.

### 4. Clear Chat and Reset

- Use the **Clear Chat** button to reset the chat interface.

---

## Project Structure

```
.
├── app.py                  # Streamlit app for the user interface
├── rag.py                  # Core RAG logic for PDF ingestion and question-answering
├── requirements.txt        # List of required Python dependencies
├── data/                   # Have sample of csv file
└── README.md               # Project documentation
```

---

## Configuration

You can modify the following parameters in `rag.py` to suit your needs:

1. **Models**:

   - Default LLM: `deepseek-r1:latest` (7B parameters)
   - Default Embedding: `mxbai-embed-large` (1024 dimensions)
   - Change these in the `ChatCSV` class constructor or when initializing the class
   - Any Ollama-compatible model can be used by updating the `llm_model` parameter

2. **Chunking Parameters**:

   - `chunk_size=1024` and `chunk_overlap=100`
   - Adjust for larger or smaller document splits

3. **Retrieval Settings**:
   - Adjust `k` (number of retrieved results) and `score_threshold` in `ask()` to control the quality of retrieval.

---

## Requirements

- **Python**: 3.8+
- **Streamlit**: Web framework for the user interface.
- **Ollama**: For embedding and LLM models.
- **LangChain**: Core framework for RAG.
- **Qdrant**: Vector store for document embeddings.

---
