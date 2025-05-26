# rag.py
from langchain_core.globals import set_verbose, set_debug
from langchain_ollama import ChatOllama, OllamaEmbeddings
from langchain.schema.output_parser import StrOutputParser
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema.runnable import RunnablePassthrough
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain_core.prompts import ChatPromptTemplate
import logging
from langchain_community.document_loaders import CSVLoader
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

set_debug(True)
set_verbose(True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ChatCSV:
    """A class for handling CSV ingestion and question answering using RAG."""

    def __init__(self, llm_model: str = "deepseek-r1:latest", embedding_model: str = "mxbai-embed-large"):
        """
        Initialize the ChatCSV instance with an LLM and embedding model.
        """
        self.model = ChatOllama(model=llm_model)
        self.collection_name = "my_collection"
        self.embeddings = OllamaEmbeddings(model=embedding_model)
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, chunk_overlap=100)
        self.prompt = ChatPromptTemplate.from_template(
            """
            You are a helpful assistant answering questions based on the uploaded document.
            Context:
            {context}
            
            Question:
            {question}
            
            Answer concisely and accurately in three sentences or less.
            """
        )
        self.client = QdrantClient(
            host="localhost", port=6333  # Or use `url="https://your-qdrant.cloud"` for cloud
        )
        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
        )

        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )
        self.retriever = None

    def csv_file_loader(self, file_paths):
        # Load each CSV file individually
        loader = CSVLoader(file_path=file_paths, encoding="utf-8")
        return loader.load()

    def ingest_csv_file(self, csv_file_path: str):
        docs = self.csv_file_loader(file_paths=csv_file_path)

        chunks = self.text_splitter.split_documents(docs)

        self.vector_store = Qdrant(
            client=self.client,
            collection_name=self.collection_name,
            embeddings=self.embeddings
        )

        self.vector_store.add_documents(chunks)

    def ask(self, query: str, k: int = 1, score_threshold: float = 0.7):
        print(self.client.get_collection(self.collection_name))

        if not self.vector_store:
            raise ValueError(
                "No vector store found. Please ingest a document first.")

        if not self.retriever:
            self.retriever = self.vector_store.as_retriever(
                search_type="similarity_score_threshold",
                search_kwargs={"k": k, "score_threshold": score_threshold},
            )

        logger.info(f"Retrieving context for query: {query}")
        retrieved_docs = self.retriever.invoke(query)

        if not retrieved_docs:
            return "No relevant context found in the document to answer your question."

        formatted_input = {
            "context": "\n\n".join(doc.page_content for doc in retrieved_docs),
            "question": query,
        }

        # Build the RAG chain
        chain = (
            RunnablePassthrough()  # Passes the input as-is
            | self.prompt           # Formats the input for the LLM
            | self.model            # Queries the LLM
            | StrOutputParser()     # Parses the LLM's output
        )
        logger.info("Generating response using the LLM.")
        return chain.invoke(formatted_input)

    def clear(self):
        """
        Reset the retriever.
        """
        logger.info("Clearing retriever.")
        self.retriever = None
