# from langchain.document_loaders import TextLoader
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import git
import os
import re
import threading
import pickle
import faiss
from queue import Queue
# from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate
from langchain.docstore.in_memory import InMemoryDocstore
from langchain.docstore.document import Document
from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_community.vectorstores import FAISS
from ollama_copilot_enterprise.chunker import HybridCodeChunker
from langchain_community.vectorstores import FAISS

allowed_extensions = ['.py', '.ipynb', '.md', '.json', '.yml']
model_name = "llama3.1:latest"

# Predefined templates for different file types
REPO_FILE_TEMPLATES = {
    '.py': 'This file contains Python code that defines {purpose}.',
    '.md': 'This markdown file provides information regarding {purpose}.',
    '.json': 'This JSON file is used for {purpose} configurations.',
    '.yml': 'This YAML file defines {purpose} settings.'
}

class Embedder:
    """
    Embedder class handles the cloning, loading, and chunking of GitHub repositories, 
    followed by embedding the chunks and allowing for conversational retrieval.

    Methods:
    - clone_repo: Clones a GitHub repository into a local folder.
    - extract_all_files: Extracts relevant files (e.g., .py, .md) from the repository.
    - chunk_files: Uses HybridCodeChunker to split the extracted files into logical chunks.
    - load_db: Loads or creates a vector store of embeddings.
    - save_db: Saves the vector store and associated metadata.
    - retrieve_results: Allows for conversational query retrieval using the embedded codebase.
    """

    def __init__(self, git_link=None, db_path="vectorstore_index.faiss", metadata_path="vectorstore_metadata.pkl") -> None:
        self.model = ChatOllama(model=model_name, temperature=0)
        self.embeddings = OllamaEmbeddings(model=model_name)
        self.MyQueue = Queue(maxsize=2)
        self.db_path = db_path
        self.metadata_path = metadata_path
        if git_link is not None:
            self.git_link = git_link
            last_name = self.git_link.split('/')[-1]
            self.clone_path = last_name.split('.')[0]
        else:
            self.clone_path = None
        self.chunker = HybridCodeChunker(chunk_size=1000, chunk_overlap=100)  # Use the hybrid chunker

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def clone_repo(self):
        """Clones the GitHub repository if it does not already exist locally."""
        if not os.path.exists(self.clone_path):
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        """Extracts all code files with allowed extensions from the repository."""
        print(f"Extracting files from {self.clone_path}")
        root_dir = self.clone_path
        self.docs = []
        allowed_extensions = ['.py', '.ipynb', '.md', '.json', '.yml']
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try:
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        documents = loader.load_and_split()
                        for doc in documents:
                            self.docs.append(doc.page_content)
                    except Exception as e:
                        pass
        print("All files extracted")

    def chunk_files(self):
        """Uses HybridCodeChunker to split the extracted files into manageable chunks."""
        self.texts = []
        for doc in self.docs:
            self.texts.extend(self.chunker.split_code(doc))  # Use the hybrid chunker

    def load_db(self):
        """Loads the vector store from disk or creates a new one from the extracted and chunked code."""
        print("Loading DB")
        if os.path.exists(self.db_path) and os.path.exists(self.metadata_path):
            # Load vector store from disk
            index = faiss.read_index(self.db_path)
            with open(self.metadata_path, "rb") as f:
                doc_texts = pickle.load(f)
            
            # Create a docstore and index_to_docstore_id map
            docstore = InMemoryDocstore({str(i): Document(page_content=t) for i, t in enumerate(doc_texts)})
            index_to_docstore_id = {i: str(i) for i in range(len(doc_texts))}
            
            # Initialize the FAISS vector store correctly
            self.vectorstore = FAISS(
                embedding_function=self.embeddings,  # Correctly pass the embedding function
                index=index, 
                docstore=docstore, 
                index_to_docstore_id=index_to_docstore_id
            )            
            self.retriever = self.vectorstore.as_retriever()
        else:
            # If not found, create a new vector store
            print("Creating a niew vectorstore")
            self.extract_all_files()
            self.chunk_files()
            texts = [t for t in self.texts]
            print("Creating vectorstore")
            self.vectorstore = FAISS.from_texts(texts, embedding=self.embeddings)
            self.retriever = self.vectorstore.as_retriever()
            self.save_db()

    def save_db(self):
        """Saves the FAISS index and metadata (document texts) to disk."""
        print("Saving DB")
        faiss.write_index(self.vectorstore.index, self.db_path)
        with open(self.metadata_path, "wb") as f:
            pickle.dump(self.texts, f)

    def retrieve_results(self, query):
        """Retrieves results for a conversational query using the embedded codebase."""
        chat_history = list(self.MyQueue.queue)
        qa = ConversationalRetrievalChain.from_llm(
            self.model,
            chain_type="stuff",
            retriever=self.retriever,
            condense_question_llm=ChatOllama(model="llama3.1:latest")
        )
        result = qa({"question": query, "chat_history": chat_history})
        self.add_to_queue((query, result["answer"]))
        return result['answer']




if __name__ == "__main__":
    em = Embedder()
    em.clone_path = "/home/developer/proyectos/the-learning-curve/Manim/ollama_copilot_enterprise/manim"
    # em.extract_all_files()
    em.load_db()
    print(em.retrieve_results("How can I create a circle using manim"))

