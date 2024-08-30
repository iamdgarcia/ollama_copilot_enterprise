from langchain.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
import git
import os
from queue import Queue
local = False
if local:
    from dotenv import load_dotenv
    load_dotenv()


from langchain_core.vectorstores import InMemoryVectorStore

from langchain_ollama import OllamaEmbeddings, ChatOllama
model_name = "llama2:latest"
model_kwargs = {"device": "cpu"}
allowed_extensions = ['.py', '.ipynb', '.md']

from langchain.chains import ConversationalRetrievalChain

class Embedder:
    def __init__(self, git_link) -> None:
        self.git_link = git_link
        last_name = self.git_link.split('/')[-1]
        self.clone_path = last_name.split('.')[0]
        self.model = ChatOllama(model=model_name,temperature=0)  # switch to 'gpt-4'
        self.model.invoke("HELO")
        self.embeddings = OllamaEmbeddings(model=model_name,)
        self.MyQueue =  Queue(maxsize=2)

    def add_to_queue(self, value):
        if self.MyQueue.full():
            self.MyQueue.get()
        self.MyQueue.put(value)

    def clone_repo(self):
        if not os.path.exists(self.clone_path):
            # Clone the repository
            git.Repo.clone_from(self.git_link, self.clone_path)

    def extract_all_files(self):
        root_dir = self.clone_path
        self.docs = []
        for dirpath, dirnames, filenames in os.walk(root_dir):
            for file in filenames:
                file_extension = os.path.splitext(file)[1]
                if file_extension in allowed_extensions:
                    try: 
                        loader = TextLoader(os.path.join(dirpath, file), encoding='utf-8')
                        self.docs.extend(loader.load_and_split())
                    except Exception as e: 
                        pass
    
    def chunk_files(self):
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        self.texts = text_splitter.split_documents(self.docs)
        self.num_texts = len(self.texts)

    def load_db(self):
        self.extract_all_files()
        self.chunk_files()
        texts = [t.page_content for t in self.texts]
        vectorstore = InMemoryVectorStore.from_texts(texts,embedding=self.embeddings,)
        self.retriever = vectorstore.as_retriever()
    
    def delete_directory(self, path):
        if os.path.exists(path):
            for root, dirs, files in os.walk(path, topdown=False):
                for file in files:
                    file_path = os.path.join(root, file)
                    os.remove(file_path)
                for dir in dirs:
                    dir_path = os.path.join(root, dir)
                    os.rmdir(dir_path)
            os.rmdir(path)
        


    def retrieve_results(self, query):
        chat_history = list(self.MyQueue.queue)
        qa = ConversationalRetrievalChain.from_llm(self.model, chain_type="stuff", retriever=self.retriever, condense_question_llm = ChatOllama(model=model_name))
        result = qa({"question": query, "chat_history": chat_history})
        self.add_to_queue((query, result["answer"]))
        return result['answer']