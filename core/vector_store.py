import os

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

CHROMA_DIR = "vector_db"
COLLECTION_NAME = "meeting_transcript"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

def load_embedding_model():
    embedding_model = HuggingFaceEmbeddings(
        model_name = EMBEDDING_MODEL,
        model_kwargs = {"device" : "cpu"}
    )
    return embedding_model

def build_vector_store(transcript : str) -> Chroma:
    splitter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap = 50)

    chunks = splitter.split_text(text = transcript)

    docs = [ Document(page_content = chunk, metadata = {'chunk_index' : i}) for i,chunk in enumerate(chunks)]

    embedding_model = load_embedding_model()

    vector_store = Chroma.from_documents(
        documents = docs,
        embedding = embedding_model,
        collection_name = COLLECTION_NAME,
        persist_directory = CHROMA_DIR
    ) 

    return vector_store

def load_vector_store() -> Chroma:
    embedding_model = load_embedding_model()
    vector_store = Chroma(
        collection_name = COLLECTION_NAME,
        embedding_function = embedding_model,
        persist_directory = CHROMA_DIR
    )

    return vector_store

def get_retriever(vector_store : Chroma, k : int = 4):
    retriever = vector_store.as_retriever(search_type = 'similarity', search_kwargs = {"k":k})
    return retriever


