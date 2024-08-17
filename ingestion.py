import os
from dotenv import load_dotenv

load_dotenv()
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_pinecone import PineconeVectorStore
if __name__ == '__main__':
    print("Ingesting...")
    loader = TextLoader("C:/Users/kunal/ice_breaker/mediumblog1.txt",encoding='utf-8')
    document = loader.load()

    print("splitting...")
    text_splitter = CharacterTextSplitter(chunk_size=1000,chunk_overlap=0)
    texts = text_splitter.split_documents(document)
    print(f"created {len(texts)} chunks")

    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    print("ingesting")
    PineconeVectorStore.from_documents(texts, embeddings, index_name=os.environ['INDEX_NAME'])
    print("finish")

