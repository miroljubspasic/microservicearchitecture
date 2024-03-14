import os
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.pgvector import PGVector
from langchain.document_loaders import DirectoryLoader, TextLoader

load_dotenv(find_dotenv())

db_user = os.environ.get("DB_USER", "admin")
db_password = os.environ.get("DB_PASSWORD", "admin")
db_host = os.environ.get("DB_HOST", "postres")
db_port = os.environ.get("DB_PORT", 5432)
db_name = os.environ.get("DB_NAME", "vectordb")

embeddings = OpenAIEmbeddings()
loader = DirectoryLoader(
    "./FAQ", glob="**/*.txt", loader_cls=TextLoader, show_progress=True
)
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=250, chunk_overlap=25)
docs = text_splitter.split_documents(documents)

# PGVector needs the connection string to the database.
CONNECTION_STRING = (
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)
COLLECTION_NAME = f"{db_name}"


PGVector.from_documents(
    docs,
    embeddings,
    collection_name=COLLECTION_NAME,
    connection_string=CONNECTION_STRING,
)
