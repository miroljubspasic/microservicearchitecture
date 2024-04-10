from typing import List
from fastapi import FastAPI
from pydantic import BaseModel
import os
from fastapi.middleware.cors import CORSMiddleware
import openai
from langchain.prompts import PromptTemplate
import logging
from dotenv import find_dotenv, load_dotenv
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain_openai import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.pgvector import PGVector
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.prompts import (
    PromptTemplate,
    SystemMessagePromptTemplate,
)

ROLE_CLASS_MAP = {"assistant": AIMessage, "user": HumanMessage, "system": SystemMessage}

load_dotenv(find_dotenv())

openai.api_key = os.environ.get("OPENAI_API_KEY")

db_user = os.environ.get("DB_USER", "admin")
db_password = os.environ.get("DB_PASSWORD", "admin")
db_host = os.environ.get("DB_HOST", "postres")
db_port = os.environ.get("DB_PORT", 5432)
db_name = os.environ.get("DB_NAME", "vectordb")

CONNECTION_STRING = (
    f"postgresql+psycopg2://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    conversation: List[Message]


embeddings = OpenAIEmbeddings()
chat = ChatOpenAI(temperature=0)
store = PGVector(
    collection_name=db_name,
    connection_string=CONNECTION_STRING,
    embedding_function=embeddings,
)
retriever = store.as_retriever()

prompt_template = """As a FAQ Bot for our restaurant, you have the following information about our restaurant:

{context}

Please provide the most suitable response for the users question.
Answer:"""

prompt = PromptTemplate(template=prompt_template, input_variables=["context"])
system_message_prompt = SystemMessagePromptTemplate(prompt=prompt)


def create_messages(conversation):
    return [
        ROLE_CLASS_MAP[message.role](content=message.content)
        for message in conversation
    ]


def format_docs(docs):
    formatted_docs = []
    for doc in docs:
        formatted_doc = "Source: " + doc.metadata["source"]
        formatted_docs.append(formatted_doc)
    return "\n".join(formatted_docs)


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Creating new application instance
app = FastAPI()

# This is a decorator that describes the route (/{conversation_id}) and the HTTP method (POST)
@app.post("/{conversation_id}")
# Definition of asynchronous function service3 that takes two parameters:
# conversation_id (type str) and conversation (type Conversation)
async def service3(conversation_id: str, conversation: Conversation):
    # Get the content of the last message in the conversation
    query = conversation.conversation[-1].content

    # Get the documents relevant to the query using the retriever module
    docs = retriever.get_relevant_documents(query=query)
    # Format the retrieved documents for further processing
    docs = format_docs(docs=docs)

    # Prepare a system message prompt using the formatted documents
    prompt = system_message_prompt.format(context=docs)
    # Create a list of messages by concatenating the prompt and the content of the conversation
    messages = [prompt] + create_messages(conversation=conversation.conversation)

    # Invoke a chat function with the prepared messages
    result = chat.invoke(messages)

    # Return a dictionary with the conversation_id and the content of the result
    return {"id": conversation_id, "reply": result.content}