import os
from dotenv import find_dotenv, load_dotenv
from typing import List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import redis
import requests
import json
import logging

load_dotenv(find_dotenv())

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

r = redis.Redis(host="redis", port=6379, db=0)

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class Message(BaseModel):
    role: str
    content: str


class Conversation(BaseModel):
    conversation: List[Message]


@app.get("/{conversation_id}")
async def get_conversation(conversation_id: str):
    logger.info(f"Retrieving initial id {conversation_id}")
    existing_conversation_json = r.get(conversation_id)
    if existing_conversation_json:
        existing_conversation = json.loads(existing_conversation_json)
        return existing_conversation
    else:
        return {"error": "Conversation not found"}


@app.post("/{conversation_id}")
async def service2(conversation_id: str, conversation: Conversation):
    logger.info(f"Sending Conversation with ID {conversation_id} to OpenAI")
    existing_conversation_json = r.get(conversation_id)
    if existing_conversation_json:
        existing_conversation = json.loads(existing_conversation_json)
    else:
        existing_conversation = {
            "conversation": [
                {"role": "system", "content": "You are a helpful assistant."}
            ]
        }

    existing_conversation["conversation"].append(
        conversation.model_dump()["conversation"][-1]
    )

    try:
        service3 = os.environ.get("SERVICE3", "http://service3")
        response = requests.post(
            f"{service3}/{conversation_id}",
            json=existing_conversation,
        )
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Request to service3 failed: {e}")
        return {"error": f"Request to service3 failed: {e}"}

    assistant_message = response.json()["reply"]

    existing_conversation["conversation"].append(
        {"role": "assistant", "content": assistant_message}
    )

    r.set(conversation_id, json.dumps(existing_conversation))

    return existing_conversation
