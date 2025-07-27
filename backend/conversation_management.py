import uuid
import os
from typing import List, Dict, Optional, Tuple
from pymongo import MongoClient, errors

# MongoDB connection pooling and error handling
_MONGO_CLIENT: Optional[MongoClient] = None

def get_mongo_client() -> MongoClient:
    global _MONGO_CLIENT
    if _MONGO_CLIENT is None:
        try:
            mongo_uri = os.getenv("MONGODB_URI", "mongodb://mongodb:27017/")
            _MONGO_CLIENT = MongoClient(
                mongo_uri,
                maxPoolSize=10,
                serverSelectionTimeoutMS=3000
            )
            # Trigger server selection to catch connection errors early
            _MONGO_CLIENT.admin.command('ping')
        except errors.PyMongoError as e:
            raise RuntimeError(f"Could not connect to MongoDB: {e}")
    return _MONGO_CLIENT

def get_conversation_history(conversation_id: Optional[str] = None) -> Tuple[bool, List[Dict], Optional[str]]:
    if conversation_id is None:
        return False, [], "No conversation_id provided"
    try:
        client = get_mongo_client()
        db = client["chat_db"]
        collection = db["conversations"]
        doc = collection.find_one({"conversation_id": conversation_id})
        if doc and "chat_history" in doc:
            return True, doc["chat_history"], None
        return False, [], "Conversation not found"
    except errors.PyMongoError as e:
        # Log error in production
        return False, [], str(e)

def append_conversation_history(conversation_id: str, new_chat: List[Dict]) -> Tuple[bool, Optional[dict], Optional[str]]:
    try:
        client = get_mongo_client()
        db = client["chat_db"]
        collection = db["conversations"]
        result = collection.update_one(
            {"conversation_id": conversation_id},
            {"$push": {"chat_history": {"$each": new_chat}}},
            upsert=True
        )
        if result.acknowledged:
            return True, {"acknowledged": True}, None
        else:
            return False, None, "Update not acknowledged"
    except errors.PyMongoError as e:
        # Log error in production
        return False, None, str(e)

def set_conversation_history(conversation_id: str, chat: List[Dict]) -> Tuple[bool, Optional[dict], Optional[str]]:
    try:
        client = get_mongo_client()
        db = client["chat_db"]
        collection = db["conversations"]
        result = collection.update_one(
            {"conversation_id": conversation_id},
            {"$set": {"chat_history": chat}},
            upsert=True
        )
        if result.acknowledged:
            return True, {"acknowledged": True}, None
        else:
            return False, None, "Update not acknowledged"
    except errors.PyMongoError as e:
        # Log error in production
        return False, None, str(e)

def set_new_conversation_history(user_query: str) -> Tuple[bool, Optional[Tuple[str, List[Dict]]], Optional[str]]:
    conversation_id = str(uuid.uuid4())
    chat_history = [{"role": "user", "content": user_query}]
    try:
        client = get_mongo_client()
        db = client["chat_db"]
        collection = db["conversations"]
        collection.insert_one({
            "conversation_id": conversation_id,
            "chat_history": chat_history
        })
        return True, (conversation_id, chat_history), None
    except errors.PyMongoError as e:
        # Log error in production
        return False, None, str(e)
