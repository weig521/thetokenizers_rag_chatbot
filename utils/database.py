import os
import uuid
from datetime import datetime
import json

import streamlit as st
import chromadb
from chromadb.api.models.Collection import Collection

def get_chroma_collection(persist_dir: str, collection: str) -> Collection:
    os.makedirs(persist_dir, exist_ok=True)
    client = chromadb.PersistentClient(path=persist_dir)
    try:
        return client.get_collection(collection)
    except Exception:
        return client.create_collection(collection)

class ChatDatabase:
    """
    In-memory per-Streamlit-session chat store.
    Structure: st.session_state.chat_db[session_id] = {
        "session_id", "user_id", "session_name", "created_at", "updated_at", "messages": [...]
    }
    """
    def __init__(self):
        if "chat_db" not in st.session_state:
            st.session_state.chat_db = {}

    def create_session(self, user_id: str, session_name: str) -> str:
        sid = str(uuid.uuid4())
        st.session_state.chat_db[sid] = {
            "session_id": sid,
            "user_id": user_id,
            "session_name": session_name,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
            "messages": []
        }
        return sid

    def get_user_sessions(self, user_id: str):
        return [s for s in st.session_state.chat_db.values() if s["user_id"] == user_id]

    def get_session_messages(self, session_id: str):
        s = st.session_state.chat_db.get(session_id)
        return s.get("messages", []) if s else []

    def add_message(self, session_id: str, role: str, content: str):
        s = st.session_state.chat_db.get(session_id)
        if not s:
            return
        s["messages"].append({"role": role, "content": content, "ts": datetime.utcnow().isoformat()})
        s["updated_at"] = datetime.utcnow().isoformat()

    def rename_session(self, session_id: str, new_name: str):
        s = st.session_state.chat_db.get(session_id)
        if s:
            s["session_name"] = new_name
            s["updated_at"] = datetime.utcnow().isoformat()

    def delete_session(self, session_id: str):
        st.session_state.chat_db.pop(session_id, None)
    
    def search_sessions(self, user_id: str, query: str) -> list:
        """
        Search sessions by session_name or any message content.
        Returns a list of session dicts.
        """
        all_sessions = self.get_user_sessions(user_id)
        if not query:
            return all_sessions

        q = (query or "").lower()
        matching = []
        for s in all_sessions:
            # name match
            if q in s["session_name"].lower():
                matching.append(s)
                continue
            # message match
            msgs = self.get_session_messages(s["session_id"])
            if any(q in m["content"].lower() for m in msgs):
                matching.append(s)
        return matching

    def export_session_json(self, user_id: str, session_id: str) -> str:
        """
        Export one session thread to a JSON string (name, created_at, messages).
        """
        sessions = self.get_user_sessions(user_id)
        session = next((x for x in sessions if x["session_id"] == session_id), None)
        messages = self.get_session_messages(session_id)
        export_data = {
            "session_name": session["session_name"] if session else "Unknown",
            "created_at": session["created_at"] if session else "",
            "messages": messages,
        }
        return json.dumps(export_data, indent=2)