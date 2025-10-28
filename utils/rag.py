import os
import re
import json
import requests
from typing import List, Dict, Any
from urllib.parse import urlparse

import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, CrossEncoder

from utils.database import get_chroma_collection

load_dotenv()

CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

DEFAULT_SYSTEM_PROMPT = (
    "You are the USF Onboarding Assistant for Admissions, Orientation, and Registrar. "
    "Answer ONLY from the provided CONTEXT. Be concise. Add inline [Source N] markers "
    "that match the numbered sources in CONTEXT. If an answer is not in CONTEXT, say "
    "you don't know and suggest the correct USF office or link to contact."
)

def get_system_prompt() -> str:
    text = os.getenv("RAG_SYSTEM_PROMPT")
    if text:
        try:
            text = bytes(text, "utf-8").decode("unicode_escape")
        except Exception:
            pass
        return text
    return DEFAULT_SYSTEM_PROMPT

# Token estimator
_WORD_OR_PUNC = re.compile(r"\w+|[^\w\s]", re.UNICODE)

def estimate_tokens(text: str) -> int:
    """
    Simple, dependency-free token estimate.
    Roughly counts words + punctuation. Works well enough for a session budget.
    """
    if not text:
        return 0
    return len(_WORD_OR_PUNC.findall(str(text)))

# Embedding
class GemmaEmbedder:
    def __init__(self, model_name="google/embeddinggemma-300m", device=None):
        self.model = SentenceTransformer(model_name, device="cpu")
    def embed_query(self, text: str) -> List[float]:
        v = self.model.encode([text], normalize_embeddings=True, convert_to_numpy=True)[0]
        return v.astype(np.float32).tolist()

def retrieve(col, query: str, embedder: GemmaEmbedder, k=5, min_sim=0.0):
    q = embedder.embed_query(query)
    res = col.query(query_embeddings=[q], n_results=k, include=["documents","metadatas","distances"])
    docs = res.get("documents", [[]])[0]
    metas = res.get("metadatas", [[]])[0]
    dists = res.get("distances", [[]])[0]
    out = []
    for d, m, dist in zip(docs, metas, dists):
        sim = 1.0 - float(dist)
        if sim >= min_sim:
            out.append({"doc": d, "meta": m, "sim": sim})
    out.sort(key=lambda x: x["sim"], reverse=True)
    return out[:k]

def format_context(hits: List[Dict[str,Any]]) -> str:
    if not hits:
        return "No relevant context found."
    blocks = []
    for i, h in enumerate(hits[:5], 1):
        title = h["meta"].get("section_title") or h["meta"].get("filename") or "Section"
        blocks.append(f"Source {i}: {title}\n{h['doc']}")
    return "\n\n---\n\n".join(blocks)

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    if not hits:
        return ""

    def short_url(u: str) -> str:
        if not u or not u.startswith("http"):
            return ""
        p = urlparse(u)
        disp = (p.netloc + p.path).rstrip("/")
        return disp if len(disp) <= 80 else disp[:77] + "…"

    lines, seen = [], set()
    i = 0
    for h in hits:
        m = h.get("meta") or {}
        title = m.get("section_title") or m.get("filename") or "Untitled"
        cat   = m.get("category") or "Orientation"
        file  = m.get("filename") or ""
        canon = (m.get("canonical") or "").strip()

        key = (title, canon)
        if key in seen:
            continue
        seen.add(key)

        i += 1
        suffix = f" · {file}" if file else ""
        if canon:
            disp = short_url(canon)
            lines.append(f"{i}. [{title}]({canon}) — {cat}{suffix}" + (f"\n    ↳ {disp}" if disp else ""))
        else:
            lines.append(f"{i}. {title} — {cat}{suffix}")

        if i >= 5:
            break

    return "\n".join(lines)

def ollama_chat_stream(messages, model: str, url: str, options=None, timeout: int = 120):
    payload = {"model": model, "messages": messages, "stream": True}
    if options:
        payload["options"] = options

    with requests.post(f"{url.rstrip('/')}/api/chat", json=payload, stream=True, timeout=timeout) as r:
        r.raise_for_status()
        for line in r.iter_lines():
            if not line:
                continue
            try:
                data = json.loads(line.decode("utf-8"))
            except Exception:
                continue
            msg = data.get("message", {})
            delta = msg.get("content", "")
            if delta:
                yield delta
            if data.get("done"):
                break

_CE = None
def _get_cross_encoder(model_name: str | None = None) -> CrossEncoder:
    global _CE
    if _CE is None:
        _CE = CrossEncoder(model_name or CROSS_ENCODER_MODEL)
    return _CE

def rerank_hits(query: str, hits: list[dict], top_k: int = 5, model_name: str | None = None) -> list[dict]:
    if not hits:
        return []
    ce = _get_cross_encoder(model_name)
    pairs = [(query, h.get("doc", "")[:1200]) for h in hits]
    scores = ce.predict(pairs)
    ranked = sorted(zip(hits, scores), key=lambda t: float(t[1]), reverse=True)
    out = []
    for h, s in ranked[:max(1, top_k)]:
        h2 = dict(h)
        h2["rerank"] = float(s)
        out.append(h2)
    return out

def generate_with_rag(
    user_text: str,
    system_prompt: str | None,
    chroma_persist: str,
    chroma_collection: str,
    embed_model: str,
    ollama_model: str,
    ollama_url: str,
):
    col = get_chroma_collection(chroma_persist, chroma_collection)
    embedder = GemmaEmbedder(embed_model)
    initial_hits = retrieve(col, user_text, embedder, k=20, min_sim=0.0)
    hits = rerank_hits(user_text, initial_hits, top_k=5)

    system = system_prompt if system_prompt else get_system_prompt()
    ctx = format_context(hits)

    messages = [
        {"role": "system", "content": f"{system}\n\nCONTEXT:\n{ctx}"},
        {"role": "user", "content": user_text},
    ]

    response_text = ""
    for delta in ollama_chat_stream(messages, model=ollama_model, url=ollama_url, options={"temperature": 0.2}):
        response_text += delta
        yield ("delta", response_text)

    sources_block = build_sources_block(hits)
    final = (response_text or "").rstrip()
    if sources_block:
        final += "\n\n**Sources**\n" + sources_block
    yield ("final", final)
