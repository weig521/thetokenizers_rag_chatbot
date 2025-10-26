import os
import json
import requests
from typing import Iterator, Tuple
from utils.database import get_chroma_collection
from dotenv import load_dotenv
from typing import List, Dict, Any
import numpy as np
from urllib.parse import urlparse
from sentence_transformers import SentenceTransformer, CrossEncoder
load_dotenv()



CROSS_ENCODER_MODEL = os.getenv("CROSS_ENCODER_MODEL", "cross-encoder/ms-marco-MiniLM-L-6-v2")

DEFAULT_SYSTEM_PROMPT = (
    "You are the USF Onboarding Assistant for Admissions, Orientation, and Registrar. "
    "Answer ONLY from the provided CONTEXT. Be concise. Add inline [Source N] markers "
    "that match the numbered sources in CONTEXT. If an answer is not in CONTEXT, say "
    "you don't know and suggest the correct USF office or link to contact."
)

def get_system_prompt() -> str:
    """
    Use RAG_SYSTEM_PROMPT from .env if present; else use DEFAULT_SYSTEM_PROMPT.
    Allows \\n in .env to produce real newlines.
    """
    text = os.getenv("RAG_SYSTEM_PROMPT")
    if text:
        try:
            # turn "\n" into actual newlines if author used escapes in .env
            text = bytes(text, "utf-8").decode("unicode_escape")
        except Exception:
            pass
        return text
    return DEFAULT_SYSTEM_PROMPT

class GemmaEmbedder:
    def __init__(self, model_name="google/embeddinggemma-300m", device=None):
        # If you pass device explicitly, we honor it.
        # Else, allow an env override (EMBED_DEVICE=cuda|cpu|mps|npu).
        # Else, let SentenceTransformer auto-detect (uses GPU if present).
        env_device = (os.getenv("EMBED_DEVICE") or "").strip().lower()
        resolved = device if device else (env_device if env_device in {"cuda", "cpu", "mps", "npu"} else None)
        self.model = SentenceTransformer(model_name, device=resolved)
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
    for d,m,dist in zip(docs, metas, dists):
        sim = 1.0 - float(dist)
        if sim >= min_sim:
            out.append({"doc": d, "meta": m, "sim": sim})
    out.sort(key=lambda x: x["sim"], reverse=True)
    return out[:k]

def format_context(hits: List[Dict[str,Any]]) -> str:
    """
    Numbered context so the model can cite inline like [Source N].
    """
    if not hits:
        return "No relevant context found."
    blocks = []
    for i, h in enumerate(hits[:5], 1):
        title = h["meta"].get("section_title") or h["meta"].get("filename") or "Section"
        blocks.append(f"Source {i}: {title}\n{h['doc']}")
    return "\n\n---\n\n".join(blocks)

def build_sources_block(hits: List[Dict[str, Any]]) -> str:
    """
    Compact, deduped Sources block (no URL fallback):
      N. [Title](canonical) — Category · filename
         ↳ short_url(canonical)
    If no canonical is present, prints:
      N. Title — Category · filename
    """
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

        # Dedup by (title, canonical); don’t consider any non-canonical path
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
            # No canonical? No link, no URL line.
            lines.append(f"{i}. {title} — {cat}{suffix}")

        if i >= 5:
            break

    return "\n".join(lines)


def build_system_prompt(base: str, context: str) -> str:
    return f"{base}\n\nContext:\n{context}"

def ollama_chat_stream(messages, model: str, url: str, options=None, timeout: int = 120):
    """
    Stream assistant text deltas from Ollama /api/chat.
    Expects messages like [{"role":"system","content":"..."}, {"role":"user","content":"..."}].
    Yields each content delta (str) until done.
    """
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

_CE = None  # singleton
def _get_cross_encoder(model_name: str | None = None) -> CrossEncoder:
     """Load once; reuse across calls."""
     global _CE
     if _CE is None:
         _CE = CrossEncoder(model_name or CROSS_ENCODER_MODEL)
     return _CE

def rerank_hits(query: str, hits: list[dict], top_k: int = 5, model_name: str | None = None) -> list[dict]:
     """
     Cross-encoder reranking:
       1) form (query, chunk_text) pairs
       2) ce.predict → relevance scores
       3) sort desc, keep top_k
     Returns the same hit dicts, adding 'rerank' with the CE score.
     """
     if not hits:
         return []
     ce = _get_cross_encoder(model_name)
     pairs = [(query, h.get("doc", "")[:1200]) for h in hits]   # keep inputs compact
     scores = ce.predict(pairs)                                  # higher = better
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
    """
    Retrieval -> numbered CONTEXT -> stream answer -> append compact Sources block.
    Yields tuples: ("delta", partial_text) ... then ("final", full_text_with_sources).
    """
    # 1) Retrieval using your exact utilities
    col = get_chroma_collection(chroma_persist, chroma_collection)
    embedder = GemmaEmbedder(embed_model)
    initial_hits = retrieve(col, user_text, embedder, k=20, min_sim=0.0)  # recall-first
    hits = rerank_hits(user_text, initial_hits, top_k=5)                  # precision pass
    ctx = format_context(hits)  # numbered for [Source N]

    # 2) Build messages
    system = system_prompt if system_prompt else get_system_prompt()
    messages = [
        {"role": "system", "content": f"{system}\n\nCONTEXT:\n{ctx}"},
        {"role": "user", "content": user_text},
    ]

    # 3) Stream from Ollama
    response_text = ""
    for delta in ollama_chat_stream(messages, model=ollama_model, url=ollama_url, options={"temperature": 0.2}):
        response_text += delta
        yield ("delta", response_text)

    # 4) Append Sources block
    sources_block = build_sources_block(hits)
    final = (response_text or "").rstrip()
    if sources_block:
        final += "\n\n**Sources**\n" + sources_block
    yield ("final", final)

