"""
USF Onboarding Assistant — Markdown-only Data Ingestion
Markdown (.md) → Clean → Chunk → Embed (EmbeddingGemma-300m) → Chroma

Install (one time):
  pip install chromadb sentence-transformers

Run:
  python data_ingestion.py ^
    --source data/raw ^
    --persist data/processed ^
    --collection usf_onboarding ^
    --chunk 900 --overlap 150 --batch 64
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import re
from pathlib import Path
from typing import Iterable, List, Optional, Dict
from dotenv import load_dotenv
load_dotenv()

# ------------------------------ deps ------------------------------
def _need(mod: str, pip_name: str):
    try:
        return __import__(mod)
    except Exception:
        raise RuntimeError(f"Missing dependency: {mod}. Install with: pip install {pip_name}")

chromadb = _need("chromadb", "chromadb")
st_mod = _need("sentence_transformers", "sentence-transformers")
from sentence_transformers import SentenceTransformer

try:
    import torch
except Exception:
    torch = None

EMBED_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL", "google/embeddinggemma-300m")

# ------------------------------ utils ------------------------------
MD_EXT = {".md"}

_WS = re.compile(r"[ \t\f\v]+")
_NL = re.compile(r"\n{3,}")
PARA_BREAK = re.compile(r"\n\s*\n")
BULLET = re.compile(r"^\s*(?:[\u2022\-\*\u25E6]|\d+\.)\s+")
YAML_FRONT = re.compile(r"^---\s*\n.*?\n---\s*\n", re.S)
USF_LINK = re.compile(r"https?://(?:www\.)?usf\.edu[^\s\]\)]+", re.I)

def first_usf_url(text: str) -> Optional[str]:
    """Return the first usf.edu URL (without fragment) if present."""
    if not text:
        return None
    m = USF_LINK.search(text)
    if not m:
        return None
    url = m.group(0)
    return url.split("#", 1)[0]

def derive_category(path: Path) -> str:
    """Use top-level folder name as category when available; default 'USF'."""
    try:
        parts = path.parts
        return parts[0] if len(parts) > 1 else "USF"
    except Exception:
        return "USF"
    
def sha1(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8")).hexdigest()

def l2_normalize(vec: List[float]) -> List[float]:
    n = math.sqrt(sum(x * x for x in vec))
    return vec if n == 0.0 else [x / n for x in vec]

def clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = YAML_FRONT.sub("", s, count=1)  # strip YAML front-matter if present
    s = _WS.sub(" ", s).strip()
    s = _NL.sub("\n\n", s)
    return s.strip()

def reflow_paragraphs(text: str) -> str:
    """
    - Keep real paragraph breaks (blank lines)
    - Join hard-wrapped lines within a paragraph into one line
    - Preserve bullet lines
    - De-hyphenate soft wraps: immuniza-\ntion -> immunization
    """
    text = re.sub(r"-\n(?=[a-z])", "", text)
    paras = PARA_BREAK.split(text)
    out: List[str] = []
    for p in paras:
        lines = [ln.strip() for ln in p.split("\n") if ln.strip()]
        if not lines:
            out.append("")
            continue
        buf: List[str] = []
        for ln in lines:
            if BULLET.match(ln):
                if buf:
                    out.append(" ".join(buf)); buf = []
                out.append(ln)
            else:
                buf.append(ln)
        if buf:
            out.append(" ".join(buf))
        out.append("")
    return "\n".join(out).strip()

def group_faq_blocks(text: str) -> str:
    """Merge Q + A style lines into larger blocks; pass-through if none."""
    q_start = re.compile(r"^\s*(?:Q[:\-\)]|Question\b|How\b|What\b|When\b|Where\b|Why\b|Can\b|Do\b|Does\b|Is\b|Are\b)", re.I)
    a_mark  = re.compile(r"^\s*A[:\-\)]\s*", re.I)
    lines = text.split("\n")
    blocks, cur = [], []
    saw_q = False
    for ln in lines:
        if not ln.strip():
            if cur: blocks.append("\n".join(cur)); cur=[]
            blocks.append(""); continue
        if q_start.match(ln):
            saw_q = True
            if cur: blocks.append("\n".join(cur)); cur=[]
            cur.append(ln)
        else:
            cur.append(a_mark.sub("", ln))
    if cur: blocks.append("\n".join(cur))
    return "\n\n".join(blocks) if saw_q else text

def recursive_chunks(text: str, chunk_size: int, overlap: int) -> List[str]:
    if len(text) <= chunk_size:
        return [text]
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    pieces, buf = [], ""
    for p in paras:
        if len(buf) + len(p) + 2 <= chunk_size:
            buf = (buf + "\n\n" + p).strip() if buf else p
        else:
            if buf: pieces.append(buf)
            if len(p) <= chunk_size:
                buf = p
            else:
                sents = re.split(r"(?<=[.!?])\s+", p)
                sbuf = ""
                for s in sents:
                    if len(sbuf) + len(s) + 1 <= chunk_size:
                        sbuf = (sbuf + " " + s).strip() if sbuf else s
                    else:
                        if sbuf: pieces.append(sbuf)
                        # sentence too long: fall back to sliding window
                        step = max(1, chunk_size - overlap)
                        for i in range(0, len(s), step):
                            pieces.append(s[i:i+chunk_size])
                        sbuf = ""
                if sbuf: pieces.append(sbuf)
                buf = ""
    if buf: pieces.append(buf)
    # re-pack to respect chunk_size
    out, cur = [], ""
    for p in pieces:
        if not cur: cur = p
        elif len(cur) + len(p) + 1 <= chunk_size:
            cur = cur + "\n" + p
        else:
            out.append(cur); cur = p
    if cur: out.append(cur)
    # add overlap
    if overlap > 0 and len(out) > 1:
        with_ol = [out[0]]
        for i in range(1, len(out)):
            prev_tail = out[i-1][-overlap:]
            with_ol.append((prev_tail + out[i]) if prev_tail else out[i])
        out = with_ol
    return out

def glue_short_chunks(chunks: List[str], min_chars: int = 300) -> List[str]:
    if not chunks: return chunks
    out: List[str] = []
    for ch in chunks:
        if out and len(out[-1]) < min_chars:
            out[-1] = (out[-1].rstrip() + "\n\n" + ch.lstrip()).strip()
        else:
            out.append(ch)
    return out

def fingerprint(text: str) -> str:
    norm = " ".join(text.lower().split())
    return hashlib.sha1(norm.encode("utf-8")).hexdigest()

def chunk_ids(base: str, chunks: List[str]) -> List[str]:
    return [f"{base}:{i}:{sha1(base+'|'+ch)[:12]}" for i, ch in enumerate(chunks)]

def iter_md_files(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*.md")):
        if p.is_file():
            yield p

def md_title(text: str, fallback: str) -> str:
    # first ATX H1 as title; else fallback to filename stem
    m = re.search(r"^#\s+(.*)$", text, flags=re.M)
    title = (m.group(1).strip() if m else None) or fallback
    # keep titles short in metadata
    return title[:200]

# ------------------------------ embedder ------------------------------
class GemmaEmbedder:
    """
    Local embedder for google/embeddinggemma-300m.
    Uses 'title: {title} | text: {content}' document prompt.
    """
    def __init__(self, model_name: str = EMBED_MODEL):
        device = "cuda" if (torch is not None and hasattr(torch, "cuda") and torch.cuda.is_available()) else "cpu"
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[embedder] {model_name} on {device}")

    def embed_docs(self, texts: List[str], titles: Optional[List[str]] = None, batch: int = 64) -> List[List[float]]:
        def fmt(i, t):
            title = (titles[i] if titles and i < len(titles) and titles[i] else "none")
            return f"title: {title} | text: {t}"
        inputs = [fmt(i, txt) for i, txt in enumerate(texts)]
        out: List[List[float]] = []
        for i in range(0, len(inputs), max(1, batch)):
            sub = inputs[i:i+batch]
            vecs = self.model.encode(sub, normalize_embeddings=True)
            if hasattr(vecs, "tolist"): vecs = vecs.tolist()
            out.extend([l2_normalize(v) for v in vecs])
        return out

# ------------------------------ Chroma ------------------------------
def get_chroma_collection(persist_dir: Path, name: str):
    client = chromadb.PersistentClient(path=str(persist_dir))
    try:
        col = client.get_collection(name=name, metadata={"hnsw:space": "cosine"})
    except Exception:
        col = client.create_collection(name=name, metadata={"hnsw:space": "cosine"})
    return col

def collection_has_doc(col, doc_fp: str) -> bool:
    """Skip re-ingestion if a document fingerprint already exists."""
    try:
        res = col.get(where={"doc_fp": doc_fp}, limit=1)
        return bool(res and res.get("ids"))
    except Exception:
        return False

# ------------------------------ main ------------------------------
def main():
    ap = argparse.ArgumentParser(
        description="RAG — Markdown ingestion (MD → chunks → EmbeddingGemma → Chroma)"
    )
    ap.add_argument("--source", default="data/raw", help="Folder containing .md files (recurses)")
    ap.add_argument("--persist", default="data/processed", help="Chroma persist dir")
    ap.add_argument("--collection", default="usf_onboarding", help="Chroma collection name")
    ap.add_argument("--chunk", type=int, default=900, help="Target chunk size (chars)")
    ap.add_argument("--overlap", type=int, default=150, help="Chunk overlap (chars)")
    ap.add_argument("--batch", type=int, default=64, help="Embedding batch size")
    ap.add_argument(
        "--skip_unchanged",
        action="store_true",
        default=False,
        help="Skip docs whose fingerprint already exists in the collection",
    )
    ap.add_argument(
        "--dry_run",
        action="store_true",
        default=False,
        help="Parse and report without writing to Chroma",
    )
    args = ap.parse_args()

    src = Path(args.source)
    if not src.exists():
        raise SystemExit(f"[!] Source not found: {src}")

    embedder = GemmaEmbedder(EMBED_MODEL)
    col = None if args.dry_run else get_chroma_collection(Path(args.persist), args.collection)

    scanned = added = skipped = 0

    for path in iter_md_files(src):
        scanned += 1
        try:
            raw = path.read_text(encoding="utf-8", errors="ignore")
            if not raw.strip():
                print(f"[skip empty] {path}")
                skipped += 1
                continue

            # clean & normalize
            text = group_faq_blocks(reflow_paragraphs(clean_text(raw)))
            if not text.strip():
                print(f"[skip empty] {path}")
                skipped += 1
                continue

            # document-level fingerprint to avoid re-ingesting unchanged files
            doc_fp = sha1(" ".join(text.split()))
            if args.skip_unchanged and not args.dry_run and collection_has_doc(col, doc_fp):
                print(f"[skip unchanged] {path}")
                skipped += 1
                continue

            base_id = f"{path.relative_to(src)}".replace("\\", "/")
            title = md_title(text, fallback=path.stem)

            # chunk → dedupe → small-glue
            raw_chunks = recursive_chunks(text, args.chunk, args.overlap)
            raw_chunks = glue_short_chunks(raw_chunks, min_chars=300)

            seen = set()
            chunks: List[str] = []
            for ch in raw_chunks:
                fp = fingerprint(ch)
                if fp in seen:
                    continue
                seen.add(fp)
                chunks.append(ch)
            if not chunks:
                print(f"[skip no-chunks] {path}")
                skipped += 1
                continue

            ids      = chunk_ids(base_id, chunks)
            titles   = [title] * len(chunks)
            category = derive_category(path.relative_to(src))
            canonical = first_usf_url(raw) or ""
            metas  = [{
                "source": base_id,
                "filename": path.name,
                "relpath": base_id,
                "section_title": title,
                "len": len(ch),
                "embedder": f"gemma::{EMBED_MODEL}",
                "source_format": "markdown",
                "doc_fp": doc_fp,            # for skip_unchanged
                "chunk_fp": fingerprint(ch), # useful for maintenance
                "category": category,
                "canonical": canonical,
            } for ch in chunks]

            if args.dry_run:
                print(f"[dry] {base_id} -> {len(chunks)} chunks")
                added += len(chunks)
                continue

            vecs = embedder.embed_docs(chunks, titles=titles, batch=args.batch)
            col.upsert(ids=ids, embeddings=vecs, documents=chunks, metadatas=metas)
            added += len(chunks)
            print(f"[ok] {path} → {len(chunks)} chunks")

        except Exception as e:
            print(f"[error] {path}: {e}")
            skipped += 1

    # -------------------- summary --------------------
    print("\n=== Ingestion Summary ===")
    print(f"Scanned files : {scanned}")
    print(f"Chunks added  : {added}")
    print(f"Skipped       : {skipped}")
    if not args.dry_run:
        print(f"Chroma path   : {Path(args.persist).resolve()}")
        print(f"Collection    : {args.collection}")
        print(f"Embedder      : gemma::{EMBED_MODEL}")
        print(f"Chunk/Overlap : {args.chunk}/{args.overlap}")

if __name__ == "__main__":
    main()
