# retrieval.py
import json
from pathlib import Path
import numpy as np
from sentence_transformers import SentenceTransformer

def load_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

EMB_MODEL = "intfloat/multilingual-e5-small"

def build_embeddings(chunks_path: Path, emb_path: Path, meta_path: Path, embedder_name: str = EMB_MODEL):
    chunks = load_jsonl(chunks_path)
    if not chunks:
        raise RuntimeError(f"Brak danych w {chunks_path}")
    texts = [row["text"] for row in chunks]
    print(f"[BUILD] chunks={len(texts)} | model={embedder_name}")
    model = SentenceTransformer(embedder_name)
    print("[BUILD] model loaded")

    texts_pref = [f"passage: {t}" for t in texts]  # E5 prefix
    X = model.encode(texts_pref, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True)
    print(f"[BUILD] encoded -> {X.shape}")

    emb_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(emb_path, X)
    print(f"[SAVE] {emb_path}")

    with meta_path.open("w", encoding="utf-8") as f:
        for row in chunks:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[SAVE] {meta_path}")

    info_path = emb_path.parent / "embeddings.info.json"
    with info_path.open("w", encoding="utf-8") as f:
        json.dump({"embedder_name": embedder_name, "dim": int(X.shape[1])}, f, ensure_ascii=False)
    print(f"[SAVE] {info_path}")

def query(emb_path: Path, meta_path: Path, question: str,
          embedder_name: str | None = None, top_k=None, min_score: float = 0.3):
    X = np.load(emb_path)
    meta = load_jsonl(meta_path)
    if X.shape[0] != len(meta):
        raise RuntimeError(f"Niezgodnosc: vectors={X.shape[0]} vs meta={len(meta)}")
    print(f"[QUERY] vectors={X.shape[0]} dim={X.shape[1]}")

    info_path = emb_path.parent / "embeddings.info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text(encoding="utf-8"))
        embedder_name = embedder_name or info.get("embedder_name", EMB_MODEL)
    else:
        embedder_name = embedder_name or EMB_MODEL

    model = SentenceTransformer(embedder_name)
    qv = model.encode([f"query: {question}"], convert_to_numpy=True, normalize_embeddings=True)  # E5 prefix
    sims = (qv @ X.T).ravel()  # cosine przy znormalizowanych

    candidate_idxs = np.argsort(-sims)
    idxs = [i for i in candidate_idxs if sims[i] >= min_score]
    if top_k:
        idxs = idxs[:top_k]

    if not idxs:
        print(f"[QUERY] Brak wyników ≥ {min_score:.2f}")
        return []

    results = []
    for rank, i in enumerate(idxs, start=1):
        m = meta[i]
        print("=" * 80)
        print(f"[{rank}] score={sims[i]:.4f}  {m['source']}  (chunk {m['chunk_id']})")
        print(m["text"])
        results.append({"rank": rank, "score": float(sims[i]), "source": m["source"], "chunk_id": m["chunk_id"], "text": m["text"]})
    return results
