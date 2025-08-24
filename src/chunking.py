import json, hashlib
from pathlib import Path
import spacy

DOC_EXTS = ("*.txt", "*.md")

def load_nlp():
    try:
        return spacy.load("pl_core_news_sm")
    except OSError:
        # fallback, gdy brak modelu spaCy PL
        nlp = spacy.blank("pl")
        nlp.add_pipe("sentencizer")
        return nlp

def split_sentences(nlp, text: str) -> list[str]:
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]

def make_chunks(sentences: list[str], target: int = 6, overlap: int = 2) -> list[str]:
    chunks = []
    if not sentences:
        return chunks
    step = max(1, target - overlap)
    i = 0
    while i < len(sentences):
        window = sentences[i: i + target]
        if not window:
            break
        chunks.append(" ".join(window))
        if len(window) < target:
            break
        i += step
    return chunks

def iter_doc_paths(docs_dir: Path):
    for ext in DOC_EXTS:
        for p in sorted(docs_dir.glob(ext)):
            yield p

def read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="ignore")

def write_jsonl(path: Path, records: list[dict]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def file_hash(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(65536), b""):
            h.update(b)
    return h.hexdigest()

def build_chunk_records(docs_dir: Path, target: int, overlap: int) -> tuple[list[dict], dict]:
    nlp = load_nlp()
    rows, manifest = [], {}
    for p in iter_doc_paths(docs_dir):
        txt = read_text(p)
        sents = split_sentences(nlp, txt)
        chs = make_chunks(sents, target=target, overlap=overlap)
        for cid, ch in enumerate(chs):
            rows.append({"source": str(p), "chunk_id": cid, "text": ch})
        manifest[str(p)] = {"hash": file_hash(p), "n_chunks": len(chs)}
    return rows, manifest

def build_and_save_chunks(docs_dir: Path, out_dir: Path, target: int = 6, overlap: int = 2):
    rows, manifest = build_chunk_records(docs_dir, target=target, overlap=overlap)
    if not rows:
        print("[INDEX] Uwaga: brak dokumentów w ./docs — dodaj .txt/.md i uruchom ponownie.")
    write_jsonl(out_dir / "chunks.jsonl", rows)
    (out_dir / "artifacts").mkdir(exist_ok=True, parents=True)  # dla spójności, gdy out_dir to ./artifacts
    with (out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"[INDEX] Files={len(manifest)} | Chunks={len(rows)}")
    print(f"[SAVE] {(out_dir / 'chunks.jsonl').as_posix()} | {(out_dir / 'manifest.json').as_posix()}")

def ensure_chunks(docs_dir: Path, out_dir: Path, target: int = 6, overlap: int = 2):
    chunks = out_dir / "chunks.jsonl"
    manifest_path = out_dir / "manifest.json"
    # Brak indeksu => pełna budowa
    if not chunks.exists() or not manifest_path.exists():
        print("[INDEX] Brak indeksu — buduję od zera…")
        build_and_save_chunks(docs_dir, out_dir, target=target, overlap=overlap)
        return

    try:
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        print("[INDEX] Uszkodzony manifest — pełna przebudowa…")
        build_and_save_chunks(docs_dir, out_dir, target=target, overlap=overlap)
        return

    current = {}
    for p in iter_doc_paths(docs_dir):
        current[str(p)] = {"hash": file_hash(p)}

    changed = (
        set(current.keys()) != set(saved.keys())
        or any(current[k]["hash"] != saved.get(k, {}).get("hash") for k in current.keys())
    )
    if changed:
        print("[INDEX] Wykryto zmiany w dokumentach — przebudowuję indeks…")
        build_and_save_chunks(docs_dir, out_dir, target=target, overlap=overlap)
    else:
        print("[INDEX] Chunks aktualne — pomijam.")
