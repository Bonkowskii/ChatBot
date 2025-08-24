# rag_pipeline.py  (modularny orchestrator)
import os
from pathlib import Path
import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig
import re
import unicodedata

# --- nasze moduły ---
from src.chunking import ensure_chunks
from src.retrieval import build_embeddings, query, load_jsonl, EMB_MODEL

# --- ścieżki ---
BASE_DIR = Path(__file__).parent.parent.resolve()
DOCS_DIR = BASE_DIR / "docs"
ART_DIR  = BASE_DIR / "artifacts"
CHUNKS   = ART_DIR / "chunks.jsonl"
META     = ART_DIR / "meta.jsonl"
EMB      = ART_DIR / "embeddings.npy"
EMB_INFO = ART_DIR / "embeddings.info.json"

# --- konfig (ENV -> domyślne) ---
EMBEDDER       = os.getenv("EMBEDDER", EMB_MODEL)
MIN_SCORE      = float(os.getenv("MIN_SCORE", "0.40"))
MODEL_NAME     = os.getenv("LLM", "Qwen/Qwen2.5-1.5B-Instruct")

BUDGET_TOKENS  = os.getenv("BUDGET_TOKENS")

MAX_NEW_TOKENS = int(os.getenv("MAX_NEW_TOKENS", "512"))
MIN_ANS_TOKENS = int(os.getenv("MIN_ANS_TOKENS", "192"))

PER_HIT_MIN       = int(os.getenv("PER_HIT_MIN_TOKENS", "80"))

PER_HIT_HARD_CAP  = int(os.getenv("PER_HIT_HARD_CAP", "0"))

# CPU-safe cap na całe okno (można podnieść na 8192, posiadając dużo RAM/GPU)
CTX_CAP_TOKENS = int(os.getenv("CTX_CAP_TOKENS", "4096"))

DEVICE_PREF = os.getenv("DEVICE_MAP", "auto")  # 'auto'|'cpu'|'cuda'

INSTR_TOKENS = int(os.getenv("INSTR_TOKENS", "60"))

def get_context_window(model_name: str, fallback: int = 4096) -> int:
    try:
        cfg = AutoConfig.from_pretrained(model_name)
        max_ctx = int(getattr(cfg, "max_position_embeddings", fallback))
        if not max_ctx or max_ctx > 10**7:
            max_ctx = fallback
    except Exception:
        max_ctx = fallback
    return min(max_ctx, CTX_CAP_TOKENS)

def toklen(tokenizer, text: str, add_special_tokens: bool = False) -> int:
    return len(tokenizer(text, add_special_tokens=add_special_tokens)["input_ids"])

SEP_TEXT = "\n\n---\n"

def trim_text_to_tokens(tokenizer, text: str, max_tokens: int) -> str:
    ids = tokenizer(text)["input_ids"]
    if len(ids) <= max_tokens:
        return text
    cut = tokenizer.decode(ids[:max_tokens], skip_special_tokens=True)
    for mark in (". ", ".\n", "… ", "!\n", "?\n", "!\t", "?\t"):
        i = cut.rfind(mark)
        if i >= 40:
            return cut[:i+1].rstrip()
    return cut.rstrip()

# --- HINTY pod pytania "kto" ---
WHO_HINT_RE = re.compile(r"\b(kto|kto stworzył|twórc[ay]|autor[zy]?)\b", re.I)
ORG_HINT_RE = re.compile(
    r"(Politechnik\w+|Uniwersyt\w+|Instytut\w+|PAN\b|PIB\b|NASK\b|Ministerstw\w+|"
    r"Polska Akademia Nauk|Ośrodek Przetwarzania Informacji)", re.I
)

def focus_hits(question: str, hits: list[dict]) -> list[dict]:
    if not WHO_HINT_RE.search(question):
        return hits
    focused = []
    for h in hits:
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", h["text"])
        keep = [s.strip() for s in parts
                if ORG_HINT_RE.search(s) or
                   "konsorcj" in s.lower() or
                   "koordynowan" in s.lower() or
                   "wspieran" in s.lower()]
        if keep:
            focused.append({**h, "text": " ".join(keep)})
    return focused or hits

# --- HINTY pod pytania o VRAM/GB ---
VRAM_Q_RE = re.compile(r"\b(\d+\s*GB|VRAM|GPU|fp16|fp8|int4|kwantyz\w+)\b", re.I)

def focus_vram_hits(question: str, hits: list[dict]) -> list[dict]:
    q = question.lower()
    if not (("gb" in q) or ("vram" in q) or re.search(r"\b\d+\s*gb\b", q)):
        return hits
    focused = []
    for h in hits:
        parts = re.split(r"(?<=[\.\!\?])\s+|\n+", h["text"])
        keep = [s.strip() for s in parts if VRAM_Q_RE.search(s)]
        if keep:
            focused.append({**h, "text": " ".join(keep)})
    return focused or hits

def is_generic_polish(text: str) -> bool:
    t = text.lower()
    needles = [
        "powstał w oparciu o etycznie pozyskane dane",
        "modele naukowe („nc”)",
        "aligned on human preferences",
    ]
    return any(x in t for x in needles)

def estimate_prompt_overhead_tokens(tokenizer, question: str) -> int:
    empty_messages = build_messages(question, [], is_who_mode=False, is_vram_mode=False)

    empty_prompt = render_prompt(tokenizer, empty_messages)
    return toklen(tokenizer, empty_prompt, add_special_tokens=True)

def allocate_hits(tokenizer, question: str, hits: list[dict],
                  context_window: int, max_new_tokens: int,
                  per_hit_min: int = 80, per_hit_cap: int | None = None):
    sep_cost = toklen(tokenizer, SEP_TEXT)
    overhead = estimate_prompt_overhead_tokens(tokenizer, question)

    def try_pack(desired_new_tokens: int):
        budget_for_ctx = context_window - overhead - desired_new_tokens
        if budget_for_ctx <= per_hit_min:
            desired_new_tokens = max(MIN_ANS_TOKENS, context_window - overhead - per_hit_min)
            budget_for_ctx = max(0, context_window - overhead - desired_new_tokens)

        kept, used = [], 0
        for h in hits:
            t = toklen(tokenizer, h["text"])
            if per_hit_cap and per_hit_cap > 0 and t > per_hit_cap:
                h_text = trim_text_to_tokens(tokenizer, h["text"], per_hit_cap)
                t = toklen(tokenizer, h_text)
            else:
                h_text = h["text"]

            extra_sep = sep_cost if kept else 0

            if used + extra_sep + t <= budget_for_ctx:
                kept.append({**h, "text": h_text})
                used += extra_sep + toklen(tokenizer, h_text)
                continue

            remaining = budget_for_ctx - used - extra_sep
            if remaining >= per_hit_min:
                trimmed = trim_text_to_tokens(tokenizer, h_text, remaining)
                if trimmed.strip():
                    kept.append({**h, "text": trimmed})
                    used += extra_sep + toklen(tokenizer, trimmed)
            break

        return kept, used, desired_new_tokens, {
            "overhead": overhead,
            "budget_for_ctx": budget_for_ctx,
            "sep_cost": sep_cost,
        }

    kept, used, eff_new, dbg = try_pack(MAX_NEW_TOKENS)
    if not kept:
        kept, used, eff_new, dbg = try_pack(max(MIN_ANS_TOKENS, MAX_NEW_TOKENS // 2))

    return kept, used, eff_new, dbg

def ensure_all(target_sentences: int = 6, overlap: int = 2):
    ART_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)

    ensure_chunks(DOCS_DIR, ART_DIR, target=target_sentences, overlap=overlap)
    if not CHUNKS.exists() or CHUNKS.stat().st_size == 0:
        raise RuntimeError("Brak danych w ./docs — dodaj .txt/.md i uruchom ponownie.")

    need_build = False
    if not EMB.exists() or not META.exists() or not EMB_INFO.exists():
        need_build = True
    else:
        try:
            X = np.load(EMB)
            meta_rows = load_jsonl(META)
            if X.shape[0] != len(meta_rows):
                need_build = True
        except Exception:
            need_build = True

        if not need_build and CHUNKS.stat().st_mtime > EMB.stat().st_mtime:
            need_build = True

    if need_build:
        build_embeddings(CHUNKS, EMB, META, embedder_name=EMBEDDER)

    X = np.load(EMB)
    return {"vectors": int(X.shape[0]), "dim": int(X.shape[1])}

def limit_tokens_per_hit(hits, tokenizer, max_tokens_per_hit: int = 256):
    trimmed = []
    for h in hits:
        ids = tokenizer(h["text"])["input_ids"]
        if len(ids) > max_tokens_per_hit:
            h2 = dict(h)
            h2["text"] = tokenizer.decode(ids[:max_tokens_per_hit], skip_special_tokens=True)
            trimmed.append(h2)
        else:
            trimmed.append(h)
    return trimmed
# --- LISTA MODELI LLaMA ---
LIST_LLAMA_RE = re.compile(
    r"\b(jakie|wymie[nń]|wypisz|podaj)\b.*\b(modele|warianty|rozmiary|wersje)\b.*\b(llama)\b",
    re.I
)

def _is_llama_list(question: str) -> bool:
    return bool(LIST_LLAMA_RE.search(question))
def build_messages(
    question: str,
    hits: list[dict],
    *,
    is_who_mode: bool = False,
    is_vram_mode: bool = False,
    is_llama_list_mode: bool = False,   # <-- DODANE
) -> list[dict]:
    context = "\n\n---\n".join(h["text"] for h in hits)
    system = (
        "Jesteś asystentem QA. Odpowiadasz WYŁĄCZNIE na podstawie KONTEKSTU.\n"
        "Zasady (twarde):\n"
        "1) Po polsku, zwięźle.\n"
        "2) Jeżeli odpowiedź NIE wynika wprost z KONTEKSTU, masz OBOWIĄZEK napisać dokładnie: 'Brak danych w kontekście.' i NIC WIĘCEJ.\n"
        "3) Absolutnie nie wolno Ci zgadywać ani korzystać z wiedzy spoza KONTEKSTU.\n"
        "4) Gdy pytanie dotyczy listy modeli — wypunktuj i pogrupuj logicznie."
    )

    extra = ""
    if is_vram_mode:
        extra += (
            "\nNa podstawie liczb w KONTEKŚCIE wybierz JEDEN model, którego wymagania są "
            "**mniejsze lub równe** podanej pamięci VRAM. Jeśli kilka pasuje, wskaż najbliższy górny limit. "
            "Podaj nazwę/rozmiar i liczby GB z KONTEKSTU w nawiasie. "
            "Nie wybieraj modeli, które wymagają więcej VRAM niż podano **ani wariantów MoE (np. Maverick)**."
        )

    if is_who_mode:
        extra += (
            "\nODPOWIEDZ jednym krótkim zdaniem: wypisz WYŁĄCZNIE nazwy instytucji/organizacji "
            "tworzących projekt, oddzielone przecinkami. Bez wstępów ani komentarzy."
        )
    if is_llama_list_mode:
        extra += (
            "\nZbierz WSZYSTKIE warianty modeli LLaMA wymienione w KONTEKŚCIE i wypisz je wypunktowane, "
            "POGRUPOWANE wg generacji (np. 'LLaMA 1', 'LLaMA 2', 'LLaMA 3', 'LLaMA 3.1', 'LLaMA 3.2', 'LLaMA 4'). "
            "Dla każdej generacji podaj rozmiary/warianty (np. 7B, 13B, 65B; 8B, 70B; 405B; itd.). "
            "Jeżeli w KONTEKŚCIE jest 'Code Llama' – pokaż osobną sekcję. "
            "Nie dopisuj nic spoza KONTEKSTU. Jeśli w KONTEKŚCIE nie ma informacji dla danej generacji, pomiń tę sekcję."
        )

    user = (
        f"KONTEKST:\n{context}\n\n"
        f"Pytanie: {question}{extra}\n"
        "Zwróć odpowiedź WYŁĄCZNIE w tagach <final>…</final>.\n"
        "<final>"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]



def render_prompt(tokenizer, messages: list[dict]) -> str:
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

def _postprocess(text: str) -> str:
    lines = []
    for ln in text.splitlines():
        s = ln.rstrip()
        low = s.lower().strip()
        if not s:
            lines.append("")
            continue
        if low.startswith(("kontekst:", "pytanie:", "###", "odpowiedź", "odpowiedz")):
            continue
        if len(s) >= 2 and s[1] == "." and s[0].isalpha():
            continue
        lines.append(s)

    out = "\n".join(lines)
    out = re.sub(r"\n{3,}", "\n\n", out).strip()
    out = re.sub(r"\bModely\b", "Modele", out, flags=re.I)
    out = re.sub(r"(?i)^final[:\s]+", "", out)

    return out

# --- utils do normalizacji/porównania nazw instytucji ---
_PL_MAP = str.maketrans(
    "ąćęłńóśżźĄĆĘŁŃÓŚŻŹ",
    "acelnoszzACELNOSZZ"
)
_STOP = {
    "kto","jest","sa","są","byl","była","byli","czy","co","jaki","jakie","jaką","jak",
    "polska","polski","polskiej","polsce","prezydent","prezydentem","ministra","ministrem",
    "i","oraz","albo","lub","dla","nad","pod","przy","na","z","ze","do","o","od","u"
}

def _question_has_vram(q: str) -> bool:
    ql = q.lower()
    return bool(re.search(r"\b\d+\s*gb\b|\bvram\b|\bfp(16|8)\b|\bint4\b|\b8[- ]?bit|\b4[- ]?bit", ql))

def _norm(s: str) -> str:
    s = s.translate(_PL_MAP)
    s = unicodedata.normalize("NFKD", s)
    s = s.encode("ascii", "ignore").decode("ascii")
    s = s.lower()
    s = re.sub(r"[^a-z0-9 ]+", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

def _ctx_text(hits: list[dict]) -> str:
    return " ".join(h.get("text","") for h in hits)

def _mentioned_in_ctx_fuzzy(ans: str, hits: list[dict]) -> bool:
    if not ans.strip():
        return False
    toks = [t.strip() for t in re.split(r",|;| i ", ans) if t.strip()]
    if not toks:
        return False
    nctx = _norm(_ctx_text(hits))
    for t in toks:
        nt = _norm(t)
        if not nt:
            continue
        if nt in nctx:
            return True
    return False

def _extract_orgs_from_hits(hits: list[dict]) -> str:
    txt = _ctx_text(hits)
    rx = re.compile(
        r"\b("
        r"Politechnik\w+(?:\s\w+){0,3}|"
        r"Uniwersyt\w+(?:\s\w+){0,3}|"
        r"Instytut\w+(?:\s\w+){0,4}|"
        r"Polska Akademia Nauk|"
        r"O[śs]rodek Przetwarzania Informacji(?:\sPIB)?|"
        r"NASK(?:\sPIB)?|"
        r"OPI(?:\sPIB)?|"
        r"Instytut Slawistyki PAN|"
        r"PAN|PIB|U[ŁL]\b|Ministerstw\w+(?:\s\w+){0,3}"
        r")\b",
        re.I
    )
    found = rx.findall(txt)
    cleaned, seen = [], set()
    for f in found:
        f2 = re.sub(r"\s+", " ", f).strip()
        nf = _norm(f2)
        if nf and nf not in seen:
            seen.add(nf)
            cleaned.append(f2)
    return ", ".join(cleaned[:12])
def _question_matches_context(question: str, hits: list[dict]) -> bool:
    ctx = _ctx_text(hits).lower()
    toks = [w.lower() for w in re.findall(r"\w+", question, flags=re.I) if len(w) >= 4]
    toks = [w for w in toks if w not in _STOP]
    if not toks:
        return False
    return any(w in ctx for w in toks)

def _is_who_mode(question: str, hits: list[dict]) -> bool:
    # Jeśli pytanie to "kto jest ...", wyłącz tryb WHO
    if re.search(r"\bkto\s+jest\b", question, flags=re.I):
        return False
    # Inne formy 'kto stworzył/autor/twórcy' – włącz tylko, jeśli mamy dopasowanie do kontekstu
    if WHO_HINT_RE.search(question):
        return _question_matches_context(question, hits)
    return False


def normalize_question_variants(q: str) -> list[str]:
    variants = set()
    variants.add(q.strip())
    v = re.sub(r"(?i)\bllama\b", "LLaMA", q)
    v = re.sub(r"(?i)\bgpt[-\s]?oss\b", "GPT-OSS", v)
    v = re.sub(r"(?i)\bpllum\b", "PLLuM", v)
    variants.add(v.strip())
    variants.add(re.sub(r"\s+", " ", v).strip())
    return [x for x in variants if x]

def answer(question: str):
    stats = ensure_all()

    hits = []
    tried = []
    for q_try in normalize_question_variants(question):
        for ms in (MIN_SCORE, 0.50, 0.45, 0.40, 0.35, 0.30, 0.25):
            tried.append((q_try, ms))
            hits = query(
                emb_path=EMB,
                meta_path=META,
                question=q_try,
                embedder_name=None,
                top_k=12,
                min_score=ms,
            )
            if hits:
                break
        if hits:
            break

    if not hits:
        return {"answer": "", "hits": [], "message": f"Brak trafień (próby={tried})."}

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    context_window = int(BUDGET_TOKENS) if BUDGET_TOKENS else get_context_window(MODEL_NAME, fallback=4096)

    kept, used_ctx_tokens, eff_max_new, dbg = allocate_hits(
        tokenizer=tokenizer,
        question=question,
        hits=hits,
        context_window=context_window,
        max_new_tokens=MAX_NEW_TOKENS,
        per_hit_min=PER_HIT_MIN,
        per_hit_cap=(PER_HIT_HARD_CAP if PER_HIT_HARD_CAP > 0 else None),
    )
    if not kept:
        return {"answer": "", "hits": [], "message": "Niestety nie udało się zmieścić kontekstu – spróbuj krótszego pytania lub zwiększ BUDGET_TOKENS."}

    # 2.5) ustal tryby i dopiero wtedy ewentualnie zawężaj
    is_who = _is_who_mode(question, kept)
    is_vram = _question_has_vram(question) and _question_matches_context(question, kept)
    is_llama_list = _is_llama_list(question)

    if is_who:
        kept = focus_hits(question, kept)
    if is_vram:
        kept = focus_vram_hits(question, kept)

    messages = build_messages(
        question, kept,
        is_who_mode=is_who,
        is_vram_mode=is_vram,
        is_llama_list_mode=is_llama_list,
    )
    prompt = render_prompt(tokenizer, messages)

    prompt_tokens = toklen(tokenizer, prompt, add_special_tokens=True)
    max_input_len = context_window - eff_max_new

    debug_budget = {
        "context_window": context_window,
        "prompt_tokens": prompt_tokens,
        "used_ctx_tokens": used_ctx_tokens,
        "effective_max_new_tokens": eff_max_new,
        **dbg
    }

    if max_input_len <= 0:
        return {
            "answer": "",
            "hits": kept,
            "message": "Zbyt małe okno kontekstu względem MAX_NEW_TOKENS.",
            "debug": {"budget": debug_budget},
        }

    if prompt_tokens > max_input_len:
        overshoot = prompt_tokens - max_input_len
        possible_shrink = max(0, eff_max_new - MIN_ANS_TOKENS)
        shrink = min(overshoot, possible_shrink)
        eff_max_new -= shrink
        max_input_len = context_window - eff_max_new

    if prompt_tokens > max_input_len:
        kept, used_ctx_tokens, eff_max_new, dbg = allocate_hits(
            tokenizer=tokenizer,
            question=question,
            hits=hits,
            context_window=context_window,
            max_new_tokens=eff_max_new,
            per_hit_min=PER_HIT_MIN,
            per_hit_cap=(PER_HIT_HARD_CAP if PER_HIT_HARD_CAP > 0 else None),
        )
        if not kept:
            return {"answer": "", "hits": [], "message": "Nie udało się zmieścić kontekstu po repacku."}

        if is_who:
            kept = focus_hits(question, kept)
        if is_vram:
            kept = focus_vram_hits(question, kept)

        messages = build_messages(
            question, kept,
            is_who_mode=is_who,
            is_vram_mode=is_vram,
            is_llama_list_mode=is_llama_list,
        )
        prompt = render_prompt(tokenizer, messages)

        prompt_tokens = toklen(tokenizer, prompt, add_special_tokens=True)
        max_input_len = context_window - eff_max_new

        debug_budget = {
            "context_window": context_window,
            "prompt_tokens": prompt_tokens,
            "used_ctx_tokens": used_ctx_tokens,
            "effective_max_new_tokens": eff_max_new,
            **dbg
        }

        if prompt_tokens > max_input_len:
            return {"answer": "", "hits": kept, "message": "Prompt > okno kontekstu nawet po repacku.", "debug": {"budget": debug_budget}}

    # 6) tokenizacja wejścia (potrzebna nam długość promptu do stoppera)
    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=max_input_len
    )

    # 7) model
    try:
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float32,
            device_map=(DEVICE_PREF if DEVICE_PREF in ("auto", "cuda") else None),
        )
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32)

    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    # 8) twardy stop na </final> — ALE dopiero po min. N tokenach
    from transformers import StoppingCriteria, StoppingCriteriaList
    class StopOnTextAfter(StoppingCriteria):
        def __init__(self, tokenizer, stop_str: str, prompt_len: int, min_gen_tokens: int = 12):
            self.stop_ids = tokenizer(stop_str, add_special_tokens=False).input_ids
            self.prompt_len = prompt_len
            self.min_gen_tokens = min_gen_tokens
        def __call__(self, input_ids, scores, **kwargs):
            cur = input_ids[0].tolist()
            gen_len = len(cur) - self.prompt_len
            if gen_len < self.min_gen_tokens:
                return False
            if len(cur) < len(self.stop_ids):
                return False
            tail = cur[-len(self.stop_ids):]
            return tail == self.stop_ids

    stoppers = StoppingCriteriaList([StopOnTextAfter(tokenizer, "</final>", inputs["input_ids"].shape[1], 12)])

    # 9) generacja
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=eff_max_new,
            do_sample=False,
            repetition_penalty=1.05,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
            stopping_criteria=stoppers,
        )

    gen_only_ids = out[0][inputs["input_ids"].shape[1]:]
    raw = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
    m = re.search(r"<final>(.*?)</final>", raw, flags=re.S)
    if m:
        raw = m.group(1).strip()
    final = _postprocess(raw)
    # mini‑guard na VRAM: odrzuć duże / MoE, wymuś 13B/8B jeśli trzeba
    if is_vram and re.search(r"\b(405\s*B|65\s*B|Maverick)\b", final, re.I):
        ctx = " ".join(h.get("text", "") for h in kept)
        pick = None
        for sz in ("13 B", "13B", "12 B", "12B", "10 B", "10B", "8 B", "8B", "7 B", "7B"):
            if sz in ctx:
                pick = sz.replace(" ", "")
                break
        final = f"Model {pick or '13B'} — pasuje do 24 GB (wg liczb w kontekście)."

    # 10) retry dla 'kto...' jeżeli ogólnik
    if is_who and is_generic_polish(final):
        retry_messages = build_messages(
            question + " (TYLKO NAZWY INSTYTUCJI, PRZECINKI.)",
            kept,
            is_who_mode=True,
            is_vram_mode=is_vram,
            is_llama_list_mode=is_llama_list
        )

        retry_prompt = render_prompt(tokenizer, retry_messages)
        new_eff = max(MIN_ANS_TOKENS, eff_max_new // 2)
        new_max_input_len = context_window - new_eff
        retry_inputs = tokenizer(retry_prompt, return_tensors="pt", truncation=True, max_length=new_max_input_len)
        with torch.no_grad():
            out = model.generate(
                **retry_inputs,
                max_new_tokens=new_eff,
                do_sample=False,
                repetition_penalty=1.05,
                eos_token_id=tokenizer.eos_token_id,
                pad_token_id=tokenizer.pad_token_id,
                stopping_criteria=StoppingCriteriaList([StopOnTextAfter(tokenizer, "</final>", retry_inputs["input_ids"].shape[1], 8)]),
            )
        gen_only_ids = out[0][retry_inputs["input_ids"].shape[1]:]
        raw2 = tokenizer.decode(gen_only_ids, skip_special_tokens=True)
        m2 = re.search(r"<final>(.*?)</final>", raw2, flags=re.S)
        if m2:
            final = _postprocess(m2.group(1).strip())

    # 11) awaryjne fallbacki gdy final jest pusty
    if not final.strip():
        # 11) walidacja tylko gdy realnie jesteśmy w trybie 'kto...'
        if is_who:
            if not final.strip():
                fallback = _extract_orgs_from_hits(kept)
                final = fallback if fallback else "Brak danych w kontekście."
            else:
                if not _mentioned_in_ctx_fuzzy(final, kept):
                    fallback = _extract_orgs_from_hits(kept)
                    final = fallback if fallback else "Brak danych w kontekście."

        else:

            best = kept[0]["text"].strip()
            final = (best[:420] + "…") if len(best) > 420 else best

    # 12) walidacja tylko, jeśli realnie włączyliśmy tryb WHO
    if is_who:
        if not _mentioned_in_ctx_fuzzy(final, kept):
            fallback = _extract_orgs_from_hits(kept)
            final = fallback if fallback else "Brak danych w kontekście."

    return {
        "answer": final,
        "hits": kept,
        "message": "",
        "debug": {
            "stats": stats,
            "budget": debug_budget,
        }
    }

# --- CLI ---
def ask_cli(q: str):
    res = answer(q)
    if res["message"]:
        print(res["message"])
    else:
        print("\n[LLM]\n" + res["answer"])

if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3 and sys.argv[1] == "ask":
        ask_cli(" ".join(sys.argv[2:]))
    else:
        print('Użycie: python rag_pipeline.py ask "PYTANIE"')
