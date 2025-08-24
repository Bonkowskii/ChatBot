# RAG Pipeline (PL)

Polskojęzyczny, lekki system **R**etrieval‑**A**ugmented **G**eneration (RAG). Łączy indeksowanie własnej bazy dokumentów (`./docs`) z lokalnym modelem językowym (domyślnie **Qwen/Qwen2.5‑1.5B‑Instruct**), oraz udostępnia API + prosty frontend webowy.

> **Dla kogo?** Dla osób, które chcą w minutę uruchomić mały RAG na własnych plikach `.txt`/`.md`, bez chmury.

---

## Spis treści

* [Wymagania](#wymagania)
* [Szybki start](#szybki-start)
* [Konfiguracja (.env)](#konfiguracja-env)
* [Struktura projektu](#struktura-projektu)
* [Uruchomienie z CLI](#uruchomienie-z-cli)
* [Uruchomienie w trybie webowym](#uruchomienie-w-trybie-webowym)
* [Jak to działa pod spodem](#jak-to-dziala-pod-spodem)
* [Najczęstsze problemy](#najczestsze-problemy)
* [Wskazówki jakości odpowiedzi](#wskazowki-jakosci-odpowiedzi)
* [FAQ](#faq)

---

## Wymagania

* **Python 3.10–3.12**
* System: Windows / Linux / macOS
* Połączenie z HuggingFace (pierwsze pobranie modeli)
* RAM: im więcej, tym lepiej (domyślnie okno kontekstu 4096 tokenów)

> **GPU (opcjonalnie):** możesz przyspieszyć inferencję ustawiając `DEVICE_MAP=auto` i instalując `accelerate` (szczegóły niżej).

---

## Szybki start

### 1) Klon i środowisko

**Windows (PowerShell):**

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -U pip
pip install "torch>=2.3" transformers sentence-transformers fastapi uvicorn numpy spacy python-dotenv
```

**Linux/macOS (bash):**

```bash
python -m venv .venv
source .venv/bin/activate
pip install -U pip
pip install "torch>=2.3" transformers sentence-transformers fastapi uvicorn numpy spacy python-dotenv
```

*(opcjonalnie lepsze dzielenie zdań)*

```bash
python -m spacy download pl_core_news_sm
```

### 2) Dodaj dane

Umieść swoje pliki `.txt`/`.md` w katalogu **`./docs/`** (np. `docs/pllum.txt`).

### 3) Pierwszy test (CLI)

```bash
python rag_pipeline.py ask "Kto stworzył PLLuM?"
```

Przy pierwszym uruchomieniu zbuduje się indeks i embeddingi oraz pobierze się model LLM.

### 4) Tryb webowy

```bash
uvicorn src.api:app --reload
```

* Wejdź na: `http://127.0.0.1:8000/` — prosty frontend.
* Dokumentacja API (Swagger): `http://127.0.0.1:8000/docs`.

---

## Konfiguracja (.env) (Opcjonalne)

Utwórz plik **`.env`** (opcjonalny) i nadpisz domyślne wartości:

```ini
# Model językowy (HuggingFace model id)
LLM=Qwen/Qwen2.5-1.5B-Instruct

# Retriever (SentenceTransformer)
EMBEDDER=intfloat/multilingual-e5-small

# Filtr podobieństwa (im wyżej, tym mniej kontekstu, ale mocniej trafionego)
MIN_SCORE=0.40

# Budżety tokenów
CTX_CAP_TOKENS=4096
MAX_NEW_TOKENS=512
MIN_ANS_TOKENS=192
PER_HIT_MIN_TOKENS=64
PER_HIT_HARD_CAP=0

# Urządzenie: cpu | cuda | auto  (auto wymaga 'accelerate')
DEVICE_MAP=cpu

# Opcjonalne twarde okno (wtedy ignoruje max z modelu)
# BUDGET_TOKENS=4096
```

> Zmiana `.env` działa bez modyfikacji kodu – wartości są czytane przy starcie.

---

## Struktura projektu


```
├─ src/
│  └─ rag_pipeline.py   # orchestrator RAG (CLI + core)
│  └─ chunking.py     # indeksowanie: dzielenie na zdania/chunki
│  └─ retrieval.py     # embeddingi + wyszukiwanie
│  └─ api.py          # FastAPI + mount statycznego frontu
├─ static/
│  └─ index.html            # prosty frontend (formularz)
├─ docs/                    # Twoje źródła .txt/.md
│  └─ pllum.txt             # (przykład)
└─ artifacts/               # auto: chunks.jsonl, embeddings.npy, meta.jsonl, itp.
```

---

## Uruchomienie z CLI

Zapytanie do bazy:

```bash
python rag_pipeline.py ask "Twoje pytanie"
```

Przykłady:

```bash
python rag_pipeline.py ask "Kto stworzył PLLuM?"
python rag_pipeline.py ask "Jakie modele PLLuM są dostępne?"
```

---

## Uruchomienie w trybie webowym

Start serwera (z autoreloadem w devie):

```bash
uvicorn src.api:app --reload
```

* Front: `http://127.0.0.1:8000/`
* API:

  * `GET /health` → `{ "ok": true }`
  * `POST /api/ask`  body: `{ "q": "pytanie" }`
  * `POST /api/reindex` → wymusza przebudowę indeksu i embeddingów
* Swagger: `http://127.0.0.1:8000/docs`

Jeśli zmienisz pliki w `./docs/`, kliknij w UI **„Przebuduj indeks”** lub wywołaj `POST /api/reindex`.

---

## Jak to działa pod spodem

1. **Chunking** (`chunking.py`) – spaCy dzieli tekst na zdania; łączy je w chunki (domyślnie 6 zdań, overlap 2). Wynik: `artifacts/chunks.jsonl` + `manifest.json`.
2. **Embeddings** (`retrieval.py`) – `intfloat/multilingual-e5-small` koduje chunki i pytanie; zapis do `artifacts/embeddings.npy` + `embeddings.info.json` + `meta.jsonl`.
3. **Retrieval** – wybieramy najlepsze fragmenty ≥ `MIN_SCORE` (domyślnie top‑k kontroluje pipeline), mieszczące się w oknie.
4. **Prompting** – budujemy wiadomości (system+user) + twardy znacznik `<final>…</final>`.
5. **LLM** (domyślnie Qwen 1.5B Instruct, CPU) – generuje z ograniczeniami (twardy stop na `</final>`).
6. **Fokus na pytania „kto…?”** – pipeline dodatkowo filtruje zdania z nazwami organizacji (większa precyzja przy pytaniach o autorów).

---

## Najczęstsze problemy

**1) Komunikat:** `Using a device_map ... requires accelerate`

* Ustawiłeś `DEVICE_MAP=auto` bez zainstalowanego `accelerate`.
* **Rozwiązania:**

  * Zmień w `.env` na `DEVICE_MAP=cpu`, **albo**
  * `pip install accelerate` (i opcjonalnie Torch z CUDA, jeśli masz GPU).

**2) „Brak danych w ./docs”**

* Włóż do `./docs` pliki `.txt`/`.md`. Następnie uruchom ponownie lub użyj `POST /api/reindex`.

**3) „Brak trafień ≥ …”**

* Zmniejsz `MIN_SCORE` (np. 0.30) **albo** dodaj więcej treści do `docs/`.

**4) „Zbyt małe okno kontekstu względem MAX\_NEW\_TOKENS”**

* Zmniejsz `MAX_NEW_TOKENS` lub zwiększ `CTX_CAP_TOKENS` (wymaga więcej RAM).

**5) Odpowiedzi są zbyt ogólne („lanie wody”)**

* To zwykle efekt słabych/trafień lub zbyt mało precyzyjnego kontekstu. Podnieś `MIN_SCORE`, dodaj lepsze dane do `docs/`, skorzystaj z wbudowanego filtra „kto…”.

**6) spaCy nie ma modelu PL**

* Zainstaluj: `python -m spacy download pl_core_news_sm`. Pipeline ma też fallback (sentencizer), ale model PL działa lepiej.

---

## Wskazówki jakości odpowiedzi

* **Lepszy kontekst > więcej kontekstu.** Nie dawaj wszystkiego – podnieś `MIN_SCORE`, trzymaj `docs/` czyste.
* **Limity tokenów**: jeśli masz dużo RAM, zwiększ `CTX_CAP_TOKENS`; jeśli nie – zostaw 4096.
* **Inny LLM**: możesz podmienić `LLM` na większy (np. `Qwen/Qwen2.5-7B-Instruct`) – ale będzie wolniej i wymaga więcej pamięci.
* **GPU**: ustaw `DEVICE_MAP=auto` i zainstaluj `accelerate`; upewnij się, że masz zgodny build PyTorch (CUDA/ROCm).

---

## FAQ

**Czy muszę mieć internet?**

* Tylko przy pierwszym pobraniu modeli. Później działasz lokalnie.

**Czy mogę dodać PDF/HTML?**

* Aktualnie parser oczekuje `.txt/.md`. Przekonwertuj wcześniej (np. `pandoc`, `pdftotext`).

**Gdzie są logi i artefakty?**

* W katalogu `artifacts/` (chunks, embeddings, meta). Konsola pokazuje postęp i dopasowania.

**Czy projekt wspiera Docker?**

* Na razie **nie** – celem było szybkie uruchomienie lokalne. Gdy zechcesz, łatwo dodać `Dockerfile` i volume na `docs/` + `artifacts/`.

---

## Notatki developerskie
* **Front statyczny**: plik `static/index.html` musi istnieć (w repo jest już gotowy przykładowy UI).

---

## Licencja

Kod przykładowy edukacyjny. Sprawdź licencje używanych modeli (Qwen, e5) bezpośrednio w ich repozytoriach na HuggingFace i dostosuj użycie do wymogów licencyjnych.
