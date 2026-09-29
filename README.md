# Ask a YouTube Video

Ask questions about any YouTube video and get answers grounded in its transcript —
with every claim cited back to the exact second it was said.

```
You  ▸ What does he say about hidden layers?
Bot  ▸ Hidden layers are where the network detects intermediate patterns such as
       edges and loops [2]. Their design is largely arbitrary — the layer sizes
       here were picked to look nice rather than derived [3].

       Sources
       [2] 5:12–5:41  ▸ jumps to 5:12 in the video
       [3] 9:03–9:35  ▸ jumps to 9:03 in the video
```

Retrieval-augmented generation (RAG) over YouTube transcripts: fetch the
captions, split them into timestamped chunks, embed them, retrieve only the
passages that matter for a question, and answer from those.

---

## Why cite timestamps

Most transcript chatbots flatten the captions into one long string before
chunking. That is the obvious thing to do, and it throws away the only
information that makes an answer checkable — *when* something was said.

This project keeps every caption cue attached to its start time all the way
through chunking, so each retrieved passage knows the span of video it covers. A
citation is therefore a real link into the video, and you can verify any claim in
one click instead of trusting the model.

---

## Features

| | |
|---|---|
| **Timestamped citations** | Every claim is marked `[n]` and links to that second of the video |
| **Follow-up questions** | "What about the second one?" is rewritten into a standalone query before retrieval, so pronouns resolve |
| **Pluggable answer models** | Claude by default; Hugging Face for open-weights models. Adding a provider is one class |
| **Persistent index cache** | Re-opening a video you have already processed is instant — no re-embedding |
| **Resilient transcripts** | Two independent caption sources, language fallback, and YouTube-side translation |
| **Diversity-aware retrieval** | Maximal marginal relevance, so overlapping chunks don't fill every slot |
| **Multilingual** | Multilingual embeddings; answers come back in the language you asked in |
| **Tested** | 101 tests, none of which touch the network |

---

## How it works

```
                YouTube URL
                     │
                     ▼
   ┌─────────────────────────────────────┐
   │ Transcript fetch (fallback chain)   │   youtube-transcript-api → yt-dlp
   │ → Segment(text, start, duration)    │   timestamps preserved from here on
   └─────────────────┬───────────────────┘
                     ▼
   ┌─────────────────────────────────────┐
   │ Chunking over segments, not text    │   ~1000 chars, 200 overlap
   │ → Chunk(text, start, end)           │   each chunk knows its video span
   └─────────────────┬───────────────────┘
                     ▼
   ┌─────────────────────────────────────┐
   │ Local embeddings + numpy index      │   cached on disk by content hash
   └─────────────────┬───────────────────┘
                     │
   Question ─────────┤
        │            ▼
        │   ┌─────────────────────────────┐
        └──▸│ Rewrite follow-up → search  │   MMR, top-k excerpts
            │ → numbered, timestamped     │
            └─────────────┬───────────────┘
                          ▼
            ┌─────────────────────────────┐
            │ Answer, citing [n] markers  │
            └─────────────┬───────────────┘
                          ▼
              Answer + clickable sources
```

---

## Quick start

Requires **Python 3.10–3.13**. (Not 3.14 yet — PyTorch has no wheels for it.)

```bash
git clone https://github.com/RahulB-HC/ask-youtube.git
cd ask-youtube

python3.12 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

cp .env.example .env               # then add your API key
streamlit run app.py
```

Put one key in `.env`:

```env
ANTHROPIC_API_KEY=sk-ant-...
```

That is the only required setting. Everything else in `.env.example` is optional.

The first run downloads the embedding model (a few hundred MB) once; later runs
load it from the local cache.

---

## Configuration

Every setting is read from the environment first, then from Streamlit secrets, so
the same code runs locally and on Streamlit Cloud.

| Variable | Default | Purpose |
|---|---|---|
| `YTRAG_LLM_PROVIDER` | `anthropic` | `anthropic` or `huggingface` |
| `ANTHROPIC_API_KEY` | — | Required for the default provider |
| `YTRAG_ANTHROPIC_MODEL` | `claude-opus-5-5` | Answer model |
| `HF_TOKEN` | — | Required for `huggingface` |
| `YTRAG_HF_MODEL` | `meta-llama/Llama-3.1-8B-Instruct` | Hugging Face model |
| `YTRAG_EMBED_MODEL` | `paraphrase-multilingual-MiniLM-L12-v2` | Local embedding model |
| `YTRAG_CACHE_DIR` | `.cache/ytrag` | Where built indexes are stored |
| `YTRAG_LANGUAGES` | `en` | Preferred caption languages, most preferred first |
| `YTRAG_TOP_K` | `5` | Excerpts retrieved per question |
| `YTRAG_CHUNK_CHARS` | `1000` | Target chunk size |
| `YTRAG_CHUNK_OVERLAP_CHARS` | `200` | Overlap between chunks |
| `YTRAG_CONDENSE_QUESTION` | `true` | Rewrite follow-ups before retrieving |
| `YTRAG_YOUTUBE_PROXY` | — | Proxy for transcript fetches (see below) |

---

## Project layout

```
app.py                      Streamlit UI — widgets and rendering only
src/ytrag/
├── config.py               settings from env / Streamlit secrets
├── errors.py               one exception type per user-fixable failure
├── video.py                URL and ID parsing, timestamp formatting
├── transcript/
│   ├── models.py           Segment and Transcript
│   ├── subtitles.py        WebVTT and json3 parsers
│   ├── youtube_api.py      primary caption source
│   ├── ytdlp.py            fallback caption source
│   └── fetcher.py          the fallback chain
├── chunking.py             timestamp-preserving chunker
├── embeddings.py           Embedder protocol + local sentence-transformers
├── index.py                numpy vector index, MMR search, disk cache
├── llm/                    provider protocol + Claude / HF / fake
├── prompts.py              system prompts and context formatting
├── engine.py               rewrite → retrieve → answer → resolve citations
└── render.py               answer text → markdown with clickable citations
tests/                      101 tests, fully offline
```

---

## Development

```bash
pip install -r requirements-dev.txt
pytest                       # 101 tests, no network, ~1s
ruff check .
```

The test suite is deliberately offline. A stubbed embedder gives exact,
reproducible retrieval assertions, and a scripted provider stands in for the
model, so the RAG pipeline is tested end to end without spending a cent or
depending on YouTube being reachable. The Streamlit layer is covered through
Streamlit's own `AppTest` harness, which runs `app.py` exactly as the server
does.

---

## Design notes

**Why brute-force search instead of FAISS.** One video is a few hundred chunks.
An exact numpy dot product over that is measured in microseconds, so an
approximate index would add a heavyweight native dependency to save nothing. If
this ever grew to search across many videos at once, that trade would flip.

**Why embeddings run locally.** Embedding is per-chunk, so a hosted embedding API
means hundreds of round trips per video and a second key to manage. Running the
model locally costs one download and then nothing, and keeps the whole app
working with a single API key.

**Why effort is set to `low`.** Answering from supplied excerpts is reading
comprehension, not reasoning. Higher effort spends tokens and latency without
making the answers better.

**Why follow-up rewriting is a separate step.** Embedding "what about the second
one?" retrieves nothing useful, because the question carries no content. Rewriting
it against the conversation first is what makes multi-turn chat work at all. If
the rewrite fails, the original question is used — a slightly worse search beats
no answer.

---

## Limitations

- **Needs captions.** If a video has captions disabled, there is nothing to index.
  Audio transcription is not implemented.
- **YouTube blocks datacentre IPs.** This works fine locally but often fails on
  cloud hosts, including Streamlit Cloud, with "no transcript available". Set
  `YTRAG_YOUTUBE_PROXY` to a residential proxy to work around it. This is a
  YouTube policy, not a bug here.
- **Answers are only as good as the captions.** Auto-generated transcripts
  misheard words stay misheard.
- **One video at a time.** Cross-video search is not implemented.
- **Memory on Streamlit Cloud's free tier is tight** for the local embedding
  model. A smaller `YTRAG_EMBED_MODEL` helps.

---

## Acknowledgements

Written from scratch, but the idea came from a simpler single-file YouTube RAG
demo by [Yash Rajan](https://github.com/yashrajan-ai/rag-youtube-chatbot). None
of that code is reused here; this is an independent implementation with a
different architecture, timestamped citations, conversation memory, pluggable
providers, and a test suite.

## License

MIT — see [LICENSE](LICENSE).
