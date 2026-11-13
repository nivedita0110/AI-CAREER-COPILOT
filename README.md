# AI Career Copilot

🚀 **Live Demo:** [Try AI Career Copilot](https://ai-career-copilot-cgswkbfy7aw8ymavgmxnmc.streamlit.app/)

<div align="center">

**Enterprise-grade ATS Intelligence Engine & AI Career Strategist**

*Resume ATS Scoring • Skill-Gap Analysis • AI Recruiter Feedback • Resume Rewriting*

![Python](https://img.shields.io/badge/Python-3.11+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-red?logo=streamlit&logoColor=white)
![Google Gemini](https://img.shields.io/badge/Google%20Gemini-AI-orange?logo=google&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

</div>

---

## Overview

**AI Career Copilot** is a production-quality AI-powered career assistant built for software engineering candidates. It analyses resumes against job descriptions using real semantic embeddings, identifies skill gaps, provides structured AI recruiter feedback, and rewrites resumes for maximum ATS compatibility — all within a clean, modern dashboard.

> **Disclaimer:** This tool assists in resume optimisation and skill-gap awareness. It does not guarantee interview selection. AI output should always be reviewed before use. The resume rewriter never fabricates experience, degrees, or achievements.

---

## Features

| Feature | Description |
|---------|-------------|
| **ATS Score Engine** | Composite score: 70% semantic (chunk-averaged embeddings) + 30% vectorised skill coverage |
| **Matched / Missing Skills** | Batch semantic skill matching — no slow O(N×M) loops |
| **Partial Match Detection** | Identifies adjacent/related skills with configurable thresholds |
| **Resume Section Audit** | Checks for Contact Info, Experience, Education, Skills, Projects |
| **AI Recruiter Feedback** | Structured hiring verdict, impact gaps, optimisation suggestions, rewritten bullets |
| **Skill Gap Roadmap** | Prioritised competency matrix with fast-track learning paths and project ideas |
| **Resume Rewriter** | ATS-optimised rewrite — based strictly on the original resume content |
| **Multi-model Fallback** | Automatic fallback across Gemini model variants if one is unavailable |
| **Prompt Injection Safety** | Input sanitisation strips malicious tag spoofing from uploaded documents |
| **Sample Data** | Built-in sample resumes and JDs for instant demonstration |

---

## Architecture

```
ai_copilot/
├── app/                        # Core business logic (no Streamlit)
│   ├── config.py               # Centralised configuration & env loading
│   ├── parser.py               # PDF ingestion, text extraction, scanned-PDF detection
│   ├── scorer.py               # ATS engine: embeddings, skill extraction, scoring
│   ├── ai_service.py           # Gemini LLM calls, prompts, injection protection
│   └── utils.py                # Score categorisation, sample data, helpers
│
├── ui/
│   ├── app.py                  # Streamlit multi-page dashboard (UI only)
│   └── parser.py               # Backward-compat adapter → app.parser
│
├── tests/
│   └── test_core.py            # Pytest suite: 40+ tests, no network required
│
├── smoke_test.py               # Quick end-to-end integration smoke test
├── requirements.txt            # Pinned runtime dependencies
├── .env.example                # Safe placeholder (never commit .env)
├── .gitignore                  # Excludes venv, .env, caches, bytecode
└── README.md                   # This file
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Web Framework | [Streamlit](https://streamlit.io/) |
| Embedding Model | `all-MiniLM-L6-v2` via [Sentence Transformers](https://www.sbert.net/) |
| AI Engine | [Google Gemini](https://deepmind.google/technologies/gemini/) via [LangChain](https://www.langchain.com/) |
| PDF Parsing | [PyMuPDF](https://pymupdf.readthedocs.io/) |
| Similarity Engine | [Scikit-learn](https://scikit-learn.org/) cosine similarity + [NumPy](https://numpy.org/) |
| Environment | [python-dotenv](https://pypi.org/project/python-dotenv/) |

---

## How ATS Scoring Works

The score is a **weighted composite of two independent signals**:

```
Final Score = (0.70 × Semantic Score) + (0.30 × Skill Coverage Score)
```

**Semantic Score (70%):**  
The resume and JD are split into overlapping 160-word chunks (30-word overlap), each encoded into 384-dimensional vectors via `all-MiniLM-L6-v2`. Chunk embeddings are mean-pooled and L2-normalised into a single document vector. Cosine similarity between the two document vectors is the semantic score. This avoids the 256-token truncation limit that affects long documents.

**Skill Coverage Score (30%):**  
Skills are extracted from both documents using a layered approach: a curated multi-word technical phrase list (40+ terms) + regex token extraction with an expanded 100+ word stoplist. All JD skills and resume skills are batch-encoded in two calls, then compared via matrix dot product — `O(N+M)` encodings rather than the naive `O(N×M)`. Skills above 65% similarity are `matched`, 45–65% are `partial` (counted as 0.5), below 45% are `missing`.

---

## Screenshots

> *Add screenshots here after launch.*

---

## Quick Start

### Prerequisites
- Python 3.11+ (project tested on 3.13)
- A [Google Gemini API key](https://aistudio.google.com/)

### 1. Clone

```bash
git clone https://github.com/your-username/ai-career-copilot.git
cd ai-career-copilot
```

### 2. Create Virtual Environment

```bash
python -m venv venv
# Windows
venv\Scripts\activate
# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment

```bash
cp .env.example .env
# Edit .env and set your GOOGLE_API_KEY
```

Your `.env` file:
```
GOOGLE_API_KEY=your_actual_api_key_here
```

### 5. Run the Application

```bash
# From project root
streamlit run ui/app.py
```

The app opens at `http://localhost:8501`.

---

## Running Tests

```bash
# Full pytest suite (no network, no LLM required)
venv\Scripts\python.exe -m pytest tests/ -v

# Quick smoke test
venv\Scripts\python.exe smoke_test.py
```

---

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `GOOGLE_API_KEY` | ✅ Yes | Google AI Studio API key for Gemini access |

You can also paste the API key directly into the sidebar of the running app without restarting.

---

## Deployment (Streamlit Community Cloud)

1. Push your code to a **public GitHub repository** (ensure `.env` is in `.gitignore`).
2. Go to [share.streamlit.io](https://share.streamlit.io) and connect your repository.
3. Set **Main file path:** `ui/app.py`
4. Add `GOOGLE_API_KEY` as a **Secret** in the Streamlit Cloud dashboard.
5. Click **Deploy**.

> The first cold start downloads the `all-MiniLM-L6-v2` model (~90 MB). Subsequent loads use Streamlit's `@st.cache_resource`.

---

## Known Limitations

| Limitation | Detail |
|-----------|--------|
| **Scanned PDFs** | Image-only PDFs cannot be parsed. OCR is not included. Use text-based PDFs. |
| **ATS accuracy** | Score is an estimate based on keyword + semantic alignment. It does not simulate any specific vendor's ATS. |
| **Gemini API quota** | Free tier has rate limits. The app falls back across model variants and shows a clear error on quota exhaustion. |
| **Skill extraction** | Multi-word phrase detection covers ~40 curated terms. Highly niche domain skills may be missed by the keyword layer. |
| **Model token limit** | Embedding model truncation is handled via chunking, but very unusual document layouts may affect accuracy. |

---

## License

MIT License — see `LICENSE` for details.

---

## Acknowledgements

- [Sentence Transformers](https://www.sbert.net/) — `all-MiniLM-L6-v2`
- [LangChain](https://www.langchain.com/) — LLM orchestration
- [Google DeepMind / Gemini](https://deepmind.google/technologies/gemini/)
- [PyMuPDF](https://pymupdf.readthedocs.io/)
- [Streamlit](https://streamlit.io/)

