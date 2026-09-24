# CCTI — Curated Cyber Threat Intelligence

Organisations drown in threat intelligence that mostly doesn't apply to them. CCTI fixes that by building an end-to-end pipeline that fetches live threat data from MISP, extracts meaningful entities, scores each threat against an organisation's specific profile, and surfaces the most relevant ones through a web dashboard and REST API.

---

## How it works

The system runs as a nine-step pipeline:

1. **Fetch** — Pull live events from the MISP REST API
2. **Filter** — Strip duplicates, raw indicators, and low-signal noise from the available CTI datasets
3. **Format** — Convert filtered attributes into NER-ready text records
4. **NER** — Extract named entities using a hybrid spaCy + SecBERT pipeline. Entities: threat actors, malware, attack techniques, technologies, industries, CVEs, IOCs
5. **Normalise** — Map free-text MITRE technique names to official T-codes
6. **STIX** — Transform NER output into a STIX 2.1 bundle for interoperability
7. **Recommend** — Score all threats against each organisation profile using TF-IDF cosine similarity + CPE asset matching + MITRE ATT&CK overlap
8. **Evaluate** — Automated quality checks: MITRE coverage, cross-org differentiation
9. **Ingest** — Load STIX bundle and recommendations into ChromaDB for RAG-powered natural language queries

The scoring formula:
```
Final Score = (TF-IDF cosine + CPE boost + MITRE bonus) × severity_multiplier × 100
```

Entity confidence scores from the NER step are propagated into the scoring — regex-extracted T-codes (confidence 0.98) contribute more than spaCy guesses (0.75).

---

## Prerequisites

Before running anything, you need:

1. **A MISP instance** — the pipeline fetches live threat data from a cloud-hosted MISP instance that we deployed and configured based on the official MISP guidelines. In this project, `misp.cti-lab.me` is used as the CTI data source. To use your own instance, set `MISP_BASE_URL` in `.env`.

2. **MISP API key** — set `MISP_API_KEY` in your `.env` file (copy from `.env.example`).
   - Open `https://misp.cti-lab.me`.
   - Log in with the admin credentials provided by the course or lab supervisor.
   - Go to **Admin → Auth keys → Add authentication key**.
   - Copy the generated key into `.env`:
     ```
     MISP_API_KEY=replace-with-your-misp-api-key
     ```

3. **SecBERT model weights** — `model.safetensors` (331MB) is excluded from the repository due to size. To use `--use-secbert`, either retrain the model or obtain the weights file and place it in `Machine learning/secbert_cti_model_final/`. The pipeline runs fine without it using spaCy only.

4. **Groq API key** *(optional — enables SOC AI Chat)* — The dashboard includes a RAG-powered SOC AI Chat panel backed by Groq. Without a key the dashboard still works; only the chat panel is disabled.
   - Create an API key at [console.groq.com](https://console.groq.com).
   - Create an API key and add it to your `.env`:
     ```
     GROQ_API_KEY=replace-with-your-groq-api-key
     ```
   - Groq was chosen over self-hosted models (e.g. Ollama) so that `docker compose up` requires no large model downloads, keeping the deployment lightweight and accessible for any client machine.

---

## Running the pipeline

### Docker (recommended — handles all dependencies)
```bash
cp .env.example .env    # fill in MISP_API_KEY
./deploy.sh             # build image, run pipeline, start API
# → http://localhost:8000
```

To enable the optional SecBERT extraction layer during deployment:
```bash
PIPELINE_ARGS="--use-secbert" ./deploy.sh
```

To start the API and dashboard with existing pipeline data:
```bash
docker compose up -d api
```

### Local (development)
```bash
cp .env.example .env
pip install -r requirements.txt
python -m spacy download en_core_web_sm

python pipeline.py                    # full run (all 9 steps)
python pipeline.py --skip-fetch       # use existing MISP data
python pipeline.py --skip-recommend   # steps 1-5 only
python pipeline.py --use-secbert      # enable SecBERT NER layer

cd cti_microservice_v1
python dashboard_only.py              # dashboard and REST endpoints, fast startup
python api.py                         # dashboard, REST endpoints, and RAG Q&A
# → http://localhost:8000
```

### Tests
```bash
pytest                      # run all tests
pytest filtering/tests/
pytest cti_microservice_v1/tests/
pytest tests/
```

### Moving to another server or local machine

The project is portable as long as configuration and generated runtime data are moved deliberately:

1. Clone the repository on the new machine.
2. Copy `.env.example` to `.env` and set the required values again. Do not commit or copy old secrets through Git.
3. If you want to keep the current demo data, copy these generated folders/files from the old server:
   - `data/misp/`
   - `data/cleaned/`
   - `data/recommendations/`
   - `ner/ner_results_v3.json`
   - `ner/ner_results_v3_normalised.json`
   - `cti_microservice_v1/curated_threats_stix3.json`
   - `cti_microservice_v1/vector_store/`
4. If encrypted files are being reused, copy the same `CCTI_ENCRYPTION_KEY` into the new `.env`. Without the original key, encrypted JSON files cannot be read.
5. Rebuild and start the services:
   ```bash
   docker compose build
   docker compose up -d api
   ```
6. If you do not copy generated data, run a fresh pipeline instead:
   ```bash
   ./deploy.sh
   ```

For cloud migration, also update DNS, firewall rules, HTTPS reverse proxy settings, and any external MISP/Groq keys that are tied to the old host.

---

## Project structure

```
CCTI/
├── pipeline.py                   # Canonical 9-step pipeline entry point
├── recommendation_engine.py      # TF-IDF + CPE + MITRE scoring engine
├── evaluation.py                 # Cross-org evaluation and ablation tests
├── normalise_techniques.py       # MITRE T-code normalisation
├── stix_generator.py             # STIX 2.1 bundle generation
├── crypto_utils.py               # Fernet at-rest encryption utilities
├── encrypt_profiles.py           # One-time org profile encryption script
│
├── filtering/                    # Noise reduction pipeline
│   ├── cti_filter.py             # Core filtering logic
│   ├── rules.json                # Tunable filter rules
│   └── tests/                    # Filter unit tests
│
├── ner/                          # Named entity recognition
│   ├── ner_cti.py                # spaCy + SecBERT hybrid extractor
│   ├── cti_data_formatting.py    # Converts filtered CTI to NER input
│   ├── ner_results_v3.json       # Raw NER output
│   └── ner_results_v3_normalised.json  # Normalised NER output
│
├── Machine learning/             # SecBERT fine-tuning
│   ├── train_secbert.py          # Fine-tune SecBERT on CTI NER data
│   ├── prepare_cyner.py          # Convert CyNER dataset to training format
│   ├── rescue_aligner.py         # BIO tag alignment for training data
│   ├── secbert_train_data.json   # Original training data
│   ├── train_data/               # CyNER CoNLL files (train/valid/test)
│   └── secbert_cti_model_final/  # Trained model (model.safetensors gitignored)
│
├── cti_microservice_v1/          # FastAPI REST API + web dashboard
│   ├── api.py                    # Full API with RAG engine
│   ├── dashboard_only.py         # Lightweight server (dashboard only)
│   ├── threats_endpoints.py      # All REST endpoints
│   ├── rag_engine.py             # RAG Q&A engine (two-pass retrieval: org metadata sort + semantic search)
│   ├── data_ingestor.py          # Ingests STIX + recommendations into ChromaDB
│   ├── config.py                 # Environment-variable configuration
│   └── static/index.html         # Frontend dashboard (Chart.js + Plotly; clickable threat cards with AI analysis)
│
├── Profiling/                    # Organisation profiles
│   ├── Schemas/                  # JSON schema for profiles
│   └── Instances/                # 12 org profiles across Finance, Healthcare, Energy, Other
│
├── data/                         # CTI source data
│   ├── Feodo.json                # C2 tracker feed
│   ├── Malware Bazaar.json       # Malware samples feed
│   ├── OpenPhish.json            # Phishing URLs
│   ├── URLHaus.json              # Malicious URLs
│   ├── phishtank.json            # Phishing database
│   ├── misp/                     # Live MISP fetch output (gitignored)
│   ├── cleaned/                  # Filter pipeline output (gitignored)
│   └── recommendations/          # Recommendation engine output (gitignored)
│
├── Dockerfile                    # Container image definition
├── docker-compose.yml            # Two services: api (24/7) + pipeline (one-shot)
├── deploy.sh                     # Digital Ocean deployment script
├── .env.example                  # Environment variable template
└── requirements.txt              # Python dependencies
```

---

## Evaluation results

Evaluation output is generated during Step 8 and written to:

```text
data/recommendations/evaluation_report_<timestamp>.json
data/recommendations/evaluation_report_<timestamp>.txt
```

The exact counts vary with the latest MISP fetch, filtering output, NER results, and available recommendation files. Use the newest timestamped report for the current run.

**Snapshot from latest verified run (June 2026):**

| Metric | Result |
|--------|--------|
| Organisations evaluated | 12 (Finance, Healthcare, Energy, Other) |
| CTI events scored | 567 |
| MITRE matching | All 12 orgs PASS, 95–100% top-N matched |
| Max MITRE lift | +47.5 points (ORG-EN-001) |
| Cross-org differentiation | Energy vs Healthcare overlap: 0% |
| IoCs processed | 669,274 |
| NER-enriched records | 5,085 |
| ChromaDB chunks ingested | 14,429 (STIX indicators + curated recommendations) |

---

## Security

- MISP API key stored in environment variable, never in source code
- SSL certificate verification configurable via `MISP_VERIFY_SSL` / `MISP_CA_BUNDLE`
- At-rest encryption (AES-128 Fernet) for raw events, filtered data, recommendations, evaluation reports, and org profiles — controlled by `CCTI_ENCRYPTION_KEY`
- Internal MISP identifiers stripped from all API responses

---

## NER approach

The NER pipeline uses a hybrid of two models:

| Source | Model | Confidence | Entities |
|--------|-------|------------|---------|
| Regex | Pattern matching | 0.95–0.98 | CVEs, T-codes, IPs, hashes |
| Dictionary | MITRE ATT&CK lookup | 0.90 | Threat actors, techniques |
| spaCy | en_core_web_sm | 0.75 | Organisations, products |
| SecBERT | Fine-tuned on CyNER | 0.80 | Actors, TTPs, technologies, industries |

SecBERT (`jackaduma/SecBERT`) was fine-tuned on a combined dataset of 4,856 sentences from the CyNER corpus and existing CTI training data. Enable with `--use-secbert`.

---

## Tech stack

| Component | Technology |
|-----------|-----------|
| Threat feeds | MISP REST API |
| NER | spaCy + SecBERT (HuggingFace Transformers) |
| Scoring | scikit-learn TF-IDF, MITRE ATT&CK |
| Encryption | cryptography (Fernet) |
| Threat format | STIX 2.1 |
| Vector DB | ChromaDB + all-MiniLM-L6-v2 |
| RAG | LangChain + Groq (Llama 3.1) |
| API | FastAPI + Uvicorn |
| Frontend | Vanilla JS, Chart.js, Plotly |
| Deployment | Docker, Digital Ocean |
| Language | Python 3.11 |

---

## The team

| Name | Role |
|------|------|
| Maaz Arshad Beg | Technical lead, system integration, pipeline, security |
| Aini Li | Filtering optimisation, dashboard, deployment |
| Zexin Zhao | SecBERT model training, ML pipeline |
| Jiarui Li | Organisation profiling, frontend |
| Freshin Francis | Recommendation engine, similarity matching, frontend development |

Supervisor: Prof. Mamello Thinyane, Adelaide University
