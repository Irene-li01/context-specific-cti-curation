"""
CCTI 9-step pipeline: fetch → filter → format → NER → normalise → STIX → recommend → evaluate → ingest

Usage:
    python pipeline.py                    # full run
    python pipeline.py --skip-fetch       # use existing MISP data
    python pipeline.py --skip-recommend   # steps 1-5 only
    python pipeline.py --use-secbert      # enable SecBERT NER layer
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib3
from datetime import datetime, timezone
from pathlib import Path

import requests
from crypto_utils import encrypt_file, encryption_enabled

# SSL: set MISP_VERIFY_SSL=false for self-signed certs, or MISP_CA_BUNDLE=/path/to/cert.pem
_ssl = os.getenv("MISP_VERIFY_SSL", "true").strip().lower()
if _ssl == "false":
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    MISP_SSL_VERIFY: bool | str = False
elif os.getenv("MISP_CA_BUNDLE"):
    MISP_SSL_VERIFY = os.getenv("MISP_CA_BUNDLE")
else:
    MISP_SSL_VERIFY = True

ROOT            = Path(__file__).resolve().parent
RAW_EVENTS_PATH = ROOT / "data" / "misp" / "misp_raw_events.json"

MISP_BASE   = os.getenv("MISP_BASE_URL", "https://misp.cti-lab.me")
API_KEY     = os.getenv("MISP_API_KEY")
EVENT_LIMIT = int(os.getenv("MISP_EVENT_LIMIT", "200"))
HEADERS     = {"Authorization": API_KEY, "Accept": "application/json", "Content-Type": "application/json"}


def _banner(title: str) -> None:
    print(f"\n{'=' * 60}\n  {title}\n{'=' * 60}")

def _step(n: int, label: str) -> None:
    print(f"\n[Step {n}] {label}")

def _run(cmd: list[str], label: str) -> bool:
    """Run a subprocess. Returns True on success."""
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    r = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True, encoding="utf-8", env=env)
    if r.stdout.strip():
        print(r.stdout.encode(sys.stdout.encoding or "ascii", errors="replace").decode(sys.stdout.encoding or "ascii"))
    if r.returncode != 0:
        print(f"  ERROR in {label}:\n{r.stderr.encode(sys.stdout.encoding or 'ascii', errors='replace').decode(sys.stdout.encoding or 'ascii')}")
        return False
    return True


def step_fetch() -> bool:
    """Fetch live events from the MISP REST API."""
    _step(1, "Fetching events from MISP API...")
    if not API_KEY:
        print("  ERROR — MISP_API_KEY is not set. Create a .env file from .env.example.")
        return False
    try:
        r = requests.post(
            f"{MISP_BASE}/events/restSearch", headers=HEADERS,
            json={"limit": EVENT_LIMIT, "returnFormat": "json", "includeContext": True},
            verify=MISP_SSL_VERIFY, timeout=180,
        )
        r.raise_for_status()
        events = r.json().get("response", [])
        if isinstance(events, dict):
            events = events.get("Event", [])
        RAW_EVENTS_PATH.parent.mkdir(parents=True, exist_ok=True)
        with RAW_EVENTS_PATH.open("w", encoding="utf-8") as f:
            json.dump(events, f, indent=2, ensure_ascii=False)
        encrypt_file(RAW_EVENTS_PATH)
        print(f"  OK — {len(events)} events saved{' (encrypted)' if encryption_enabled() else ''}")
        return True
    except requests.exceptions.ConnectionError:
        print(f"  ERROR — could not connect to {MISP_BASE}. Use --skip-fetch to run with existing data.")
    except requests.exceptions.HTTPError as e:
        print(f"  ERROR — HTTP {e.response.status_code}: check your API key.")
    except Exception as e:
        print(f"  ERROR — {e}")
    return False


def step_filter() -> bool:
    """Run noise-reduction filter over all data sources."""
    _step(2, "Running noise-reduction filter...")
    sys.path.insert(0, str(ROOT / "filtering"))
    try:
        from cti_filter import run_pipeline  # type: ignore
        cleaned_path, stats_path = run_pipeline(ROOT)
        print(f"  Cleaned -> {cleaned_path.relative_to(ROOT)}")
        encrypt_file(cleaned_path)
        return True
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def step_prepare_ner_input() -> bool:
    """Convert filtered attributes into NER-ready text records."""
    _step(3, "Preparing NER input...")
    return _run([sys.executable, str(ROOT / "ner" / "cti_data_formatting.py")], "cti_data_formatting.py")


def step_ner(use_secbert: bool = False) -> bool:
    """Extract named entities using spaCy (+ optionally SecBERT)."""
    _step(4, "Running NER...")
    cmd = [sys.executable, str(ROOT / "ner" / "ner_cti.py")]
    if use_secbert:
        cmd.append("--use-secbert")
    return _run(cmd, "ner_cti.py")


def step_normalise() -> bool:
    """Normalise MITRE technique names to T-codes."""
    _step(5, "Normalising MITRE techniques...")
    return _run([sys.executable, str(ROOT / "normalise_techniques.py")], "normalise_techniques.py")


def step_stix() -> bool:
    """Generate a STIX 2.1 bundle from NER output."""
    _step(6, "Generating STIX 2.1 bundle...")
    sys.path.insert(0, str(ROOT))
    stix_in  = ROOT / "ner" / "ner_results_v3_normalised.json"
    stix_out = ROOT / "cti_microservice_v1" / "curated_threats_stix3.json"
    if not stix_in.exists():
        stix_in = ROOT / "ner" / "ner_results_v3.json"
    if not stix_in.exists():
        print("  ERROR — NER output not found. Run Steps 4-5 first.")
        return False
    try:
        from stix_generator import generate_stix_bundle  # type: ignore
        generate_stix_bundle(str(stix_in), str(stix_out))
        print(f"  STIX bundle -> {stix_out.relative_to(ROOT)}")
        return True
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def step_recommend() -> bool:
    """Score threats against every org profile and save ranked recommendations."""
    _step(7, "Running recommendation engine...")
    sys.path.insert(0, str(ROOT))
    try:
        from recommendation_engine import list_available_profiles, run_recommendation_engine  # type: ignore
        profiles = list_available_profiles()
        if not profiles:
            print("  No org profiles found in Profiling/Instances/")
            return False
        for p in profiles:
            print(f"  Scoring {p['org_id']} ({p['industry_sector']})...")
            try:
                run_recommendation_engine(org_id=p["org_id"], use_llm=False, save_output=True)
            except Exception as e:
                print(f"    WARNING — skipped {p['org_id']}: {e}")
        rec_dir = ROOT / "data" / "recommendations"
        if encryption_enabled():
            for f in rec_dir.glob("recommendations_*.json"):
                encrypt_file(f)
        print(f"\n  Recommendations saved -> data/recommendations/")
        return True
    except Exception as e:
        print(f"  ERROR — {e}")
        return False


def step_evaluate() -> bool:
    """Evaluate recommendation quality across all org profiles."""
    _step(8, "Evaluating recommendations...")
    ok = _run([sys.executable, str(ROOT / "evaluation.py")], "evaluation.py")
    if ok and encryption_enabled():
        for f in (ROOT / "data" / "recommendations").glob("evaluation_report_*.json"):
            encrypt_file(f)
    return ok


def step_ingest() -> bool:
    """Ingest STIX bundle and recommendations into ChromaDB."""
    _step(9, "Ingesting into ChromaDB vector store...")
    return _run([sys.executable, str(ROOT / "cti_microservice_v1" / "data_ingestor.py")], "data_ingestor.py")


def main() -> int:
    parser = argparse.ArgumentParser(description="CCTI end-to-end pipeline")
    parser.add_argument("--skip-fetch",    action="store_true", help="Skip MISP fetch, use existing data")
    parser.add_argument("--skip-recommend",action="store_true", help="Stop after step 5 (skip STIX/recommend/ingest)")
    parser.add_argument("--skip-stix",     action="store_true", help="Skip STIX bundle generation (step 6)")
    parser.add_argument("--skip-evaluate", action="store_true", help="Skip evaluation report (step 8)")
    parser.add_argument("--skip-ingest",   action="store_true", help="Skip ChromaDB ingestion (step 9)")
    parser.add_argument("--use-secbert",   action="store_true", help="Enable SecBERT NER layer in step 4")
    args = parser.parse_args()

    _banner(f"CCTI Pipeline  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    if args.skip_fetch:
        _step(1, "Skipping MISP fetch (--skip-fetch)")
    elif not step_fetch():
        return 1

    if not step_filter():           return 1
    if not step_prepare_ner_input(): return 1
    if not step_ner(args.use_secbert): return 1
    if not step_normalise():        return 1

    if args.skip_recommend:
        _step(6, "Skipping steps 6-9 (--skip-recommend)")
    else:
        if not args.skip_stix and not step_stix():         return 1
        if not step_recommend():                           return 1
        if not args.skip_evaluate and not step_evaluate(): return 1
        if not args.skip_ingest and not step_ingest():     return 1

    _banner(f"Pipeline complete!  |  {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
