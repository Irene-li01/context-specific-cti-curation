"""
Normalises spaCy attack technique labels to canonical MITRE T-codes.
Uses LLM (Ollama) where available, falls back to a static keyword dictionary.

Run once after ner_cti.py to produce ner_results_v3_normalised.json.
The recommendation engine automatically prefers the normalised file.

Usage:
    python normalise_techniques.py
    python normalise_techniques.py --dry-run   # preview without saving
    python normalise_techniques.py --batch 50
"""
from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("TechniqueNormaliser")

_HERE         = Path(__file__).resolve().parent
_INPUT_FILE   = _HERE / "ner" / "ner_results_v3.json"
_OUTPUT_FILE  = _HERE / "ner" / "ner_results_v3_normalised.json"
_OLLAMA_MODEL = os.getenv("CTI_OLLAMA_MODEL", "deepseek-r1:14b")
_TCODE_RE     = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)

# Use keyword map from recommendation_engine if available; otherwise use a smaller inline copy
try:
    from recommendation_engine import _KEYWORD_TO_TECHNIQUE as _FALLBACK_MAP
except ImportError:
    _FALLBACK_MAP: Dict[str, str] = {
        "phishing": "T1566", "spearphishing": "T1566.001", "spear phishing": "T1566.001",
        "watering hole": "T1189", "brute force": "T1110", "ransomware": "T1486",
        "backdoor": "T1505", "credential dumping": "T1003", "privilege escalation": "T1068",
        "lateral movement": "T1021", "exfiltration": "T1041", "c2": "T1071",
        "command and control": "T1071", "process injection": "T1055", "powershell": "T1059.001",
        "web shell": "T1505.003", "webshell": "T1505.003", "obfuscation": "T1027",
        "masquerading": "T1036", "keylogger": "T1056.001", "network scanning": "T1046",
        "scheduled task": "T1053.005", "valid accounts": "T1078", "supply chain": "T1195",
        "rdp": "T1021.001", "wmi": "T1047", "pass the hash": "T1550.002",
        "kerberoasting": "T1558.003", "denial of service": "T1499", "scada": "T0800",
    }


def _static_normalise(technique: str) -> Optional[str]:
    if _TCODE_RE.match(technique.strip()):
        return technique.strip().upper()
    t = technique.strip().lower()
    return _FALLBACK_MAP.get(t) or _FALLBACK_MAP.get(t.replace("-", " "))


def _static_normalise_list(techniques: List[str]) -> List[str]:
    return [(_static_normalise(t) or t) for t in techniques]


_LLM_PROMPT = """\
You are a MITRE ATT&CK expert.
Convert the following attack technique labels to their canonical MITRE ATT&CK T-codes.

Rules:
- Return ONLY a JSON array of strings, no explanation, no markdown.
- Each element must be a T-code like "T1566" or "T1059.001".
- If a label is already a T-code, include it unchanged (uppercase).
- If a label has no clear MITRE mapping, include it unchanged as a string.
- Preserve the same order and count as the input list.

Input labels:
{techniques}

Output JSON array only:"""


def _llm_normalise_batch(techniques: List[str], ollama_client: Any) -> List[str]:
    if not techniques:
        return []
    prompt = _LLM_PROMPT.format(techniques=json.dumps(techniques, ensure_ascii=False))
    try:
        response = ollama_client.chat(model=_OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
        raw = response["message"]["content"].strip()
        if "<think>" in raw:
            raw = raw.split("</think>")[-1].strip()
        if "```" in raw:
            for part in raw.split("```"):
                part = part.strip().lstrip("json").strip()
                if part.startswith("["):
                    raw = part
                    break
        parsed = json.loads(raw)
        if not isinstance(parsed, list) or len(parsed) != len(techniques):
            return _static_normalise_list(techniques)
        return [item.upper() if _TCODE_RE.match(str(item).strip()) else str(item) for item in parsed]
    except Exception as exc:
        logger.warning("LLM call failed (%s) — using static fallback", exc)
        return _static_normalise_list(techniques)


def normalise_event(event: Dict, ollama_client: Optional[Any], stats: Dict) -> Dict:
    event = copy.deepcopy(event)
    raw_techniques: List[str] = event.get("entities", {}).get("attack_techniques", [])
    if not raw_techniques:
        return event

    already_tcodes, needs_normalise = [], []
    for tech in raw_techniques:
        if _TCODE_RE.match(tech.strip()):
            already_tcodes.append(tech.upper())
        else:
            needs_normalise.append(tech)

    stats["already_tcodes"] += len(already_tcodes)

    if not needs_normalise:
        event["entities"]["attack_techniques"] = already_tcodes
        return event

    llm_result = _llm_normalise_batch(needs_normalise, ollama_client) if ollama_client else needs_normalise[:]

    final = []
    for original, llm_out in zip(needs_normalise, llm_result):
        if _TCODE_RE.match(str(llm_out).strip()):
            final.append(llm_out.upper())
            stats["llm_converted"] += 1
        else:
            static = _static_normalise(original)
            if static:
                final.append(static)
                stats["static_fallback"] += 1
            else:
                final.append(original)
                stats["unresolved"] += 1

    seen: Set[str] = set()
    merged = []
    for tech in already_tcodes + final:
        key = tech.upper() if _TCODE_RE.match(tech.strip()) else tech.lower()
        if key not in seen:
            seen.add(key)
            merged.append(tech)

    event["entities"]["attack_techniques"] = merged
    return event


def run_normalisation(dry_run: bool = False, batch_size: int = 50) -> None:
    if not _INPUT_FILE.exists():
        logger.error("Input not found: %s — run ner/ner_cti.py first.", _INPUT_FILE)
        return

    with _INPUT_FILE.open("r", encoding="utf-8") as fh:
        events: List[Dict] = json.load(fh)

    if dry_run:
        events = events[:5]
        logger.info("DRY RUN — processing first 5 events only")

    logger.info("Loaded %d events from %s", len(events), _INPUT_FILE.name)

    ollama_client = None
    try:
        import ollama as _ollama
        _ollama.chat(model=_OLLAMA_MODEL, messages=[{"role": "user", "content": "ping"}])
        ollama_client = _ollama
        logger.info("Ollama connected — LLM normalisation enabled (%s)", _OLLAMA_MODEL)
    except ImportError:
        logger.warning("ollama not installed — using static fallback only")
    except Exception as exc:
        logger.warning("Ollama unavailable (%s) — using static fallback only", exc)

    stats = {"already_tcodes": 0, "llm_converted": 0, "static_fallback": 0, "unresolved": 0}
    normalised_events = []
    start = time.time()

    for i, event in enumerate(events, start=1):
        normalised_events.append(normalise_event(event, ollama_client, stats))
        if i % batch_size == 0 or i == len(events):
            logger.info("  %d/%d events (%.1fs)", i, len(events), time.time() - start)

    total = sum(stats.values())
    print(f"\n{'='*60}")
    print(f"  Technique Normalisation Complete")
    print(f"{'='*60}")
    print(f"  Events processed      : {len(events)}")
    print(f"  Total techniques      : {total}")
    print(f"  Already T-codes       : {stats['already_tcodes']}")
    print(f"  Converted by LLM      : {stats['llm_converted']}")
    print(f"  Converted by fallback : {stats['static_fallback']}")
    print(f"  Unresolved (kept)     : {stats['unresolved']}")
    print(f"  Elapsed               : {time.time() - start:.1f}s")

    if dry_run:
        print(f"\n  DRY RUN — preview (no file written):")
        for i, ev in enumerate(normalised_events):
            orig = events[i]["entities"].get("attack_techniques", [])
            norm = ev["entities"].get("attack_techniques", [])
            print(f"  Event {ev.get('event_id', '?')}: {orig} → {norm}")
        print(f"{'='*60}\n")
        return

    _OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with _OUTPUT_FILE.open("w", encoding="utf-8") as fh:
        json.dump(normalised_events, fh, indent=2, ensure_ascii=False)
    print(f"\n  Output saved to: {_OUTPUT_FILE}\n{'='*60}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Normalise spaCy attack technique labels to MITRE T-codes.")
    parser.add_argument("--dry-run", action="store_true", help="Preview first 5 events without saving.")
    parser.add_argument("--batch",   type=int, default=50, help="Log progress every N events.")
    args = parser.parse_args()
    run_normalisation(dry_run=args.dry_run, batch_size=args.batch)
