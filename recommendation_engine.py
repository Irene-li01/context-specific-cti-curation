"""
CTI Recommendation Engine
Score = (cosine_sim + cpe_boost + mitre_bonus) x severity_multiplier x 100
"""
from __future__ import annotations

import json
import logging
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("RecommendationEngine")

_HERE = Path(__file__).resolve().parent

_LLM_NER_FILE              = _HERE / "Machine learning" / "cti_ner_results_v2.json"  # optional LLM NER output; not present by default
_SPACY_NER_FILE            = _HERE / "ner" / "ner_results_v3.json"
_SPACY_NER_FILE_NORMALISED = _HERE / "ner" / "ner_results_v3_normalised.json"
_PROFILES_DIR              = _HERE / "Profiling" / "Instances"
_OUTPUT_DIR                = _HERE / "data" / "recommendations"

_OLLAMA_MODEL      = os.getenv("CTI_OLLAMA_MODEL",          "deepseek-r1:14b")
_MIN_SCORE         = float(os.getenv("CTI_REC_MIN_SCORE",   "1.0"))
_TOP_N             = int(os.getenv("CTI_REC_TOP_N",         "20"))
_CPE_BOOST         = float(os.getenv("CTI_REC_CPE_BOOST",   "0.25"))
_MITRE_BOOST       = float(os.getenv("CTI_REC_MITRE_BOOST", "0.20"))
_MITRE_CAP         = float(os.getenv("CTI_REC_MITRE_CAP",   "0.60"))
_SPECIFICITY_FLOOR = float(os.getenv("CTI_REC_SPECIFICITY_FLOOR", "0.35"))
_KEEP_RECOMMENDATION_RUNS = int(os.getenv("CTI_REC_KEEP_RUNS", "10"))

_GENERIC_THREAT_TERMS = {
    "ransomware", "phishing", "malware", "spear-phishing",
    "spear-phishing attacks", "watering-hole attacks", "credential theft", "carding",
}

SEVERITY_WEIGHTS: Dict[str, float] = {
    "critical": 1.5, "high": 1.2, "medium": 1.0, "low": 0.8, "unknown": 0.5,
}

# Sorted longest-first so shorter substrings don't match before more specific ones.
_KEYWORD_TO_TECHNIQUE: Dict[str, str] = {
    "phishing": "T1566", "spearphishing": "T1566.001", "spear phishing": "T1566.001",
    "watering hole": "T1189", "drive-by": "T1189", "drive by": "T1189",
    "supply chain": "T1195", "valid accounts": "T1078", "default accounts": "T1078.001",
    "brute force": "T1110", "password spraying": "T1110.003", "credential stuffing": "T1110.004",
    "exploit public": "T1190", "external remote services": "T1133", "rdp": "T1021.001",
    "powershell": "T1059.001", "command line": "T1059", "cmd": "T1059.003",
    "bash": "T1059.004", "macro": "T1204.002", "malicious macro": "T1204.002",
    "script": "T1059", "wmi": "T1047", "scheduled task": "T1053.005",
    "backdoor": "T1505", "web shell": "T1505.003", "webshell": "T1505.003",
    "account manipulation": "T1098", "privilege escalation": "T1068",
    "token impersonation": "T1134", "bypass uac": "T1548.002",
    "obfuscation": "T1027", "masquerading": "T1036", "rootkit": "T1014",
    "process injection": "T1055", "dll injection": "T1055.001",
    "indicator removal": "T1070", "log clearing": "T1070.001",
    "disable security tools": "T1562", "credential dumping": "T1003",
    "mimikatz": "T1003.001", "lsass": "T1003.001", "pass the hash": "T1550.002",
    "kerberoasting": "T1558.003", "keylogger": "T1056.001",
    "network scanning": "T1046", "port scan": "T1046", "account discovery": "T1087",
    "lateral movement": "T1021", "remote services": "T1021", "smb": "T1021.002",
    "exfiltration": "T1041", "data exfiltration": "T1041",
    "dns tunnelling": "T1048.001", "dns tunneling": "T1048.001",
    "ransomware": "T1486", "data encrypted": "T1486", "disk wipe": "T1561",
    "denial of service": "T1499", "dos": "T1499",
    "c2": "T1071", "c&c": "T1071", "command and control": "T1071",
    "beaconing": "T1071", "domain fronting": "T1090.004",
    "scada": "T0800", "ics": "T0800", "modbus": "T0871", "dnp3": "T0871",
}

_KEYWORD_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\b" + re.escape(kw) + r"\b", re.IGNORECASE), tid)
    for kw, tid in sorted(_KEYWORD_TO_TECHNIQUE.items(), key=lambda x: -len(x[0]))
]
_MITRE_TCODE_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)


@lru_cache(maxsize=1)
def _profile_match_frequencies() -> Tuple[Counter[str], Counter[str]]:
    technique_counts: Counter[str] = Counter()
    actor_counts: Counter[str] = Counter()
    if not _PROFILES_DIR.is_dir():
        return technique_counts, actor_counts
    from crypto_utils import decrypt_json
    for path in sorted(_PROFILES_DIR.glob("*.json")):
        try:
            profile = decrypt_json(path)
        except Exception:
            continue
        tracked = profile.get("risk_profile", {}).get("threat_landscape", [])
        technique_counts.update(extract_techniques(tracked))
        actor_counts.update(
            item.lower().strip() for item in tracked
            if not _MITRE_TCODE_RE.search(str(item))
            and item.lower().strip() not in _GENERIC_THREAT_TERMS
        )
    return technique_counts, actor_counts


def _specificity_weight(counter: Counter[str], key: str) -> float:
    freq = counter.get(key, 0)
    return 1.0 if freq <= 1 else max(_SPECIFICITY_FLOOR, 1.0 / math.sqrt(freq))


def load_org_profile(org_id: str) -> Dict[str, Any]:
    if not _PROFILES_DIR.is_dir():
        raise FileNotFoundError(f"Profiles directory not found: {_PROFILES_DIR}")
    from crypto_utils import decrypt_json
    for path in sorted(_PROFILES_DIR.glob("*.json")):
        try:
            profile = decrypt_json(path)
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Skipping unreadable profile %s: %s", path.name, exc)
            continue
        if profile.get("metadata", {}).get("org_id") == org_id:
            logger.info("Loaded profile '%s' from %s", org_id, path.name)
            _validate_profile(profile)
            return profile
    raise FileNotFoundError(
        f"No profile found with org_id='{org_id}' under {_PROFILES_DIR}\n"
        f"Available: {[p.name for p in _PROFILES_DIR.glob('*.json')]}"
    )


def list_available_profiles() -> List[Dict[str, str]]:
    if not _PROFILES_DIR.is_dir():
        return []
    from crypto_utils import decrypt_json
    summaries = []
    for path in sorted(_PROFILES_DIR.glob("*.json")):
        try:
            profile = decrypt_json(path)
            summaries.append({
                "org_id":          profile.get("metadata", {}).get("org_id", "unknown"),
                "industry_sector": profile.get("organizational_context", {}).get("industry_sector", "unknown"),
                "filename":        path.name,
            })
        except Exception:
            pass
    return summaries


def _validate_profile(profile: Dict[str, Any]) -> None:
    for key in ["metadata", "organizational_context", "technology_stack", "risk_profile"]:
        if key not in profile:
            raise ValueError(f"Organisation profile missing required field: '{key}'")


def load_cti_events(
    llm_ner_path:   Path = _LLM_NER_FILE,
    spacy_ner_path: Path = _SPACY_NER_FILE,
) -> List[Dict[str, Any]]:
    events_by_id: Dict[str, Dict[str, Any]] = {}

    effective_spacy = _SPACY_NER_FILE_NORMALISED if _SPACY_NER_FILE_NORMALISED.exists() else spacy_ner_path
    if effective_spacy != spacy_ner_path:
        logger.info("Using normalised spaCy NER file")

    for path, label in [(effective_spacy, "spaCy"), (llm_ner_path, "LLM")]:
        if path.exists():
            try:
                with path.open("r", encoding="utf-8") as fh:
                    for ev in json.load(fh):
                        eid = str(ev.get("event_id", ""))
                        if eid:
                            events_by_id[eid] = ev
                logger.info("Loaded %s NER from %s", label, path.name)
            except Exception as exc:
                logger.warning("Could not read %s NER file: %s", label, exc)

    if not events_by_id:
        raise FileNotFoundError(
            "No NER result files found. Run ner_cti.py first.\n"
            f"  spaCy NER: {spacy_ner_path}\n"
            f"  LLM NER  : {llm_ner_path}"
        )

    merged = list(events_by_id.values())
    logger.info("Merged corpus: %d unique CTI events", len(merged))
    return merged


def _profile_to_document(profile: Dict[str, Any]) -> str:
    """Industry repeated 3x for higher TF-IDF weight."""
    ctx      = profile.get("organizational_context", {})
    stack    = profile.get("technology_stack", {})
    risk     = profile.get("risk_profile", {})
    industry = (ctx.get("industry_sector", "") + " ") * 3
    assets   = " ".join(a.get("asset_name", "") for a in stack.get("assets", []))
    threats  = " ".join(risk.get("threat_landscape", []))
    critical = " ".join(risk.get("critical_assets", []))
    return f"{industry} {assets} {threats} {critical}".lower()


def _cti_event_to_document(entities: Dict[str, Any]) -> str:
    parts = (
        entities.get("technologies",      []),
        entities.get("industries",        []),
        entities.get("attack_techniques", []),
        entities.get("threat_actors",     []),
    )
    return " ".join(w for group in parts for w in group).lower()


def _get_entity_confidence(entities: Dict[str, Any], bucket: str, text: str) -> float:
    text_lower = text.lower()
    for detail in entities.get("entity_details", []):
        if detail.get("bucket") == bucket and detail.get("text", "").lower() == text_lower:
            return float(detail.get("confidence", 0.7))
    return 0.7


def _compute_cpe_boost(profile: Dict[str, Any], entities: Dict[str, Any],
                       boost_per_match: float = _CPE_BOOST) -> float:
    assets = profile.get("technology_stack", {}).get("assets", [])
    cti_techs_map = {t.lower(): t for t in entities.get("technologies", [])}
    if not assets or not cti_techs_map:
        return 0.0

    hits = 0.0
    for asset in assets:
        asset_name  = asset.get("asset_name", "").lower()
        criticality = asset.get("criticality", 1)
        cpe_parts   = asset.get("cpe", "").split(":")
        cpe_product = cpe_parts[4].replace("_", " ").lower() if len(cpe_parts) >= 5 else ""

        matched = next(
            (t for t in cti_techs_map
             if asset_name in t or t in asset_name
             or (cpe_product and (cpe_product in t or t in cpe_product))),
            None,
        )
        if matched is not None:
            confidence = _get_entity_confidence(entities, "technologies", cti_techs_map[matched])
            hits += (criticality / 5.0) * confidence
        if hits >= 3.0:
            break

    return min(hits * boost_per_match, 3 * boost_per_match)


def extract_techniques(text_or_list: str | List[str]) -> Set[str]:
    combined = " | ".join(str(x) for x in text_or_list) if isinstance(text_or_list, list) else str(text_or_list)
    techniques: Set[str] = set()
    for match in _MITRE_TCODE_RE.finditer(combined):
        techniques.add(match.group(0).upper())
    for pattern, tid in _KEYWORD_PATTERNS:
        if pattern.search(combined):
            techniques.add(tid.upper())
    return techniques


# Industry-sector fallback techniques — used when an org's explicit threat_landscape
# produces zero matches against the current CTI dataset. Keeps scoring meaningful
# regardless of which events were fetched, without hard-coding dataset-specific values.
_INDUSTRY_FALLBACK_TECHNIQUES: Dict[str, Set[str]] = {
    "finance":     {"T1566", "T1486", "T1078", "T1190", "T1505"},
    "healthcare":  {"T1566", "T1486", "T1078", "T1505", "T1068"},
    "energy":      {"T1566", "T1486", "T1190", "T1505", "T1068"},
    "other":       {"T1566", "T1486", "T1078"},
}


def _industry_fallback(profile: Dict[str, Any]) -> Set[str]:
    """Return baseline T-codes for the org's sector when explicit landscape has no matches."""
    sector = profile.get("organizational_context", {}).get("industry_sector", "").lower()
    for key in _INDUSTRY_FALLBACK_TECHNIQUES:
        if key in sector:
            return _INDUSTRY_FALLBACK_TECHNIQUES[key]
    return _INDUSTRY_FALLBACK_TECHNIQUES["other"]


def compute_mitre_score(profile: Dict[str, Any], entities: Dict[str, Any],
                        boost_per_match: float = _MITRE_BOOST,
                        max_bonus: float = _MITRE_CAP) -> float:
    tracked = profile.get("risk_profile", {}).get("threat_landscape", [])

    technique_counts, actor_counts = _profile_match_frequencies()
    org_t_codes = extract_techniques(tracked) if tracked else set()
    org_actors  = {
        t.lower().strip() for t in tracked
        if not _MITRE_TCODE_RE.search(t) and t.lower().strip() not in _GENERIC_THREAT_TERMS
    }
    cti_t_codes    = extract_techniques(entities.get("attack_techniques", []))
    cti_actors_raw = {a.strip(): a.strip() for a in entities.get("threat_actors", [])}
    cti_actors     = {a.lower() for a in cti_actors_raw}

    score = 0.0
    for t_code in org_t_codes & cti_t_codes:
        conf = _get_entity_confidence(entities, "attack_techniques", t_code)
        score += boost_per_match * conf * _specificity_weight(technique_counts, t_code)
    for actor_lower in org_actors & cti_actors:
        original = cti_actors_raw.get(actor_lower, actor_lower)
        conf = _get_entity_confidence(entities, "threat_actors", original)
        score += boost_per_match * 1.5 * conf * _specificity_weight(actor_counts, actor_lower)

    # If explicit profile produced no matches, fall back to industry-sector baseline
    # so scoring stays meaningful regardless of which events the dataset contains.
    if score == 0.0 and cti_t_codes:
        fallback = _industry_fallback(profile)
        for t_code in fallback & cti_t_codes:
            conf = _get_entity_confidence(entities, "attack_techniques", t_code)
            score += (boost_per_match * 0.5) * conf * _specificity_weight(technique_counts, t_code)

    return min(score, max_bonus)


def explain_mitre_matches(profile: Dict[str, Any], entities: Dict[str, Any]) -> List[Dict[str, str]]:
    tracked = profile.get("risk_profile", {}).get("threat_landscape", [])
    if not tracked:
        return []
    org_t_codes = extract_techniques(tracked)
    org_actors  = {
        t.lower().strip() for t in tracked
        if not _MITRE_TCODE_RE.search(t) and t.lower().strip() not in _GENERIC_THREAT_TERMS
    }
    cti_t_codes = extract_techniques(entities.get("attack_techniques", []))
    cti_actors  = {a.lower().strip() for a in entities.get("threat_actors", [])}

    matches = []
    for t_code in sorted(org_t_codes & cti_t_codes):
        matches.append({"org_technique": t_code, "cti_technique": t_code, "match_type": "technique"})
    for actor in sorted(org_actors & cti_actors):
        matches.append({"org_technique": actor.upper(), "cti_technique": actor.upper(), "match_type": "actor"})
    return matches


def compute_recommendations(profile: Dict[str, Any], cti_events: List[Dict[str, Any]],
                             min_score: float = _MIN_SCORE, top_n: int = _TOP_N) -> List[Dict[str, Any]]:
    org_id = profile.get("metadata", {}).get("org_id", "unknown")
    logger.info("Scoring %d events for org_id='%s'", len(cti_events), org_id)

    org_doc  = _profile_to_document(profile)
    cti_docs = [_cti_event_to_document(ev.get("entities", {})) for ev in cti_events]

    non_empty       = [i for i, d in enumerate(cti_docs) if d.strip()]
    events_filtered = [cti_events[i] for i in non_empty]
    corpus          = [org_doc] + [cti_docs[i] for i in non_empty]

    if not non_empty:
        logger.warning("All CTI event documents are empty — no recommendations possible.")
        return []

    vectorizer   = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    recommendations: List[Dict[str, Any]] = []
    for i, event in enumerate(events_filtered):
        entities    = event.get("entities", {})
        base_sim    = float(similarities[i])
        cpe_boost   = _compute_cpe_boost(profile, entities)
        mitre_bonus = compute_mitre_score(profile, entities)
        severity    = entities.get("severity", "unknown").lower()
        multiplier  = SEVERITY_WEIGHTS.get(severity, 0.5)
        final_score = (base_sim + cpe_boost + mitre_bonus) * multiplier * 100

        if final_score < min_score:
            continue

        recommendations.append({
            "rank": 0, "score": round(final_score, 3),
            "cosine_sim": round(base_sim, 4), "cpe_boost": round(cpe_boost, 4),
            "mitre_bonus": round(mitre_bonus, 4), "severity": severity,
            "severity_multiplier": multiplier, "event_id": event.get("event_id", ""),
            "attr_id": event.get("attr_id", ""),
            "category": event.get("category", ""),
            "type": event.get("type", ""),
            "indicator": event.get("indicator", ""),
            "title": event.get("title", ""),
            "description": event.get("description", ""),
            "raw_text": event.get("raw_text", ""),
            "entities": entities,
            "mitre_matches": explain_mitre_matches(profile, entities),
            "llm_explanation": "",
        })

    recommendations.sort(key=lambda r: r["score"], reverse=True)
    for rank, rec in enumerate(recommendations[:top_n], start=1):
        rec["rank"] = rank

    result = recommendations[:top_n]
    logger.info("Generated %d recommendations (min_score=%.1f)", len(result), min_score)
    return result


_EXPLANATION_PROMPT = """\
You are a concise cybersecurity analyst.
Given the following threat intelligence match for an organisation, write ONE short paragraph \
(3-4 sentences) explaining:
1. Why this threat is relevant to the organisation.
2. What the key risk is.
3. One recommended defensive action.

Organisation:
  Industry        : {industry}
  Key assets      : {assets}

Matched CTI Event:
  Title           : {title}
  Severity        : {severity}
  Summary         : {summary}
  Threat actors   : {actors}
  Technologies    : {techs}
  Attack techniques: {attacks}
  Direct profile match: {mitre_matches}

Write only the paragraph. No bullet points. No headers.\
"""


def generate_llm_explanations(profile: Dict[str, Any], recommendations: List[Dict[str, Any]],
                               top_n_explain: int = 5) -> List[Dict[str, Any]]:
    try:
        import ollama as _ollama
    except ImportError:
        logger.warning("ollama not installed — skipping LLM explanations.")
        return recommendations

    ctx      = profile.get("organizational_context", {})
    industry = ctx.get("industry_sector", "unknown")
    assets   = ", ".join(a.get("asset_name", "") for a in profile.get("technology_stack", {}).get("assets", []))
    logger.info("Generating LLM explanations for top %d results...", min(top_n_explain, len(recommendations)))

    for rec in recommendations[:top_n_explain]:
        entities  = rec.get("entities", {})
        matches   = ", ".join(f"{m['match_type']}:{m['cti_technique']}" for m in rec.get("mitre_matches", [])) or "None"
        prompt    = _EXPLANATION_PROMPT.format(
            industry=industry, assets=assets, title=rec.get("title", ""),
            severity=rec.get("severity", ""), summary=entities.get("summary", ""),
            actors=", ".join(entities.get("threat_actors", [])) or "unknown",
            techs=", ".join(entities.get("technologies", [])) or "unknown",
            attacks=", ".join(entities.get("attack_techniques", [])) or "unknown",
            mitre_matches=matches,
        )
        try:
            response = _ollama.chat(model=_OLLAMA_MODEL, messages=[{"role": "user", "content": prompt}])
            raw = response["message"]["content"].strip()
            if "<think>" in raw:
                raw = raw.split("</think>")[-1].strip()
            rec["llm_explanation"] = raw
        except Exception as exc:
            logger.warning("  [rank %d] LLM call failed: %s", rec["rank"], exc)
            rec["llm_explanation"] = ""

    return recommendations


def save_recommendations(profile: Dict[str, Any], recommendations: List[Dict[str, Any]],
                         output_dir: Path = _OUTPUT_DIR) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    org_id   = profile.get("metadata", {}).get("org_id", "unknown")
    industry = profile.get("organizational_context", {}).get("industry_sector", "unknown")
    stamp    = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    out_path = output_dir / f"recommendations_{org_id}_{stamp}.json"
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump({
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "org_id": org_id, "industry": industry,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations,
        }, fh, indent=2, ensure_ascii=False)
    logger.info("Saved recommendations to %s", out_path)
    prune_old_recommendations(org_id, output_dir)
    return out_path


def prune_old_recommendations(org_id: str, output_dir: Path = _OUTPUT_DIR,
                              keep: int = _KEEP_RECOMMENDATION_RUNS) -> None:
    if keep <= 0:
        return
    files = sorted(output_dir.glob(f"recommendations_{org_id}_*.json"), key=lambda p: p.name)
    for old_file in files[:-keep]:
        try:
            old_file.unlink()
            logger.info("Removed old recommendation file: %s", old_file)
        except OSError as exc:
            logger.warning("Could not remove old recommendation file %s: %s", old_file, exc)


def run_recommendation_engine(org_id: str, use_llm: bool = True, top_n: int = _TOP_N,
                               top_n_explain: int = 5, min_score: float = _MIN_SCORE,
                               save_output: bool = True) -> Dict[str, Any]:
    profile    = load_org_profile(org_id)
    cti_events = load_cti_events()
    recs       = compute_recommendations(profile, cti_events, min_score=min_score, top_n=top_n)
    if use_llm and recs:
        recs = generate_llm_explanations(profile, recs, top_n_explain)
    out_file = save_recommendations(profile, recs) if save_output and recs else None
    _print_summary(profile, recs)
    return {
        "org_id":                profile.get("metadata", {}).get("org_id"),
        "industry":              profile.get("organizational_context", {}).get("industry_sector"),
        "total_recommendations": len(recs),
        "recommendations":       recs,
        "output_file":           str(out_file) if out_file else None,
    }


def _safe_str(text: str, max_len: int = 0) -> str:
    """Prepare titles for compact console output without losing UTF-8 text."""
    out = str(text)
    replacements = {
        "â€œ": '"',
        "â€�": '"',
        "â€": '"',
        "â€˜": "'",
        "â€™": "'",
        "â€TM": "'",
        "â€“": "-",
        "â€”": "-",
        "â€¦": "...",
        "Ã©": "é",
        "Ã¨": "è",
        "Ã ": "à",
        "Ã¡": "á",
        "Ã¢": "â",
        "Ã±": "ñ",
        "Ã¶": "ö",
        "Ã¼": "ü",
    }
    for bad, good in replacements.items():
        out = out.replace(bad, good)
    return out[:max_len] if max_len else out


def _print_summary(profile: Dict[str, Any], recommendations: List[Dict[str, Any]]) -> None:
    org_id   = profile.get("metadata", {}).get("org_id", "?")
    industry = profile.get("organizational_context", {}).get("industry_sector", "?")
    print(f"\n{'='*60}\n  {org_id} ({industry}) - {len(recommendations)} recommendations\n{'='*60}")
    if not recommendations:
        print("  No recommendations met the minimum score threshold.\n")
        return
    for rec in recommendations:
        score = rec["score"]
        level = "[HIGH]" if score > 20 else "[MED] " if score > 10 else "[LOW] "
        mitre = " [MITRE]" if rec.get("mitre_bonus", 0) > 0 else ""
        print(f"  #{rec['rank']:02d} {level} {score:.1f}  {rec['severity'].upper()}{mitre}  {_safe_str(rec['title'], 60)}")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="CTI Recommendation Engine")
    parser.add_argument("--org-id",  default=None, help="Organisation ID from Profiling/Instances.")
    parser.add_argument("--all",     action="store_true", help="Process all available org profiles.")
    parser.add_argument("--no-llm",  action="store_true", help="Skip LLM explanation step.")
    parser.add_argument("--top-n",   type=int, default=_TOP_N, help="Max recommendations to return.")
    args = parser.parse_args()

    if args.all:
        for p in list_available_profiles():
            try:
                run_recommendation_engine(p["org_id"], use_llm=not args.no_llm, top_n=args.top_n)
            except Exception as e:
                logger.error("Failed for %s: %s", p["org_id"], e)
    elif args.org_id:
        run_recommendation_engine(args.org_id, use_llm=not args.no_llm, top_n=args.top_n)
    else:
        parser.print_help()
