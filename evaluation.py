"""
Evaluates recommendation engine outputs across all org profiles.
Checks cross-org differentiation, MITRE coverage, and component contribution.

Usage:
    python evaluation.py
    python evaluation.py --profiles ORG-EN-001 ORG-SB-99
    python evaluation.py --top-n 20 --min-score 0.0
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from recommendation_engine import (
    load_org_profile, load_cti_events, list_available_profiles,
    _profile_to_document, _cti_event_to_document,
    _compute_cpe_boost, compute_mitre_score, explain_mitre_matches,
    SEVERITY_WEIGHTS, _OUTPUT_DIR,
)

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger("Evaluation")

_REPORT_DIR = _OUTPUT_DIR


def _name_from_profile_filename(filename: str) -> str:
    """Convert profile filenames into report-friendly organisation names."""
    stem = Path(filename).stem.replace("_profile", "")
    known_names = {
        "commbank": "CommBank",
        "electranet": "ElectraNet",
        "foodfirst": "FoodFirst",
        "freshmart": "FreshMart",
        "harboureats": "HarbourEats",
        "meddata": "MedData",
        "payswift": "PaySwift",
        "pharmacoaus": "PharmaCoAus",
        "powerco": "PowerCo",
        "royaladelaidehospital": "Royal Adelaide Hospital",
        "securebank": "SecureBank",
        "sunpower": "SunPower",
    }
    return known_names.get(stem, stem.replace("_", " ").title())


def _org_label(org_id: str, metrics: Dict[str, Any]) -> str:
    """Build a readable label without changing the stable organisation ID."""
    name = metrics.get(org_id, {}).get("org_name") or org_id
    industry = metrics.get(org_id, {}).get("industry", "")
    if name == org_id:
        return f"{org_id} ({industry})" if industry else org_id
    return f"{name} ({org_id}, {industry})" if industry else f"{name} ({org_id})"


def score_with_breakdown(profile: Dict[str, Any], cti_events: List[Dict[str, Any]],
                          top_n: int = 20, min_score: float = 0.0) -> List[Dict[str, Any]]:
    """Score all CTI events against an org profile, returning per-component breakdowns."""
    org_doc  = _profile_to_document(profile)
    cti_docs = [_cti_event_to_document(ev.get("entities", {})) for ev in cti_events]
    non_empty       = [i for i, d in enumerate(cti_docs) if d.strip()]
    events_filtered = [cti_events[i] for i in non_empty]
    if not non_empty:
        return []

    corpus       = [org_doc] + [cti_docs[i] for i in non_empty]
    vectorizer   = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(corpus)
    similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    results = []
    for i, event in enumerate(events_filtered):
        entities    = event.get("entities", {})
        base_sim    = float(similarities[i])
        cpe_boost   = _compute_cpe_boost(profile, entities)
        mitre_bonus = compute_mitre_score(profile, entities)
        severity    = entities.get("severity", "unknown").lower()
        multiplier  = SEVERITY_WEIGHTS.get(severity, 0.5)
        score_base  = (base_sim + cpe_boost) * multiplier * 100
        score_full  = (base_sim + cpe_boost + mitre_bonus) * multiplier * 100

        if score_full < min_score:
            continue

        results.append({
            "event_id": event.get("event_id", ""), "title": event.get("title", "")[:80],
            "severity": severity, "cosine_sim": round(base_sim, 4),
            "cpe_boost": round(cpe_boost, 4), "mitre_bonus": round(mitre_bonus, 4),
            "severity_mult": multiplier, "score_no_mitre": round(score_base, 3),
            "score_with_mitre": round(score_full, 3),
            "mitre_matches": explain_mitre_matches(profile, entities),
        })

    results.sort(key=lambda r: r["score_with_mitre"], reverse=True)
    for rank, r in enumerate(results[:top_n], start=1):
        r["rank"] = rank
    return results[:top_n]


def overlap_at_k(results_a: List[Dict], results_b: List[Dict], k: int = 10) -> float:
    """Jaccard overlap between top-k event IDs of two result lists. 0=different, 1=identical."""
    ids_a = {r["event_id"] for r in results_a[:k]}
    ids_b = {r["event_id"] for r in results_b[:k]}
    if not ids_a and not ids_b:
        return 1.0
    return len(ids_a & ids_b) / len(ids_a | ids_b)


def score_lift_from_mitre(results: List[Dict]) -> Dict[str, Any]:
    """Measure how much the MITRE bonus changed scores and ranking."""
    if not results:
        return {}
    lifts = [r["score_with_mitre"] - r["score_no_mitre"] for r in results]
    no_mitre_order = sorted(results, key=lambda r: r["score_no_mitre"], reverse=True)
    no_mitre_ranks = {r["event_id"]: i + 1 for i, r in enumerate(no_mitre_order)}
    rank_changes = sum(1 for r in results if abs(r["rank"] - no_mitre_ranks.get(r["event_id"], r["rank"])) > 0)
    return {
        "mean_lift": round(float(np.mean(lifts)), 3),
        "max_lift":  round(float(np.max(lifts)),  3),
        "events_lifted": sum(1 for l in lifts if l > 0),
        "rank_changes":  rank_changes,
    }


def component_contribution(results: List[Dict]) -> Dict[str, float]:
    """Average % contribution of TF-IDF, CPE, and MITRE to the raw score."""
    if not results:
        return {}
    rows = []
    for r in results:
        total = r["cosine_sim"] + r["cpe_boost"] + r["mitre_bonus"]
        if total == 0:
            continue
        rows.append({"cosine_pct": r["cosine_sim"] / total * 100,
                     "cpe_pct": r["cpe_boost"] / total * 100,
                     "mitre_pct": r["mitre_bonus"] / total * 100})
    if not rows:
        return {"cosine_avg_pct": 100.0, "cpe_avg_pct": 0.0, "mitre_avg_pct": 0.0}
    return {
        "cosine_avg_pct": round(float(np.mean([r["cosine_pct"] for r in rows])), 1),
        "cpe_avg_pct":    round(float(np.mean([r["cpe_pct"]    for r in rows])), 1),
        "mitre_avg_pct":  round(float(np.mean([r["mitre_pct"]  for r in rows])), 1),
    }


def mitre_coverage(results: List[Dict]) -> Dict[str, Any]:
    """Fraction of top-N results with at least one MITRE match and most-matched techniques."""
    total      = len(results)
    with_mitre = sum(1 for r in results if r["mitre_bonus"] > 0)
    matched    = [m.get("cti_technique", "") for r in results for m in r.get("mitre_matches", []) if m.get("cti_technique")]
    return {
        "total_results":      total,
        "with_mitre_match":   with_mitre,
        "coverage_pct":       round(with_mitre / total * 100, 1) if total else 0.0,
        "top_techniques_hit": Counter(matched).most_common(5),
    }


def severity_distribution(results: List[Dict]) -> Dict[str, int]:
    """Count results per severity tier."""
    return dict(Counter(r["severity"] for r in results))


def _fmt_table(headers: List[str], rows: List[List[str]], col_width: int = 16) -> str:
    """Build a plain-text fixed-width table."""
    def pad(s: Any, w: int) -> str:
        return str(s)[:w].ljust(w)
    sep = "+" + "+".join("-" * (col_width + 2) for _ in headers) + "+"
    lines = [sep, "|" + "|".join(f" {pad(h, col_width)} " for h in headers) + "|", sep]
    for row in rows:
        lines.append("|" + "|".join(f" {pad(c, col_width)} " for c in row) + "|")
    lines.append(sep)
    return "\n".join(lines)


def build_text_report(org_ids: List[str], cti_event_count: int, top_n: int,
                       profile_results: Dict[str, List[Dict]], metrics: Dict[str, Any],
                       overlap_matrix: Dict[str, float]) -> str:
    """Build the full human-readable evaluation report."""
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    lines = [
        "=" * 70,
        f"  CTI Recommendation Engine — Evaluation Report",
        f"  Generated      : {stamp}",
        f"  Organisations  : {', '.join(_org_label(org_id, metrics) for org_id in org_ids)}",
        f"  CTI events     : {cti_event_count}  |  Top-N: {top_n}",
        "=" * 70,
        "", f"SECTION 1 — Top-{min(10, top_n)} recommendations per organisation", "-" * 70,
    ]

    for org_id, results in profile_results.items():
        lines += [f"\n  {_org_label(org_id, metrics)}", ""]
        rows = [[str(r["rank"]), f"{r['score_with_mitre']:.1f}", r["severity"],
                 "yes" if r["mitre_bonus"] > 0 else "-", r["title"][:30]] for r in results[:10]]
        lines.append(_fmt_table(["Rank", "Score", "Severity", "MITRE", "Title"], rows, col_width=14))

    lines += ["", "SECTION 2 — Cross-organisation differentiation (Overlap@10)", "-" * 70, ""]
    if overlap_matrix:
        for pair, overlap in overlap_matrix.items():
            a, b = pair.split("_vs_")
            judgment = "PASS — engine differentiates well" if overlap < 0.3 \
                else "WARN — high overlap, profiles may need more distinct terms"
            lines += [f"  {_org_label(a, metrics)}  vs  {_org_label(b, metrics)}",
                      f"    Overlap@10 : {overlap:.2%}  [{judgment}]", ""]
    else:
        lines.append("  (Need 2+ organisations to compute overlap)")

    lines += ["", "SECTION 3 — MITRE ATT&CK matching contribution", "-" * 70, ""]
    for org_id, m in metrics.items():
        lift    = m.get("score_lift",        {})
        contrib = m.get("component_contrib", {})
        cov     = m.get("mitre_coverage",    {})
        lines.append(f"  {_org_label(org_id, metrics)}")
        if lift.get("events_lifted", 0) > 0:
            lines += [
                f"    Events improved : {lift['events_lifted']} / {m.get('total_results', 0)}",
                f"    Mean lift       : +{lift['mean_lift']} pts  |  Max: +{lift['max_lift']} pts",
                f"    MITRE coverage  : {cov.get('coverage_pct', 0)}% of top-{top_n}",
                f"    Score breakdown : cosine={contrib.get('cosine_avg_pct', 0):.1f}%  "
                f"cpe={contrib.get('cpe_avg_pct', 0):.1f}%  mitre={contrib.get('mitre_avg_pct', 0):.1f}%",
            ]
        else:
            lines.append("    No MITRE bonus applied. Check threat_landscape entries in the org profile.")
        lines.append("")

    lines += ["SECTION 4 — Severity distribution", "-" * 70, ""]
    sev_rows = [
        [_org_label(org_id, metrics)] + [str(severity_distribution(r).get(s, 0)) for s in ["critical", "high", "medium", "low", "unknown"]]
        for org_id, r in profile_results.items()
    ]
    lines.append(_fmt_table(["Organisation", "Critical", "High", "Medium", "Low", "Unknown"], sev_rows, col_width=32))

    lines += ["", "SECTION 5 — Key findings", "-" * 70, ""]
    for pair, overlap in overlap_matrix.items():
        a, b = pair.split("_vs_")
        tag  = "PASS" if overlap < 0.3 else "WARN"
        lines.append(f"  [{tag}] Differentiation: top-10 overlap between {_org_label(a, metrics)} "
                     f"and {_org_label(b, metrics)} is {overlap:.0%}. "
                     + ("The engine produces sector-specific results." if overlap < 0.3
                        else "Consider adding more sector-specific terms to org profiles."))
    for org_id, m in metrics.items():
        lift = m.get("score_lift", {})
        cov  = m.get("mitre_coverage", {})
        if lift.get("events_lifted", 0) > 0:
            lines.append(f"  [PASS] MITRE matching ({_org_label(org_id, metrics)}): improved {lift['events_lifted']} events, "
                         f"max lift +{lift['max_lift']} pts, {cov.get('coverage_pct', 0)}% of top-N matched.")
        else:
            lines.append(f"  [NOTE] MITRE matching ({org_id}): no bonus applied. Review threat_landscape entries.")

    lines += ["", "=" * 70, "  End of report", "=" * 70, ""]
    return "\n".join(lines)


def run_evaluation(org_ids: Optional[List[str]] = None, top_n: int = 20,
                   min_score: float = 0.0, save: bool = True) -> Dict[str, Any]:
    """Run the full evaluation across all org profiles and optionally save reports."""
    if not org_ids:
        available = list_available_profiles()
        if not available:
            print("No organisation profiles found in Profiling/Instances/")
            return {}
        org_ids = [p["org_id"] for p in available]
    profile_summaries = {p["org_id"]: p for p in list_available_profiles()}

    print(f"\n{'='*65}\n  CTI Evaluation — {len(org_ids)} org(s)\n{'='*65}\n")

    try:
        cti_events = load_cti_events()
    except FileNotFoundError as exc:
        print(f"  ERROR: {exc}")
        return {}
    print(f"  Loaded {len(cti_events)} events\n")

    profile_results: Dict[str, List[Dict]] = {}
    metrics:         Dict[str, Any]        = {}

    for org_id in org_ids:
        print(f"  Scoring {org_id}...")
        try:
            profile = load_org_profile(org_id)
        except FileNotFoundError as exc:
            print(f"    SKIP: {exc}")
            continue
        results  = score_with_breakdown(profile, cti_events, top_n=top_n, min_score=min_score)
        industry = profile.get("organizational_context", {}).get("industry_sector", "")
        summary = profile_summaries.get(org_id, {})
        org_name = (
            profile.get("metadata", {}).get("org_name")
            or profile.get("metadata", {}).get("organisation_name")
            or summary.get("organisation_name")
            or _name_from_profile_filename(summary.get("filename", ""))
        )
        profile_results[org_id] = results
        metrics[org_id] = {
            "org_name":          org_name,
            "industry":          industry,
            "total_results":     len(results),
            "severity_dist":     severity_distribution(results),
            "score_lift":        score_lift_from_mitre(results),
            "component_contrib": component_contribution(results),
            "mitre_coverage":    mitre_coverage(results),
        }
        if results:
            top = results[0]
            print(f"    Top: score={top['score_with_mitre']:.2f}  sev={top['severity']}  \"{top['title'][:50]}\"")
        print(f"    MITRE coverage: {metrics[org_id]['mitre_coverage'].get('coverage_pct', 0)}% of top-{top_n}\n")

    if not profile_results:
        print("  No results to report.")
        return {}

    org_list = list(profile_results.keys())
    overlap_matrix: Dict[str, float] = {
        f"{org_list[i]}_vs_{org_list[j]}": round(overlap_at_k(profile_results[org_list[i]], profile_results[org_list[j]]), 4)
        for i in range(len(org_list)) for j in range(i + 1, len(org_list))
    }

    report_text = build_text_report(org_ids, len(cti_events), top_n, profile_results, metrics, overlap_matrix)
    print(report_text)

    payload: Dict[str, Any] = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "org_ids": org_ids, "cti_event_count": len(cti_events),
        "top_n": top_n, "overlap_matrix": overlap_matrix,
        "per_org": {oid: {"metrics": metrics[oid], "top_results": r} for oid, r in profile_results.items()},
    }

    if save:
        _REPORT_DIR.mkdir(parents=True, exist_ok=True)
        stamp     = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        json_path = _REPORT_DIR / f"evaluation_report_{stamp}.json"
        txt_path  = _REPORT_DIR / f"evaluation_report_{stamp}.txt"
        with json_path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        with txt_path.open("w", encoding="utf-8") as fh:
            fh.write(report_text)
        print(f"  Reports saved:\n    JSON : {json_path}\n    TXT  : {txt_path}\n{'='*65}\n")

    return payload


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate CTI recommendation engine outputs.")
    parser.add_argument("--profiles", nargs="+", default=None, metavar="ORG_ID",
                        help="Org IDs to evaluate (default: all available).")
    parser.add_argument("--top-n",     type=int,   default=20,  help="Recommendations per org to analyse.")
    parser.add_argument("--min-score", type=float, default=0.0, help="Minimum score threshold.")
    parser.add_argument("--no-save",   action="store_true",     help="Do not write report files.")
    args = parser.parse_args()
    run_evaluation(org_ids=args.profiles, top_n=args.top_n, min_score=args.min_score, save=not args.no_save)
