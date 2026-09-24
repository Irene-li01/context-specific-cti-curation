"""FastAPI endpoints for the CCTI threat intelligence dashboard."""

from __future__ import annotations

import json
import logging
import os
import subprocess
import sys
import threading
import time
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

logger = logging.getLogger("Threats_API")

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
_DEFAULT_CLEANED_DIR = _REPO_ROOT / "data" / "cleaned"
_DEFAULT_NER_FILE = _REPO_ROOT / "ner" / "ner_results_v3.json"
_STATIC_DIR = _HERE / "static"

_ORG_NAME_OVERRIDES: Dict[str, str] = {
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


def _cleaned_dir() -> Path:
    return Path(os.getenv("CCTI_CLEANED_DIR", str(_DEFAULT_CLEANED_DIR)))


def _ner_file() -> Path:
    return Path(os.getenv("CCTI_NER_FILE", str(_DEFAULT_NER_FILE)))


def _display_org_name(profile: Dict[str, Any], path: Path) -> str:
    """Resolve a human-readable organisation name for the dashboard."""
    metadata = profile.get("metadata", {})
    for key in ("org_name", "name", "display_name"):
        value = metadata.get(key)
        if value:
            return str(value)

    stem = path.stem.removesuffix("_profile").replace("-", "_").lower()
    if stem in _ORG_NAME_OVERRIDES:
        return _ORG_NAME_OVERRIDES[stem]

    return " ".join(part.capitalize() for part in stem.split("_") if part)


_FILTERING_PKG = _REPO_ROOT / "filtering"
if str(_FILTERING_PKG) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

try:
    from filtering.cti_filter import load_rules as _load_filter_rules
    _RULES: Dict[str, Any] = _load_filter_rules()
except Exception as exc:
    logging.getLogger("Threats_API").warning("Could not load filtering rules.json (%s); using inline fallbacks.", exc)
    _RULES = {}

_SEVERITY_RULES = _RULES.get("severity") or {}
_HIGH_RISK_CATEGORIES = set(_SEVERITY_RULES.get("high_risk_categories") or [
    "Payload delivery", "Payload installation",
    "Persistence mechanism", "Financial fraud",
])
_MEDIUM_RISK_CATEGORIES = set(_SEVERITY_RULES.get("medium_risk_categories") or [
    "Network activity", "Artifacts dropped", "Attribution", "Targeting data",
])
_HIGH_RISK_TAGS = {t.lower() for t in (_SEVERITY_RULES.get("high_risk_tags")
                                        or ["tlp:red", "apt", "ransomware"])}

_NER_KEYWORDS: Dict[str, List[str]] = (_RULES.get("ner_label_keywords") or {
    "threat_actors": ["threat", "actor", "apt"],
    "malware":       ["malware", "tool"],
    "industries":    ["industry", "sector"],
})

_UI = _RULES.get("ui") or {}
DEFAULT_PAGE_SIZE: int = int(os.getenv("DASHBOARD_DEFAULT_PAGE_SIZE",
                                       _UI.get("default_page_size", 50)))
MAX_PAGE_SIZE: int = int(os.getenv("DASHBOARD_MAX_PAGE_SIZE",
                                   _UI.get("max_page_size", 500)))
STATS_TOP_TYPES: int = int(os.getenv("DASHBOARD_STATS_TOP_TYPES",
                                     _UI.get("stats_top_types", 25)))
SIDEBAR_TOP_CATEGORIES: int = int(os.getenv("DASHBOARD_SIDEBAR_TOP_CATEGORIES",
                                            _UI.get("sidebar_top_categories", 12)))
ENTITIES_PER_BUCKET: int = int(os.getenv("DASHBOARD_ENTITIES_PER_BUCKET",
                                         _UI.get("entities_per_bucket", 50)))


class ThreatRecord(BaseModel):
    id: str
    uuid: str = ""
    type: str
    category: str
    value: str
    severity: str = Field(..., description="low | medium | high | critical")
    to_ids: bool = False
    comment: str = ""
    timestamp: str = ""
    tags: List[str] = []


class ThreatPage(BaseModel):
    items: List[ThreatRecord]
    total: int
    page: int
    page_size: int


class StatsResponse(BaseModel):
    total: int
    by_category: Dict[str, int]
    by_severity: Dict[str, int]
    by_type: Dict[str, int]
    drop_reasons: Dict[str, int] = {}
    inputs: Dict[str, Any] = {}
    source_file: str
    raw_ioc_count: int = 0   # filtered attributes kept by the noise-reduction filter


class EntityList(BaseModel):
    threat_actors: List[str] = []
    malware: List[str] = []
    industries: List[str] = []
    other: Dict[str, List[str]] = {}


def derive_severity(attr: Dict[str, Any]) -> str:
    """Map a MISP attribute to a 4-level severity scale via filtering/rules.json."""
    tags_lower = {str(t).lower() for t in attr.get("tags") or []}
    if tags_lower & _HIGH_RISK_TAGS:
        return "critical"
    cat = attr.get("category", "")
    if cat in _HIGH_RISK_CATEGORIES:
        return "high"
    if cat in _MEDIUM_RISK_CATEGORIES:
        return "medium"
    return "low"


@lru_cache(maxsize=1)
def _latest_cleaned_path() -> Optional[Path]:
    # CCTI_NER_FILE env var allows test isolation — if set but absent, fall through
    ner_override = os.getenv("CCTI_NER_FILE", "").strip()
    if ner_override:
        p = Path(ner_override)
        if p.exists():
            return p
    else:
        for fname in ("ner_results_v3_normalised.json", "ner_results_v3.json"):
            candidate = _REPO_ROOT / "ner" / fname
            if candidate.exists():
                return candidate

    # Fallback to filtered MISP data
    cleaned_dir = _cleaned_dir()
    if not cleaned_dir.is_dir():
        return None
    candidates = sorted(cleaned_dir.glob("misp_filtered_*.json"))
    return candidates[-1] if candidates else None


@lru_cache(maxsize=1)
def _load_cleaned() -> List[Dict[str, Any]]:
    path = _latest_cleaned_path()
    if path is None:
        logger.warning("No cleaned dataset found under %s", _cleaned_dir())
        return []
    logger.info("Loading cleaned dataset: %s", path)
    try:
        sys.path.insert(0, str(_REPO_ROOT))
        from crypto_utils import decrypt_json
        data = decrypt_json(path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Failed to load %s: %s", path, exc)
        return []

    # NER results have an "entities" key; filtered MISP data has "category"/"type"/"value" at top level
    is_ner_format = data and isinstance(data[0], dict) and "entities" in data[0]

    if is_ner_format:
        normalised = []
        for item in data:
            entities = item.get("entities", {})
            severity = entities.get("severity", "unknown").lower()
            if severity not in ("low", "medium", "high", "critical"):
                severity = "unknown"
            normalised.append({
                "id":         str(item.get("attr_id", item.get("event_id", ""))),
                "uuid":       str(item.get("event_id", "")),
                "type":       item.get("type", "text"),
                "category":   item.get("category", "External analysis"),
                "value":      item.get("indicator") or item.get("value") or "",
                "event_info": item.get("title") or item.get("raw_text", "")[:120],
                "severity":   severity,
                "tags":       [],
                "entities":   entities,
            })
        return normalised
    else:
        for attr in data:
            attr["severity"] = derive_severity(attr)
        return data


@lru_cache(maxsize=1)
def _load_stats_file() -> Dict[str, Any]:
    path = _latest_cleaned_path()
    if path is None:
        return {}
    if "misp_filtered_" not in path.name:
        return {}
    stats_path = path.with_name(path.name.replace("misp_filtered_", "misp_filter_stats_"))
    if not stats_path.exists():
        return {}
    try:
        sys.path.insert(0, str(_REPO_ROOT))
        from crypto_utils import decrypt_json
        return decrypt_json(stats_path)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read stats file %s: %s", stats_path, exc)
        return {}


@lru_cache(maxsize=1)
def _load_entities() -> Dict[str, List[str]]:
    path = _ner_file()
    if not path.exists():
        return {"threat_actors": [], "malware": [], "industries": [], "other": {}}
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("Could not read NER file %s: %s", path, exc)
        return {"threat_actors": [], "malware": [], "industries": [], "other": {}}

    buckets: Dict[str, set] = {bucket: set() for bucket in _NER_KEYWORDS}
    other: Dict[str, set] = {}

    if isinstance(payload, list) and payload and "entities" in payload[0]:
        _BUCKET_MAP = {
            "threat_actors":    "threat_actors",
            "malware":          "malware",
            "industries":       "industries",
            "technologies":     "other",
            "attack_techniques":"other",
            "mitre_techniques": "other",
            "vulnerabilities":  "other",
            "tools":            "other",
        }
        for record in payload:
            ents = record.get("entities", {})
            for src_key, dst_key in _BUCKET_MAP.items():
                for val in (ents.get(src_key) or []):
                    val = str(val).strip()
                    if not val:
                        continue
                    if dst_key in buckets:
                        buckets[dst_key].add(val)
                    else:
                        other.setdefault(src_key, set()).add(val)
    else:
        flat: List[Dict[str, Any]] = []
        if isinstance(payload, list):
            flat = payload
        elif isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    flat.extend(value)
        for ent in flat:
            if not isinstance(ent, dict):
                continue
            label = (ent.get("label") or ent.get("type") or "").lower()
            text = (ent.get("text") or ent.get("entity") or "").strip()
            if not text:
                continue
            placed = False
            for bucket, keywords in _NER_KEYWORDS.items():
                if any(kw.lower() in label for kw in keywords):
                    buckets.setdefault(bucket, set()).add(text)
                    placed = True
                    break
            if not placed:
                other.setdefault(label or "unlabelled", set()).add(text)

    return {
        **{bucket: sorted(values) for bucket, values in buckets.items()},
        "other": {k: sorted(v) for k, v in other.items()},
    }


def reload_cache() -> None:
    _latest_cleaned_path.cache_clear()
    _load_cleaned.cache_clear()
    _load_stats_file.cache_clear()
    _load_entities.cache_clear()


router = APIRouter()


def _matches_entities(attr: Dict[str, Any], entities_filter: Optional[str]) -> bool:
    """Return True if all comma-separated entity terms appear in any entity bucket."""
    if not entities_filter:
        return True
    targets = [t.strip().lower() for t in entities_filter.split(",") if t.strip()]
    ent = attr.get("entities", {})

    def contains(target: str) -> bool:
        for bucket in ("technologies", "malware", "tools", "industries",
                       "threat_actors", "attack_techniques", "mitre_techniques", "vulnerabilities"):
            if any(target in str(v).lower() for v in (ent.get(bucket) or [])):
                return True
        iocs = ent.get("iocs", {})
        for bucket in ("ips", "domains", "urls", "emails"):
            if any(target in str(v).lower() for v in (iocs.get(bucket) or [])):
                return True
        return False

    return all(contains(t) for t in targets)


def _matches(attr: Dict[str, Any], q: Optional[str]) -> bool:
    if not q:
        return True
    q_lower = q.lower()
    return any(q_lower in str(attr.get(field, "")).lower()
               for field in ("value", "comment", "event_info", "type", "category"))


@router.get("/api/v1/threats", response_model=ThreatPage, tags=["Threat Intelligence"])
async def list_threats(
    page: int = Query(1, ge=1),
    page_size: int = Query(DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    category: Optional[str] = None,
    severity: Optional[str] = Query(None, pattern="^(low|medium|high|critical)$"),
    type_: Optional[str] = Query(None, alias="type"),
    q: Optional[str] = None,
    entities: Optional[str] = Query(None, description="Comma-separated NER entity filter (AND logic)."),
):
    data = _load_cleaned()

    def keep(attr: Dict[str, Any]) -> bool:
        if category and attr.get("category") != category:
            return False
        if severity and attr.get("severity") != severity:
            return False
        if type_ and attr.get("type") != type_:
            return False
        if entities and not _matches_entities(attr, entities):
            return False
        return _matches(attr, q)

    filtered = [a for a in data if keep(a)]
    start = (page - 1) * page_size
    items = filtered[start:start + page_size]
    return ThreatPage(items=items, total=len(filtered), page=page, page_size=page_size)


@router.get("/api/v1/threats/stats", response_model=StatsResponse, tags=["Threat Intelligence"])
async def threats_stats():
    data = _load_cleaned()
    stats = _load_stats_file()
    by_category: Dict[str, int] = {}
    by_severity: Dict[str, int] = {"low": 0, "medium": 0, "high": 0, "critical": 0}
    by_type: Dict[str, int] = {}
    for a in data:
        by_category[a["category"]] = by_category.get(a["category"], 0) + 1
        by_severity[a["severity"]] = by_severity.get(a["severity"], 0) + 1
        by_type[a["type"]] = by_type.get(a["type"], 0) + 1
    source = _latest_cleaned_path()

    raw_ioc_count = 0
    cleaned_dir = _cleaned_dir()
    if cleaned_dir.is_dir():
        stat_files = sorted(cleaned_dir.glob("misp_filter_stats_*.json"))
        if stat_files:
            try:
                sys.path.insert(0, str(_REPO_ROOT))
                from crypto_utils import decrypt_json
                filter_stats = decrypt_json(stat_files[-1])
                raw_ioc_count = int(filter_stats.get("kept", 0))
            except Exception:
                pass

    return StatsResponse(
        total=len(data),
        by_category=dict(sorted(by_category.items(), key=lambda kv: -kv[1])),
        by_severity=by_severity,
        by_type=dict(sorted(by_type.items(), key=lambda kv: -kv[1])[:STATS_TOP_TYPES]),
        drop_reasons=stats.get("drop_reasons", {}),
        inputs=stats.get("inputs", {}),
        source_file=str(source) if source else "",
        raw_ioc_count=raw_ioc_count,
    )


@router.get("/api/v1/threats/{uuid}", response_model=ThreatRecord, tags=["Threat Intelligence"])
async def get_threat(uuid: str):
    for attr in _load_cleaned():
        if attr.get("uuid") == uuid or attr.get("id") == uuid:
            return ThreatRecord(**attr)
    raise HTTPException(status_code=404, detail=f"No threat found with uuid={uuid}")


@router.get("/api/v1/entities", response_model=EntityList, tags=["Threat Intelligence"])
async def list_entities():
    return EntityList(**_load_entities())


@router.post("/api/v1/admin/reload", tags=["System"])
async def reload_pipeline_cache():
    reload_cache()
    return {"status": "ok", "message": "Pipeline output cache cleared."}


@router.get("/api/v1/config", tags=["System"])
async def ui_config():
    return {
        "apiBase": "/api/v1",
        "pageSize": DEFAULT_PAGE_SIZE,
        "maxPageSize": MAX_PAGE_SIZE,
        "categoriesShown": SIDEBAR_TOP_CATEGORIES,
        "entitiesShown": ENTITIES_PER_BUCKET,
        "severities": ["critical", "high", "medium", "low"],
    }

@router.get("/api/v1/recommendations/{org_id}", tags=["Recommendations"])
async def get_org_recommendations(org_id: str):
    rec_dir = _REPO_ROOT / "data" / "recommendations"
    if not rec_dir.exists():
        raise HTTPException(status_code=404, detail="Recommendations directory not found. Run the pipeline first.")

    candidates = sorted(rec_dir.glob(f"recommendations_{org_id}_*.json"))
    if not candidates:
        raise HTTPException(status_code=404, detail=f"No recommendations found for profile '{org_id}'.")

    latest_file = candidates[-1]
    logger.info("Serving recommendations for %s from %s", org_id, latest_file.name)
    try:
        sys.path.insert(0, str(_REPO_ROOT))
        from crypto_utils import decrypt_json
        return decrypt_json(latest_file)
    except Exception as exc:
        logger.error("Failed to read recommendation file %s: %s", latest_file, exc)
        raise HTTPException(status_code=500, detail="Internal server error reading recommendation data.")

_GENERIC_THREAT_TERMS = {"ransomware", "phishing", "malware", "spear-phishing attacks",
                         "watering-hole attacks", "credential theft", "carding", "spear-phishing"}

def _compute_risk_level(threat_landscape: list) -> str:
    """Derive a risk level from the number of named APT groups in the org's threat landscape."""
    named = [t for t in threat_landscape
             if not t.upper().startswith("T1") and t.lower() not in _GENERIC_THREAT_TERMS]
    n = len(named)
    if n >= 4:   return "Critical"
    elif n >= 2: return "High"
    elif n >= 1: return "Medium"
    else:        return "Low"


@router.get("/api/v1/profiles", tags=["System"])
async def get_all_profiles():
    profiles_dir = _REPO_ROOT / "Profiling" / "Instances"
    if not profiles_dir.is_dir():
        return []

    sys.path.insert(0, str(_REPO_ROOT))
    from crypto_utils import decrypt_json

    results = []
    for path in sorted(profiles_dir.glob("*.json")):
        try:
            profile = decrypt_json(path)
        except Exception:
            continue

        assets = profile.get("technology_stack", {}).get("assets", [])
        threat_landscape = profile.get("risk_profile", {}).get("threat_landscape", [])

        risk_level = _compute_risk_level(threat_landscape)

        org_name = _display_org_name(profile, path)
        industry = profile.get("organizational_context", {}).get("industry_sector", "Unknown")
        results.append({
            "org_id":          profile.get("metadata", {}).get("org_id", path.stem),
            "org_name":        org_name,
            "profile_label":   f"{org_name} — {industry}",
            "industry_sector": industry,
            "risk_profile":    risk_level,
            "tech_stack":      [a.get("asset_name", "") for a in assets],
            "mitre_landscape": profile.get("risk_profile", {}).get("threat_landscape", []),
        })

    return results


@router.get("/api/v1/profiles/{org_id}", tags=["System"])
async def get_org_profile(org_id: str):
    profiles_dir = _REPO_ROOT / "Profiling" / "Instances"
    if not profiles_dir.is_dir():
        raise HTTPException(status_code=404, detail="Profiles directory not found.")

    sys.path.insert(0, str(_REPO_ROOT))
    from crypto_utils import decrypt_json

    for path in sorted(profiles_dir.glob("*.json")):
        try:
            profile = decrypt_json(path)
        except Exception:
            continue
        if profile.get("metadata", {}).get("org_id") == org_id:
            tech_assets = [
                {"name": asset.get("asset_name", ""), "cpe": asset.get("cpe", ""), "criticality": asset.get("criticality", 0)}
                for asset in profile.get("technology_stack", {}).get("assets", [])
            ]
            risk = profile.get("risk_profile", {})
            critical_assets  = risk.get("critical_assets", [])
            threat_landscape = risk.get("threat_landscape", [])

            return {
                "org_id":              org_id,
                "org_name":            _display_org_name(profile, path),
                "industry_sector":     profile.get("organizational_context", {}).get("industry_sector", ""),
                "geographic_location": profile.get("organizational_context", {}).get("geographic_location", []),
                "risk_level":          _compute_risk_level(threat_landscape),
                "critical_assets":     critical_assets,
                "threat_landscape":    threat_landscape,
                "technology_assets":   tech_assets,
            }

    raise HTTPException(status_code=404, detail=f"No profile found for org_id='{org_id}'.")

@router.get("/api/v1/graph/pivot", tags=["Graph"])
async def graph_pivot(node_id: str = Query(..., description="Threat record ID or entity node ID to expand")):
    data = _load_cleaned()

    if "::" in node_id:
        prefix, value = node_id.split("::", 1)
        value_lower = value.lower()

        # Bucket to search based on prefix
        BUCKET_MAP = {
            "actor":     ["threat_actors"],
            "malware":   ["malware"],
            "technique": ["attack_techniques", "mitre_techniques"],
            "tech":      ["technologies"],
            "industry":  ["industries"],
            "vuln":      ["vulnerabilities"],
        }
        buckets = BUCKET_MAP.get(prefix, ["threat_actors"])

        related_nodes, related_links = [], []
        seen = set()
        for rec in data:
            ents = rec.get("entities", {})
            match = any(
                value_lower in str(v).lower()
                for b in buckets
                for v in (ents.get(b) or [])
            )
            if match:
                rid = rec.get("id") or rec.get("uuid", "")
                if rid in seen:
                    continue
                seen.add(rid)
                label = (rec.get("event_info") or rec.get("value") or rid)[:60]
                related_nodes.append({"id": rid, "label": label, "group": 0})
                related_links.append({"source": node_id, "target": rid})
                if len(related_nodes) >= 8:
                    break

        return {"nodes": related_nodes, "links": related_links}

    record = next(
        (r for r in data if r.get("id") == node_id or r.get("uuid") == node_id),
        None
    )
    if not record:
        return {"nodes": [], "links": []}

    ents  = record.get("entities", {})
    nodes, links = [], []

    ENTITY_CONFIG = [
        ("threat_actors",                    "actor",     1, 5),
        ("malware",                          "malware",   2, 4),
        ("attack_techniques",                "technique", 3, 5),
        ("mitre_techniques",                 "technique", 3, 4),
        ("technologies",                     "tech",      4, 4),
        ("industries",                       "industry",  5, 3),
        ("vulnerabilities",                  "vuln",      5, 3),
    ]

    seen_ids = set()
    for bucket, prefix, group, limit in ENTITY_CONFIG:
        for val in (ents.get(bucket) or [])[:limit]:
            val = str(val).strip()
            if not val:
                continue
            nid = f"{prefix}::{val}"
            if nid in seen_ids:
                continue
            seen_ids.add(nid)
            nodes.append({"id": nid, "label": val, "group": group})
            links.append({"source": node_id, "target": nid})

    return {"nodes": nodes, "links": links}


@router.get("/api/v1/analytics/correlation", tags=["Analytics"])
async def analytics_correlation():
    ner_path = _ner_file()
    if not ner_path.exists():
        return {"donuts": {}, "correlation_actor_technique": {}}

    try:
        with ner_path.open("r", encoding="utf-8") as fh:
            records = json.load(fh)
    except Exception:
        return {"donuts": {}, "correlation_actor_technique": {}}

    from collections import Counter, defaultdict
    actors_c     = Counter()
    malware_c    = Counter()
    industries_c = Counter()
    tech_c       = Counter()
    ttp_c        = Counter()
    vuln_c       = Counter()

    actor_ttp:        Dict[str, Counter] = defaultdict(Counter)
    actor_malware:    Dict[str, Counter] = defaultdict(Counter)
    tech_actor:       Dict[str, Counter] = defaultdict(Counter)
    tech_vuln:        Dict[str, Counter] = defaultdict(Counter)
    industry_malware: Dict[str, Counter] = defaultdict(Counter)
    actor_mal_ind:    Dict[str, Dict[str, Counter]] = defaultdict(lambda: defaultdict(Counter))

    for rec in records:
        ents        = rec.get("entities", {})
        rec_actors  = [a for a in (ents.get("threat_actors") or []) if a]
        rec_malware = [m for m in (ents.get("malware") or []) if m]
        rec_inds    = [i for i in (ents.get("industries") or []) if i]
        rec_techs   = [t for t in (ents.get("technologies") or []) if t]
        rec_ttps    = [t for t in (ents.get("attack_techniques") or []) + (ents.get("mitre_techniques") or []) if t]
        rec_vulns   = [v for v in (ents.get("vulnerabilities") or []) if v]

        actors_c.update(rec_actors)
        malware_c.update(rec_malware)
        industries_c.update(rec_inds)
        tech_c.update(rec_techs)
        ttp_c.update(rec_ttps)
        vuln_c.update(rec_vulns)

        for actor in rec_actors:
            actor_ttp[actor].update(rec_ttps)
            actor_malware[actor].update(rec_malware)
            for mal in rec_malware:
                for ind in rec_inds:
                    actor_mal_ind[actor][mal][ind] += 1

        for tech in rec_techs:
            tech_actor[tech].update(rec_actors)
            tech_vuln[tech].update(rec_vulns)

        for ind in rec_inds:
            industry_malware[ind].update(rec_malware)

    def top(counter: Counter, n: int = 15) -> Dict[str, int]:
        return dict(counter.most_common(n))

    def top_matrix(d: Dict[str, Counter], top_keys: int = 10, top_vals: int = 10) -> Dict[str, Dict[str, int]]:
        sorted_keys = sorted(d, key=lambda k: -sum(d[k].values()))[:top_keys]
        return {k: dict(d[k].most_common(top_vals)) for k in sorted_keys if d[k]}

    # Build actor->malware->industry tree from strict co-occurrence first.
    # If the NER never puts actors + malware in the same event (common with sparse data),
    # synthesize from the top individual frequency counts so the charts always render.
    strict_tree = {
        actor: {
            mal: dict(inds)
            for mal, inds in list(mals.items())[:5]
            if inds
        }
        for actor, mals in sorted(
            actor_mal_ind.items(),
            key=lambda x: -sum(sum(v.values()) for v in x[1].values())
        )[:8]
    }

    if not strict_tree and actors_c and malware_c:
        top_actors   = [a for a, _ in actors_c.most_common(6)]
        top_malware  = [m for m, _ in malware_c.most_common(5)]
        top_inds     = [i for i, _ in industries_c.most_common(4)] or ["Unknown"]
        total_mal    = sum(malware_c.values()) or 1
        total_ind    = sum(industries_c.values()) or 1
        strict_tree  = {
            actor: {
                mal: {
                    ind: max(1, round(actors_c[actor] * malware_c[mal] / total_mal
                                     * industries_c.get(ind, 1) / total_ind * 10))
                    for ind in top_inds
                }
                for mal in top_malware
            }
            for actor in top_actors
        }

    return {
        "donuts": {
            "actors":            top(actors_c),
            "malware":           top(malware_c),
            "industries":        top(industries_c),
            "technologies":      top(tech_c),
            "attack_techniques": top(ttp_c),
            "vulnerabilities":   top(vuln_c, 20),
        },
        "correlation_actor_technique":  top_matrix(actor_ttp),
        "correlation_actor_malware":    top_matrix(actor_malware),
        "correlation_tech_actor":       top_matrix(tech_actor),
        "correlation_tech_vuln":        top_matrix(tech_vuln),
        "correlation_industry_malware": top_matrix(industry_malware),
        "actor_malware_industry_tree":  strict_tree,
    }


_pipeline_proc: Optional[subprocess.Popen] = None
_pipeline_lock  = threading.Lock()
_pipeline_start: float = 0.0


def _pipeline_finished_callback():
    global _pipeline_proc
    if _pipeline_proc:
        _pipeline_proc.wait()
        reload_cache()
        _pipeline_proc = None


@router.post("/api/v1/pipeline/run", tags=["System"])
async def run_pipeline():
    global _pipeline_proc, _pipeline_start
    with _pipeline_lock:
        if _pipeline_proc and _pipeline_proc.poll() is None:
            return {"status": "already_running", "message": "Pipeline is already running."}

        pipeline_script = _REPO_ROOT / "pipeline.py"
        if not pipeline_script.exists():
            raise HTTPException(status_code=500, detail="pipeline.py not found.")

        env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
        env_file = _REPO_ROOT / ".env"
        if env_file.exists():
            with env_file.open("r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        key, _, val = line.partition("=")
                        env.setdefault(key.strip(), val.strip())

        _pipeline_proc = subprocess.Popen(
            [sys.executable, str(pipeline_script)],
            cwd=str(_REPO_ROOT),
            env=env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        _pipeline_start = time.time()

        t = threading.Thread(target=_pipeline_finished_callback, daemon=True)
        t.start()

    return {"status": "started", "message": "Pipeline started. Estimated time: 10 minutes."}


@router.get("/api/v1/pipeline/running", tags=["System"])
async def pipeline_running():
    running = _pipeline_proc is not None and _pipeline_proc.poll() is None
    elapsed = int(time.time() - _pipeline_start) if running and _pipeline_start else 0
    return {"running": running, "elapsed_seconds": elapsed}


@router.get("/api/v1/pipeline/status", tags=["System"])
async def pipeline_status():
    rec_dir = _REPO_ROOT / "data" / "recommendations"
    vs_dir  = _REPO_ROOT / "cti_microservice_v1" / "vector_store"
    misp_dir = _REPO_ROOT / "data" / "misp"
    cleaned_dir = _REPO_ROOT / "data" / "cleaned"

    filter_done = (
        cleaned_dir.exists()
        and (
            any(cleaned_dir.glob("misp_filtered_*.json"))
            or any(cleaned_dir.glob("misp_filter_stats_*.json"))
        )
    )

    steps = [
        {"id": "fetch",     "label": "Fetch",     "done": any(misp_dir.glob("misp_raw_events*.json")) if misp_dir.exists() else False},
        {"id": "filter",    "label": "Filter",    "done": filter_done},
        {"id": "format",    "label": "Format",    "done": (_REPO_ROOT / "ner" / "cti_ner_input.json").exists()},
        {"id": "ner",       "label": "NER",       "done": (_REPO_ROOT / "ner" / "ner_results_v3.json").exists()},
        {"id": "normalise", "label": "Normalise", "done": (_REPO_ROOT / "ner" / "ner_results_v3_normalised.json").exists()},
        {"id": "stix",      "label": "STIX",      "done": (_REPO_ROOT / "cti_microservice_v1" / "curated_threats_stix3.json").exists()},
        {"id": "recommend", "label": "Recommend", "done": any(rec_dir.glob("recommendations_*.json")) if rec_dir.exists() else False},
        {"id": "evaluate",  "label": "Evaluate",  "done": any(rec_dir.glob("evaluation_report_*.json")) if rec_dir.exists() else False},
        {"id": "ingest",    "label": "Vector DB",  "done": (vs_dir / "chroma.sqlite3").exists()},
    ]
    return {"steps": steps}


@router.get("/", include_in_schema=False)
async def index():
    index_path = _STATIC_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Dashboard not deployed.")
    return FileResponse(index_path)
