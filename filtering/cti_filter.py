"""
CTI noise-reduction filter. Strips irrelevant/duplicate MISP attributes,
normalises text fields, and writes a clean JSON dataset for downstream use.

Inputs:  data/*.json, data/misp/*.json, data/misp/*.xlsx (all loaded together)
Outputs: data/cleaned/misp_filtered_<DATE>.json
         data/cleaned/misp_filter_stats_<DATE>.json
"""

from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import logging
import os
import re
import sys
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple


logging.basicConfig(
    level=os.getenv("CTI_FILTER_LOG_LEVEL", "INFO"),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("CTI_Filter")


RULES_PATH = Path(
    os.getenv("CTI_FILTER_RULES_PATH", str(Path(__file__).with_name("rules.json")))
)

_FALLBACK_RULES: Dict[str, Any] = {
    "noise_categories": ["Person", "Internal reference", "Support Tool"],
    "noise_type_prefixes": [
        "whois-", "email-subject", "email-message-id", "email-header",
        "email-reply-to", "phone-number", "comment", "named pipe",
    ],
    "ip_types": ["ip-src", "ip-dst", "ip-src|port", "ip-dst|port"],
    "url_like_types": ["url", "uri", "domain", "hostname"],
    "hash_types": ["md5", "sha1", "sha224", "sha256", "sha384", "sha512",
                   "ssdeep", "imphash", "tlsh"],
    "defang_patterns": [
        {"pattern": r"\bhxxp(s?)://", "replacement": r"http\1://", "ignore_case": True},
        {"pattern": r"\bfxp://",      "replacement": "ftp://",     "ignore_case": True},
        {"pattern": r"\[\.\]",        "replacement": ".",          "ignore_case": False},
        {"pattern": r"\[:\]",         "replacement": ":",          "ignore_case": False},
        {"pattern": r"\[@\]",         "replacement": "@",          "ignore_case": False},
        {"pattern": r"\(dot\)",       "replacement": ".",          "ignore_case": True},
        {"pattern": r"\(at\)",        "replacement": "@",          "ignore_case": True},
    ],
    "filter_defaults": {
        "min_value_length": 3,
        "max_value_length": 4096,
        "require_to_ids": False,
        "drop_private_ips": True,
        "dedup_strategy": "latest",
    },
}


def load_rules(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read rules.json; merge in fallbacks for any missing keys."""
    target = path or RULES_PATH
    if target.is_file():
        try:
            with target.open("r", encoding="utf-8") as fh:
                user_rules = json.load(fh)
            merged = {**_FALLBACK_RULES, **{k: v for k, v in user_rules.items()
                                            if not k.startswith("_")}}
            merged["filter_defaults"] = {
                **_FALLBACK_RULES["filter_defaults"],
                **(user_rules.get("filter_defaults") or {}),
            }
            return merged
        except (OSError, json.JSONDecodeError) as exc:
            logger.warning("Could not read %s (%s); using built-in fallback rules.",
                           target, exc)
    else:
        logger.info("No rules.json at %s; using built-in fallback rules.", target)
    return _FALLBACK_RULES


_RULES = load_rules()

DEFAULT_NOISE_CATEGORIES: frozenset[str] = frozenset(_RULES["noise_categories"])
DEFAULT_NOISE_TYPE_PREFIXES: frozenset[str] = frozenset(_RULES["noise_type_prefixes"])
IP_TYPES: frozenset[str] = frozenset(_RULES["ip_types"])
URL_LIKE_TYPES: frozenset[str] = frozenset(_RULES["url_like_types"])
HASH_TYPES: frozenset[str] = frozenset(_RULES["hash_types"])

_filter_defaults = _RULES["filter_defaults"]


@dataclass(frozen=True)
class FilterConfig:
    noise_categories: frozenset[str] = DEFAULT_NOISE_CATEGORIES
    noise_type_prefixes: frozenset[str] = DEFAULT_NOISE_TYPE_PREFIXES
    require_to_ids: bool = _filter_defaults["require_to_ids"]
    drop_private_ips: bool = _filter_defaults["drop_private_ips"]
    min_value_length: int = _filter_defaults["min_value_length"]
    max_value_length: int = _filter_defaults["max_value_length"]
    dedup_strategy: str = _filter_defaults["dedup_strategy"]


def _compile_defang_patterns(spec: List[Dict[str, Any]]) -> List[Tuple[re.Pattern[str], str]]:
    compiled: List[Tuple[re.Pattern[str], str]] = []
    for rule in spec:
        flags = re.IGNORECASE if rule.get("ignore_case") else 0
        try:
            compiled.append((re.compile(rule["pattern"], flags), rule["replacement"]))
        except (re.error, KeyError) as exc:
            logger.warning("Skipping invalid defang rule %r: %s", rule, exc)
    return compiled


_DEFANG_PATTERNS: List[Tuple[re.Pattern[str], str]] = _compile_defang_patterns(
    _RULES.get("defang_patterns", _FALLBACK_RULES["defang_patterns"])
)

# Strip ASCII control chars (also XML-illegal in xlsx).
_CONTROL_CHARS_RE = re.compile(r"[\x00-\x08\x0b-\x0c\x0e-\x1f]")


def _refang(text: str) -> str:
    for pat, repl in _DEFANG_PATTERNS:
        text = pat.sub(repl, text)
    return text


def _normalise_text(text: str) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _CONTROL_CHARS_RE.sub("", text)
    return text.strip()


def _is_private_ip(value: str) -> bool:
    candidate = value.split("|", 1)[0]  # strip "|port" suffix if present
    try:
        ip = ipaddress.ip_address(candidate)
    except ValueError:
        return False
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_multicast


def _is_valid_ip(value: str) -> bool:
    candidate = value.split("|", 1)[0]
    try:
        ipaddress.ip_address(candidate)
        return True
    except ValueError:
        return False


def _is_valid_hash(value: str, hash_type: str) -> bool:
    """Length-based check only — not cryptographic verification."""
    expected_lengths = {
        "md5": 32, "sha1": 40, "sha224": 56, "sha256": 64,
        "sha384": 96, "sha512": 128,
    }
    expected = expected_lengths.get(hash_type)
    if expected is None:
        return bool(value)  # ssdeep / imphash / tlsh — accept any non-empty
    return len(value) == expected and bool(re.fullmatch(r"[A-Fa-f0-9]+", value))


def _canonical(attr: Dict[str, Any], event: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    event = event or attr.get("Event") or {}
    tags = attr.get("Tag") or []
    return {
        "id": str(attr.get("id", "")),
        "uuid": attr.get("uuid", ""),
        "type": attr.get("type", "") or "",
        "category": attr.get("category", "") or "",
        "value": attr.get("value", "") or "",
        "to_ids": bool(attr.get("to_ids", False)),
        "comment": attr.get("comment", "") or "",
        "timestamp": str(attr.get("timestamp", "") or ""),
        "event_id": str(event.get("id") or attr.get("event_id") or ""),
        "event_info": event.get("info", "") or "",
        "tags": [t.get("name", "") for t in tags if isinstance(t, dict)],
    }


def _load_misp_event_json(path: Path) -> Iterable[Dict[str, Any]]:
    logger.info("Loading MISP event JSON: %s", path)
    try:
        with path.open("r", encoding="utf-8") as fh:
            payload = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Invalid JSON in %s: %s", path, exc)
        return

    events = payload.get("response") if isinstance(payload, dict) else payload
    if not isinstance(events, list):
        logger.warning("Unexpected payload shape in %s; skipping.", path)
        return

    for wrapper in events:
        event = wrapper.get("Event", wrapper) if isinstance(wrapper, dict) else {}
        for attr in event.get("Attribute", []) or []:
            yield _canonical(attr, event)


def _load_misp_xlsx(path: Path) -> Iterable[Dict[str, Any]]:
    logger.info("Loading MISP xlsx: %s", path)
    try:
        from openpyxl import load_workbook
    except ImportError as exc:  # pragma: no cover - env issue
        logger.error("openpyxl is required to read xlsx inputs: %s", exc)
        return

    wb = load_workbook(path, read_only=True, data_only=True)
    sheet_names = [name for name in wb.sheetnames if name != "Summary"]
    expected_headers = ["ID", "Type", "Value", "UUID", "To IDs", "Comment",
                        "Timestamp", "Event ID", "Event Info", "Tags"]
    for name in sheet_names:
        ws = wb[name]
        rows = ws.iter_rows(values_only=True)
        try:
            headers = next(rows)
        except StopIteration:
            continue
        if list(headers)[: len(expected_headers)] != expected_headers:
            logger.warning("Sheet %s headers don't match expected layout; skipping.", name)
            continue
        category = name
        for row in rows:
            if not row or all(cell is None for cell in row):
                continue
            (_id, _type, _value, _uuid, _to_ids,
             _comment, _ts, _ev_id, _ev_info, _tags) = (list(row) + [None] * 10)[:10]
            yield {
                "id": str(_id or ""),
                "uuid": str(_uuid or ""),
                "type": str(_type or ""),
                "category": category,
                "value": str(_value) if _value is not None else "",
                "to_ids": str(_to_ids).lower() in {"1", "true", "yes"},
                "comment": str(_comment or ""),
                "timestamp": str(_ts or ""),
                "event_id": str(_ev_id or ""),
                "event_info": str(_ev_info or ""),
                "tags": [t.strip() for t in str(_tags or "").split(",") if t.strip()],
            }
    wb.close()


def _empty_value(attr: Dict[str, Any], cfg: FilterConfig) -> Optional[str]:
    if not attr["value"] or len(attr["value"]) < cfg.min_value_length:
        return "empty_or_short_value"
    if len(attr["value"]) > cfg.max_value_length:
        return "value_too_long"
    return None


def _category_blocked(attr: Dict[str, Any], cfg: FilterConfig) -> Optional[str]:
    return "noise_category" if attr["category"] in cfg.noise_categories else None


def _type_is_noisy(attr: Dict[str, Any], cfg: FilterConfig) -> Optional[str]:
    t = attr["type"].lower()
    return "noise_type" if any(t.startswith(p) for p in cfg.noise_type_prefixes) else None


def _to_ids_required(attr: Dict[str, Any], cfg: FilterConfig) -> Optional[str]:
    return "to_ids_false" if (cfg.require_to_ids and not attr["to_ids"]) else None


def _bad_ip(attr: Dict[str, Any], cfg: FilterConfig) -> Optional[str]:
    if attr["type"] not in IP_TYPES:
        return None
    if not _is_valid_ip(attr["value"]):
        return "invalid_ip"
    if cfg.drop_private_ips and _is_private_ip(attr["value"]):
        return "private_ip"
    return None


def _bad_hash(attr: Dict[str, Any], _cfg: FilterConfig) -> Optional[str]:
    if attr["type"] not in HASH_TYPES:
        return None
    return None if _is_valid_hash(attr["value"], attr["type"]) else "invalid_hash"


def _is_deleted(attr: Dict[str, Any], _cfg: FilterConfig) -> Optional[str]:
    return "deleted_in_misp" if attr.get("deleted") is True else None


_RULES = (_empty_value, _category_blocked, _type_is_noisy,
          _to_ids_required, _bad_ip, _bad_hash, _is_deleted)


def _normalise_value(attr: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(attr)
    out["value"] = _normalise_text(attr["value"])
    out["comment"] = _normalise_text(attr["comment"])
    out["event_info"] = _normalise_text(attr["event_info"])

    if attr["type"] in URL_LIKE_TYPES:
        out["value"] = _refang(out["value"]).lower()
    elif attr["type"] in HASH_TYPES:
        out["value"] = out["value"].lower()
    elif attr["type"] in IP_TYPES:
        out["value"] = out["value"].lower()
    return out


def _dedup_key(attr: Dict[str, Any]) -> str:
    raw = f"{attr['type'].lower()}|{attr['value']}"
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


class CTIFilter:

    def __init__(self, cfg: Optional[FilterConfig] = None) -> None:
        self.cfg = cfg or FilterConfig()
        self.drop_counts: Counter[str] = Counter()
        self.kept: int = 0
        self.seen: int = 0

    def _evaluate(self, attr: Dict[str, Any]) -> Optional[str]:
        for rule in _RULES:
            reason = rule(attr, self.cfg)
            if reason is not None:
                return reason
        return None

    def filter_attributes(self, attrs: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
        survivors: Dict[str, Dict[str, Any]] = {}

        for attr in attrs:
            self.seen += 1
            reason = self._evaluate(attr)
            if reason is not None:
                self.drop_counts[reason] += 1
                continue

            cleaned = _normalise_value(attr)

            # Re-check after normalisation — a refang can produce an empty value
            post_reason = _empty_value(cleaned, self.cfg)
            if post_reason is not None:
                self.drop_counts[post_reason] += 1
                continue

            key = _dedup_key(cleaned)
            existing = survivors.get(key)
            if existing is None:
                survivors[key] = cleaned
            else:
                self.drop_counts["duplicate"] += 1
                if self.cfg.dedup_strategy == "latest":
                    if cleaned["timestamp"] > existing["timestamp"]:
                        survivors[key] = cleaned

        out = sorted(
            survivors.values(),
            key=lambda a: (a["timestamp"], a["id"]),
            reverse=True,
        )
        self.kept = len(out)
        logger.info("Filter run complete: seen=%d kept=%d dropped=%d",
                    self.seen, self.kept, self.seen - self.kept)
        return out

    def stats(self) -> Dict[str, Any]:
        return {
            "seen": self.seen,
            "kept": self.kept,
            "dropped": self.seen - self.kept,
            "drop_reasons": dict(self.drop_counts.most_common()),
            "config": {
                "noise_categories": sorted(self.cfg.noise_categories),
                "noise_type_prefixes": sorted(self.cfg.noise_type_prefixes),
                "require_to_ids": self.cfg.require_to_ids,
                "drop_private_ips": self.cfg.drop_private_ips,
                "dedup_strategy": self.cfg.dedup_strategy,
            },
        }


def discover_inputs(repo_root: Path) -> List[Path]:
    data_dir = repo_root / "data"
    if not data_dir.is_dir():
        return []

    misp_dir = data_dir / "misp"
    inputs = sorted(data_dir.glob("*.json"))
    if misp_dir.is_dir():
        inputs.extend(sorted(misp_dir.glob("*.json")))
        inputs.extend(sorted(misp_dir.glob("*.xlsx")))

    return inputs


def _input_provenance(repo_root: Path, inputs: List[Path]) -> Dict[str, Any]:
    data_dir = repo_root / "data"
    misp_dir = data_dir / "misp"
    legacy_xlsx = sorted(misp_dir.glob("*.xlsx")) if misp_dir.is_dir() else []

    def rel(path: Path) -> str:
        try:
            return str(path.relative_to(repo_root))
        except ValueError:
            return str(path)

    if not data_dir.is_dir():
        mode = "no_data_dir"
    elif not inputs:
        mode = "no_inputs"
    elif any(path.suffix.lower() == ".json" for path in inputs):
        mode = "json_preferred"
    else:
        mode = "xlsx_fallback"

    skipped = legacy_xlsx if mode == "json_preferred" else []
    return {
        "mode": mode,
        "used_files": [rel(path) for path in inputs],
        "skipped_files": [rel(path) for path in skipped],
    }


def _iter_inputs(paths: Iterable[Path]) -> Iterable[Dict[str, Any]]:
    for path in paths:
        try:
            if path.suffix.lower() == ".json":
                yield from _load_misp_event_json(path)
            elif path.suffix.lower() == ".xlsx":
                yield from _load_misp_xlsx(path)
            else:
                logger.warning("Unsupported input type: %s", path)
        except Exception:
            logger.exception("Failed to read %s — continuing.", path)


def run_pipeline(
    repo_root: Path,
    output_dir: Optional[Path] = None,
    cfg: Optional[FilterConfig] = None,
    today: Optional[date] = None,
) -> Tuple[Path, Path]:
    inputs = discover_inputs(repo_root)
    inputs_meta = _input_provenance(repo_root, inputs)
    if not inputs:
        raise FileNotFoundError(f"No MISP-shaped inputs found under {repo_root / 'data'}")

    output_dir = output_dir or (repo_root / "data" / "cleaned")
    output_dir.mkdir(parents=True, exist_ok=True)

    stamp = (today or datetime.now(timezone.utc).date()).isoformat()
    cleaned_path = output_dir / f"misp_filtered_{stamp}.json"
    stats_path = output_dir / f"misp_filter_stats_{stamp}.json"

    cti_filter = CTIFilter(cfg)
    cleaned = cti_filter.filter_attributes(_iter_inputs(inputs))

    with cleaned_path.open("w", encoding="utf-8") as fh:
        json.dump(cleaned, fh, ensure_ascii=False, indent=2)
    with stats_path.open("w", encoding="utf-8") as fh:
        stats = cti_filter.stats()
        stats["inputs"] = inputs_meta
        json.dump(stats, fh, ensure_ascii=False, indent=2)

    logger.info("Wrote %d cleaned attributes -> %s", len(cleaned), cleaned_path)
    logger.info("Wrote stats -> %s", stats_path)
    return cleaned_path, stats_path


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="cti_filter",
        description="Reduce noise in CTI data and emit a cleaned JSON dataset.",
    )
    p.add_argument(
        "--repo-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Path to the CCTI repo root (default: parent of this file).",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Where to write the cleaned dataset (default: <repo>/data/cleaned).",
    )
    p.add_argument(
        "--require-to-ids",
        action="store_true",
        help="Drop attributes where to_ids is False (stricter).",
    )
    p.add_argument(
        "--keep-private-ips",
        action="store_true",
        help="Do NOT drop RFC1918/loopback/link-local IPs.",
    )
    p.add_argument(
        "--dedup",
        choices=("latest", "first"),
        default="latest",
        help="Which record to keep when duplicates collide.",
    )
    return p


def main(argv: Optional[List[str]] = None) -> int:
    args = _build_arg_parser().parse_args(argv)
    cfg = FilterConfig(
        require_to_ids=args.require_to_ids,
        drop_private_ips=not args.keep_private_ips,
        dedup_strategy=args.dedup,
    )
    try:
        run_pipeline(args.repo_root, args.output_dir, cfg=cfg)
    except FileNotFoundError as exc:
        logger.error("%s", exc)
        return 2
    except Exception:
        logger.exception("Unhandled error during pipeline run.")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
