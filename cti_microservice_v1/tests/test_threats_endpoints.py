"""Smoke + behaviour tests for ``threats_endpoints``.

These tests stand up only the threats router (not the heavy RAG engine
loaded by ``api.py``'s lifespan) and point the loaders at a synthetic
cleaned dataset under a temporary directory. They run in well under a
second on the Droplet and do not touch the real pipeline output.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

# The microservice files import each other by *bare* module names
# (``from rag_engine import ThreatIntelRAG``), so the directory containing
# them must be on sys.path before the import.
_MICROSERVICE_DIR = Path(__file__).resolve().parents[1]
if str(_MICROSERVICE_DIR) not in sys.path:
    sys.path.insert(0, str(_MICROSERVICE_DIR))


@pytest.fixture
def client(tmp_path: Path, monkeypatch):
    """A FastAPI TestClient wired to a synthetic cleaned dataset."""
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    payload = [
        {"id": "1", "uuid": "u1", "type": "ip-dst", "category": "Network activity",
         "value": "8.8.8.8", "to_ids": True, "comment": "", "timestamp": "100",
         "event_id": "1", "event_info": "feed", "tags": []},
        {"id": "2", "uuid": "u2", "type": "md5", "category": "Payload delivery",
         "value": "a" * 32, "to_ids": True, "comment": "", "timestamp": "200",
         "event_id": "1", "event_info": "feed", "tags": ["apt"]},
    ]
    stats = {
        "drop_reasons": {"noise_category": 5, "duplicate": 12},
        "inputs": {"mode": "json_preferred", "used_files": ["data/feed.json"], "skipped_files": []},
    }
    (cleaned_dir / "misp_filtered_2026-04-29.json").write_text(json.dumps(payload), encoding="utf-8")
    (cleaned_dir / "misp_filter_stats_2026-04-29.json").write_text(json.dumps(stats), encoding="utf-8")

    monkeypatch.setenv("CCTI_CLEANED_DIR", str(cleaned_dir))
    # Use a NER file that doesn't exist so the entities endpoint returns empty buckets.
    monkeypatch.setenv("CCTI_NER_FILE", str(tmp_path / "absent_ner.json"))

    # Imports happen *after* env vars are set so _DEFAULT_* are picked up.
    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import threats_endpoints

    threats_endpoints.reload_cache()
    app = FastAPI()
    app.include_router(threats_endpoints.router)
    return TestClient(app)


def test_stats_returns_totals_and_drop_reasons(client):
    r = client.get("/api/v1/threats/stats")
    assert r.status_code == 200
    data = r.json()
    assert data["total"] == 2
    assert data["by_category"]["Network activity"] == 1
    assert data["drop_reasons"]["noise_category"] == 5
    assert data["inputs"]["mode"] == "json_preferred"
    # AC2 of frontend story: severity field is populated.
    assert "critical" in data["by_severity"]


def test_list_threats_default_paging(client):
    r = client.get("/api/v1/threats")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2 and body["page"] == 1
    # Severity is computed server-side and surfaced to UI.
    severities = {item["severity"] for item in body["items"]}
    assert severities == {"critical", "medium"}  # tag "apt" -> critical


def test_list_threats_filter_by_severity(client):
    r = client.get("/api/v1/threats?severity=critical")
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 1
    assert body["items"][0]["uuid"] == "u2"


def test_list_threats_search_q(client):
    r = client.get("/api/v1/threats?q=8.8.8.8")
    assert r.status_code == 200
    assert r.json()["total"] == 1


def test_get_threat_by_uuid(client):
    r = client.get("/api/v1/threats/u1")
    assert r.status_code == 200 and r.json()["value"] == "8.8.8.8"


def test_get_threat_404_when_missing(client):
    r = client.get("/api/v1/threats/does-not-exist")
    assert r.status_code == 404


def test_entities_returns_empty_when_ner_absent(client):
    r = client.get("/api/v1/entities")
    assert r.status_code == 200
    body = r.json()
    assert body == {"threat_actors": [], "malware": [], "industries": [], "other": {}}


def test_admin_reload_clears_cache(client):
    r = client.post("/api/v1/admin/reload")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_display_org_name_falls_back_to_profile_filename():
    import threats_endpoints

    assert threats_endpoints._display_org_name({}, Path("foodfirst_profile.json")) == "FoodFirst"
    assert threats_endpoints._display_org_name({}, Path("royaladelaidehospital_profile.json")) == "Royal Adelaide Hospital"
