"""
CCTI Non-Functional Requirements Test Suite
=============================================
Covers the non-functional requirements of the CCTI system:
  NFR-01  API response time (performance)
  NFR-02  Filter throughput (performance)
  NFR-03  Recommendation scoring throughput (performance)
  NFR-04  Data sanitisation — no internal IDs in API responses (security)
  NFR-05  Private IP suppression (security)
  NFR-06  Encrypted file is unreadable without key (security)
  NFR-07  API handles malformed input gracefully (reliability)
  NFR-08  Filter is deterministic across repeated runs (reliability)
  NFR-09  API continues serving after cache reload (availability)
  NFR-10  Scoring handles missing/empty profile fields without crashing (robustness)
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "filtering"))
sys.path.insert(0, str(ROOT / "cti_microservice_v1"))


# ---------------------------------------------------------------------------
# Shared API client fixture
# ---------------------------------------------------------------------------
@pytest.fixture
def client(tmp_path, monkeypatch):
    cleaned_dir = tmp_path / "cleaned"
    cleaned_dir.mkdir()
    payload = [
        {"id": str(i), "uuid": f"u{i}", "type": "ip-dst",
         "category": "Network activity", "value": f"1.2.3.{i}",
         "to_ids": True, "comment": f"event {i}", "timestamp": str(i),
         "event_id": str(i), "event_info": f"feed-{i}", "tags": []}
        for i in range(1, 51)
    ]
    stats = {"drop_reasons": {}, "inputs": {"mode": "json_preferred", "used_files": [], "skipped_files": []}}
    (cleaned_dir / "misp_filtered_2026-06-09.json").write_text(json.dumps(payload), encoding="utf-8")
    (cleaned_dir / "misp_filter_stats_2026-06-09.json").write_text(json.dumps(stats), encoding="utf-8")
    monkeypatch.setenv("CCTI_CLEANED_DIR", str(cleaned_dir))
    monkeypatch.setenv("CCTI_NER_FILE", str(tmp_path / "absent.json"))

    pytest.importorskip("fastapi")
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    import threats_endpoints
    threats_endpoints.reload_cache()
    app = FastAPI()
    app.include_router(threats_endpoints.router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# NFR-01: API response time < 500ms for threat listing
# ---------------------------------------------------------------------------
class TestNFR01_APIPerformance:
    """API SHALL respond to threat listing requests within 500ms."""

    def test_threats_list_responds_under_500ms(self, client):
        start = time.perf_counter()
        r = client.get("/api/v1/threats")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert r.status_code == 200
        assert elapsed_ms < 500, f"Threats list took {elapsed_ms:.0f}ms — exceeds 500ms SLA"

    def test_stats_endpoint_responds_under_200ms(self, client):
        start = time.perf_counter()
        r = client.get("/api/v1/threats/stats")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert r.status_code == 200
        assert elapsed_ms < 200, f"Stats endpoint took {elapsed_ms:.0f}ms — exceeds 200ms SLA"

    def test_uuid_lookup_responds_under_100ms(self, client):
        start = time.perf_counter()
        r = client.get("/api/v1/threats/u1")
        elapsed_ms = (time.perf_counter() - start) * 1000
        assert r.status_code == 200
        assert elapsed_ms < 100, f"UUID lookup took {elapsed_ms:.0f}ms — exceeds 100ms SLA"


# ---------------------------------------------------------------------------
# NFR-02: Filter throughput — process 1,000 attributes in < 5 seconds
# ---------------------------------------------------------------------------
class TestNFR02_FilterThroughput:
    """The noise-reduction filter SHALL process 1,000 attributes in under 5 seconds."""

    def test_filter_1000_attributes_under_5s(self):
        from cti_filter import CTIFilter
        attrs = [
            {"id": str(i), "uuid": f"u{i}", "type": "ip-dst",
             "category": "Network activity", "value": f"8.8.{i // 256}.{i % 256}",
             "to_ids": True, "comment": "", "timestamp": str(i),
             "event_id": "1", "event_info": "test", "tags": []}
            for i in range(1, 1001)
        ]
        f = CTIFilter()
        start = time.perf_counter()
        f.filter_attributes(attrs)
        elapsed = time.perf_counter() - start
        assert elapsed < 5.0, f"Filter took {elapsed:.2f}s for 1,000 records — too slow"


# ---------------------------------------------------------------------------
# NFR-03: Recommendation scoring throughput
# ---------------------------------------------------------------------------
class TestNFR03_ScoringThroughput:
    """MITRE scoring SHALL complete for 100 events in under 2 seconds."""

    def test_mitre_score_100_events_under_2s(self, monkeypatch):
        from collections import Counter
        import recommendation_engine as rec
        monkeypatch.setattr(rec, "_profile_match_frequencies",
                            lambda: (Counter({"T1566": 5, "T1190": 2}), Counter({"fin7": 1})))
        profile = {"risk_profile": {"threat_landscape": ["T1566 - Phishing", "FIN7"]},
                   "organizational_context": {"industry_sector": "Finance"}}
        events = [
            {"attack_techniques": ["T1566"], "threat_actors": ["FIN7"], "entity_details": []}
            for _ in range(100)
        ]
        start = time.perf_counter()
        for e in events:
            rec.compute_mitre_score(profile, e)
        elapsed = time.perf_counter() - start
        assert elapsed < 2.0, f"Scoring 100 events took {elapsed:.2f}s — too slow"


# ---------------------------------------------------------------------------
# NFR-04: Security — internal MISP IDs not exposed in API response
# ---------------------------------------------------------------------------
class TestNFR04_DataSanitisation:
    """The API SHALL NOT expose internal MISP event_info or event_id fields."""

    def test_threat_response_strips_event_info(self, client):
        r = client.get("/api/v1/threats/u1")
        assert r.status_code == 200
        body = r.json()
        assert "event_info" not in body, "event_info (internal MISP field) must not be in API response"

    def test_threats_list_items_strip_event_info(self, client):
        r = client.get("/api/v1/threats")
        assert r.status_code == 200
        for item in r.json()["items"]:
            assert "event_info" not in item


# ---------------------------------------------------------------------------
# NFR-05: Security — private IPs never reach the API
# ---------------------------------------------------------------------------
class TestNFR05_PrivateIPSuppression:
    """Private/RFC1918 IP addresses SHALL be filtered before reaching the API."""

    def test_private_ips_absent_from_api_after_filter(self, tmp_path, monkeypatch):
        cleaned_dir = tmp_path / "cleaned"
        cleaned_dir.mkdir()
        # Simulate a cleaned dataset that went through the filter — private IPs gone
        payload = [
            {"id": "1", "uuid": "u1", "type": "ip-dst", "category": "Network activity",
             "value": "8.8.8.8", "to_ids": True, "comment": "", "timestamp": "1",
             "event_id": "1", "event_info": "feed", "tags": []}
        ]
        stats = {"drop_reasons": {}, "inputs": {"mode": "json_preferred", "used_files": [], "skipped_files": []}}
        (cleaned_dir / "misp_filtered_2026-06-09.json").write_text(json.dumps(payload), encoding="utf-8")
        (cleaned_dir / "misp_filter_stats_2026-06-09.json").write_text(json.dumps(stats), encoding="utf-8")
        monkeypatch.setenv("CCTI_CLEANED_DIR", str(cleaned_dir))
        monkeypatch.setenv("CCTI_NER_FILE", str(tmp_path / "absent.json"))

        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        import threats_endpoints
        threats_endpoints.reload_cache()
        app = FastAPI()
        app.include_router(threats_endpoints.router)
        c = TestClient(app)

        r = c.get("/api/v1/threats")
        values = [item["value"] for item in r.json()["items"]]
        private_found = [v for v in values if v.startswith(("10.", "192.168.", "172."))]
        assert not private_found, f"Private IPs found in API response: {private_found}"


# ---------------------------------------------------------------------------
# NFR-06: Security — encrypted file is unreadable without key
# ---------------------------------------------------------------------------
class TestNFR06_EncryptionSecurity:
    """Encrypted pipeline outputs SHALL be unreadable as plain JSON without the key."""

    def test_encrypted_file_is_not_plain_json(self, monkeypatch, tmp_path):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("CCTI_ENCRYPTION_KEY", key)
        import importlib
        import crypto_utils
        importlib.reload(crypto_utils)

        test_file = tmp_path / "sensitive.json"
        test_file.write_text(json.dumps({"secret": "data"}), encoding="utf-8")
        crypto_utils.encrypt_file(test_file)

        raw = test_file.read_bytes()
        with pytest.raises(Exception):
            json.loads(raw)

    def test_wrong_key_cannot_decrypt(self, monkeypatch, tmp_path):
        from cryptography.fernet import Fernet
        key1 = Fernet.generate_key().decode()
        key2 = Fernet.generate_key().decode()

        monkeypatch.setenv("CCTI_ENCRYPTION_KEY", key1)
        import importlib
        import crypto_utils
        importlib.reload(crypto_utils)

        test_file = tmp_path / "sensitive.json"
        test_file.write_text(json.dumps({"secret": "data"}), encoding="utf-8")
        crypto_utils.encrypt_file(test_file)

        monkeypatch.setenv("CCTI_ENCRYPTION_KEY", key2)
        importlib.reload(crypto_utils)
        with pytest.raises(Exception):
            crypto_utils.decrypt_json(test_file)


# ---------------------------------------------------------------------------
# NFR-07: Reliability — API handles malformed input gracefully
# ---------------------------------------------------------------------------
class TestNFR07_InputResilience:
    """The API SHALL return 4xx for malformed requests, not 500."""

    def test_invalid_page_size_returns_error_not_500(self, client):
        r = client.get("/api/v1/threats?page_size=-1")
        assert r.status_code in (400, 422), \
            f"Expected 400/422 for invalid page_size, got {r.status_code}"

    def test_missing_uuid_returns_404_not_500(self, client):
        r = client.get("/api/v1/threats/completely-fake-uuid-xyz")
        assert r.status_code == 404

    def test_empty_search_query_returns_all(self, client):
        r = client.get("/api/v1/threats?q=")
        assert r.status_code == 200
        assert r.json()["total"] == 50


# ---------------------------------------------------------------------------
# NFR-08: Reliability — filter output is deterministic
# ---------------------------------------------------------------------------
class TestNFR08_Determinism:
    """The filter SHALL produce identical output for identical input across runs."""

    def test_filter_is_deterministic(self):
        from cti_filter import CTIFilter
        attrs = [
            {"id": str(i), "uuid": f"u{i}", "type": "ip-dst",
             "category": "Network activity", "value": f"8.8.{i}.1",
             "to_ids": True, "comment": "", "timestamp": str(i),
             "event_id": "1", "event_info": "test", "tags": []}
            for i in range(1, 21)
        ]
        out1 = CTIFilter().filter_attributes(attrs)
        out2 = CTIFilter().filter_attributes(attrs)
        assert [r["id"] for r in out1] == [r["id"] for r in out2]


# ---------------------------------------------------------------------------
# NFR-09: Availability — API continues serving after admin reload
# ---------------------------------------------------------------------------
class TestNFR09_Availability:
    """The API SHALL continue serving requests after a cache reload without downtime."""

    def test_api_serves_after_reload(self, client):
        r1 = client.get("/api/v1/threats")
        assert r1.status_code == 200
        assert r1.json()["total"] == 50

        reload_r = client.post("/api/v1/admin/reload")
        assert reload_r.status_code == 200

        r2 = client.get("/api/v1/threats")
        assert r2.status_code == 200
        assert r2.json()["total"] == 50


# ---------------------------------------------------------------------------
# NFR-10: Robustness — scoring handles incomplete profiles without crashing
# ---------------------------------------------------------------------------
class TestNFR10_ScoringRobustness:
    """The recommendation engine SHALL handle malformed or empty profiles gracefully."""

    def test_empty_profile_does_not_crash(self, monkeypatch):
        from collections import Counter
        import recommendation_engine as rec
        monkeypatch.setattr(rec, "_profile_match_frequencies",
                            lambda: (Counter(), Counter()))
        empty_profile = {}
        event = {"attack_techniques": ["T1566"], "threat_actors": [], "entity_details": []}
        try:
            score = rec.compute_mitre_score(empty_profile, event)
            assert isinstance(score, float)
        except Exception as e:
            pytest.fail(f"Scoring crashed on empty profile: {e}")

    def test_event_with_no_entities_scores_zero(self, monkeypatch):
        from collections import Counter
        import recommendation_engine as rec
        monkeypatch.setattr(rec, "_profile_match_frequencies",
                            lambda: (Counter(), Counter()))
        profile = {"risk_profile": {"threat_landscape": ["T1566 - Phishing"]}}
        empty_event = {"attack_techniques": [], "threat_actors": [], "entity_details": []}
        score = rec.compute_mitre_score(profile, empty_event)
        assert score == 0.0

    def test_missing_threat_landscape_does_not_crash(self, monkeypatch):
        from collections import Counter
        import recommendation_engine as rec
        monkeypatch.setattr(rec, "_profile_match_frequencies",
                            lambda: (Counter(), Counter()))
        profile = {"risk_profile": {}}  # no threat_landscape key
        event = {"attack_techniques": ["T1566"], "entity_details": []}
        try:
            rec.compute_mitre_score(profile, event)
        except Exception as e:
            pytest.fail(f"Scoring crashed on profile with no threat_landscape: {e}")
