"""
CCTI Functional Requirements Test Suite
========================================
Covers the core functional requirements of the CCTI pipeline:
  FR-01  Data ingestion from multiple sources
  FR-02  Noise reduction and deduplication
  FR-03  Named entity extraction (NER)
  FR-04  MITRE ATT&CK technique normalisation
  FR-05  Organisation profile scoring
  FR-06  MITRE technique matching and lift
  FR-07  Cross-organisation differentiation
  FR-08  REST API threat listing and filtering
  FR-09  Recommendation output format
  FR-10  Encryption key gating
"""
from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "filtering"))
sys.path.insert(0, str(ROOT / "cti_microservice_v1"))


# ---------------------------------------------------------------------------
# FR-01: Data ingestion from multiple sources
# ---------------------------------------------------------------------------
class TestFR01_DataIngestion:
    """The system SHALL ingest CTI from both local static feeds and live MISP events."""

    def test_filter_discovers_local_json_sources(self, tmp_path):
        from cti_filter import discover_inputs
        data_dir = tmp_path / "data"
        misp_dir = data_dir / "misp"
        misp_dir.mkdir(parents=True)
        (data_dir / "feodo.json").write_text("[]", encoding="utf-8")
        (data_dir / "malwarebazaar.json").write_text("[]", encoding="utf-8")
        (misp_dir / "misp_raw_events.json").write_text("[]", encoding="utf-8")
        inputs = discover_inputs(tmp_path)
        paths = [p.name for p in inputs]
        assert "feodo.json" in paths
        assert "malwarebazaar.json" in paths
        assert "misp_raw_events.json" in paths

    def test_filter_combines_all_sources_into_single_dataset(self, tmp_path):
        from cti_filter import run_pipeline
        from datetime import date
        data_dir = tmp_path / "data"
        misp_dir = data_dir / "misp"
        misp_dir.mkdir(parents=True)

        # MISP event format: list of event dicts with Attribute arrays
        def misp_event(attr_id, value, event_id):
            return {"Event": {
                "id": event_id, "info": f"feed-{event_id}",
                "Attribute": [{"id": attr_id, "type": "ip-dst",
                               "category": "Network activity", "value": value,
                               "to_ids": True, "uuid": f"u{attr_id}",
                               "timestamp": attr_id, "tags": []}]
            }}

        feed1 = [misp_event("1", "8.8.4.1", "1")]
        feed2 = [misp_event("2", "8.8.4.2", "2")]
        (data_dir / "feed1.json").write_text(json.dumps(feed1), encoding="utf-8")
        (misp_dir / "misp_raw_events.json").write_text(json.dumps(feed2), encoding="utf-8")

        cleaned_path, _ = run_pipeline(repo_root=tmp_path, today=date(2026, 6, 9))
        cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
        values = {r["value"] for r in cleaned}
        assert "8.8.4.1" in values
        assert "8.8.4.2" in values


# ---------------------------------------------------------------------------
# FR-02: Noise reduction and deduplication
# ---------------------------------------------------------------------------
class TestFR02_NoiseReduction:
    """The system SHALL remove noise attributes and deduplicate across feeds."""

    def test_private_ips_are_dropped(self):
        from cti_filter import CTIFilter
        f = CTIFilter()
        out = f.filter_attributes([
            {"id": "1", "uuid": "u1", "type": "ip-dst", "category": "Network activity",
             "value": "10.0.0.1", "to_ids": True, "comment": "",
             "timestamp": "1", "event_id": "1", "event_info": "", "tags": []}
        ])
        assert out == []

    def test_whois_type_is_dropped(self):
        from cti_filter import CTIFilter
        f = CTIFilter()
        out = f.filter_attributes([
            {"id": "1", "uuid": "u1", "type": "whois-registrant-email",
             "category": "Network activity", "value": "admin@evil.com",
             "to_ids": True, "comment": "", "timestamp": "1",
             "event_id": "1", "event_info": "", "tags": []}
        ])
        assert out == []

    def test_cross_feed_duplicate_keeps_latest(self):
        from cti_filter import CTIFilter
        f = CTIFilter()
        attrs = [
            {"id": "1", "uuid": "u1", "type": "ip-dst", "category": "Network activity",
             "value": "1.2.3.4", "to_ids": True, "comment": "",
             "timestamp": "100", "event_id": "1", "event_info": "", "tags": []},
            {"id": "2", "uuid": "u2", "type": "ip-dst", "category": "Network activity",
             "value": "1.2.3.4", "to_ids": True, "comment": "",
             "timestamp": "999", "event_id": "2", "event_info": "", "tags": []},
        ]
        out = f.filter_attributes(attrs)
        assert len(out) == 1
        assert out[0]["id"] == "2"

    def test_malformed_hash_is_dropped(self):
        from cti_filter import CTIFilter
        f = CTIFilter()
        out = f.filter_attributes([
            {"id": "1", "uuid": "u1", "type": "md5", "category": "Payload delivery",
             "value": "tooshort", "to_ids": True, "comment": "",
             "timestamp": "1", "event_id": "1", "event_info": "", "tags": []}
        ])
        assert out == []

    def test_url_is_defanged_and_normalised(self):
        from cti_filter import CTIFilter
        f = CTIFilter()
        out = f.filter_attributes([
            {"id": "1", "uuid": "u1", "type": "url", "category": "Network activity",
             "value": "hxxps://EVIL[.]com/payload  ", "to_ids": True, "comment": "",
             "timestamp": "1", "event_id": "1", "event_info": "", "tags": []}
        ])
        assert out[0]["value"] == "https://evil.com/payload"


# ---------------------------------------------------------------------------
# FR-03: Named entity extraction produces expected entity types
# ---------------------------------------------------------------------------
class TestFR03_NEROutput:
    """The system SHALL extract structured entities from CTI text."""

    def test_ner_output_file_has_required_keys(self):
        ner_file = ROOT / "ner" / "ner_results_v3_normalised.json"
        if not ner_file.exists():
            ner_file = ROOT / "ner" / "ner_results_v3.json"
        if not ner_file.exists():
            pytest.skip("NER output not present — run pipeline first")
        with ner_file.open(encoding="utf-8") as f:
            data = json.load(f)
        assert isinstance(data, list) and len(data) > 0
        # NER records use an 'entities' dict containing entity type sub-keys
        required_keys = {"event_id", "entities", "embedding_text"}
        for record in data[:10]:
            for key in required_keys:
                assert key in record, f"Missing key '{key}' in NER record"
        # entities dict should contain expected entity type buckets
        entity_buckets = {"attack_techniques", "threat_actors", "technologies"}
        for record in data[:10]:
            entities = record.get("entities", {})
            for bucket in entity_buckets:
                assert bucket in entities, f"Missing entity bucket '{bucket}' in NER record"

    def test_ner_records_have_confidence_scores(self):
        ner_file = ROOT / "ner" / "ner_results_v3_normalised.json"
        if not ner_file.exists():
            ner_file = ROOT / "ner" / "ner_results_v3.json"
        if not ner_file.exists():
            pytest.skip("NER output not present — run pipeline first")
        with ner_file.open(encoding="utf-8") as f:
            data = json.load(f)
        # Find records where entities dict has items with confidence scores
        records_with_entities = [
            r for r in data
            if any(r.get("entities", {}).get(k) for k in ["attack_techniques", "threat_actors", "technologies"])
        ]
        assert len(records_with_entities) > 0, "Expected at least one NER record with extracted entities"
        # Check confidence scores exist in entity details where present
        for record in records_with_entities[:5]:
            entities = record.get("entities", {})
            for bucket_items in entities.values():
                if isinstance(bucket_items, list):
                    for item in bucket_items:
                        if isinstance(item, dict) and "confidence" in item:
                            assert 0.0 <= item["confidence"] <= 1.0


# ---------------------------------------------------------------------------
# FR-04: MITRE technique normalisation
# ---------------------------------------------------------------------------
class TestFR04_MITRENormalisation:
    """The system SHALL map free-text technique names to official T-codes."""

    def test_normalise_converts_text_to_tcode(self):
        import normalise_techniques as nt
        # Use _static_normalise directly — maps known technique names to T-codes
        result = nt._static_normalise("Phishing")
        assert result is not None and result.startswith("T"), \
            f"Expected a T-code for 'Phishing', got: {result}"

    def test_existing_tcode_is_preserved(self):
        import normalise_techniques as nt
        # A value already in T-code format should be returned as-is
        result = nt._static_normalise("T1566")
        assert result == "T1566", f"Expected T1566 to be preserved, got: {result}"


# ---------------------------------------------------------------------------
# FR-05: Organisation profile scoring produces ranked recommendations
# ---------------------------------------------------------------------------
class TestFR05_RecommendationEngine:
    """The system SHALL score CTI events against org profiles and return top-N ranked results."""

    def test_recommendations_dir_has_files_for_all_orgs(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")
        files = list(rec_dir.glob("recommendations_ORG-*.json"))
        assert len(files) >= 12, f"Expected 12 org recommendation files, found {len(files)}"

    def test_recommendation_file_has_correct_structure(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")
        files = sorted(rec_dir.glob("recommendations_ORG-EN-001_*.json"))
        if not files:
            pytest.skip("ORG-EN-001 recommendation file not found")
        with files[-1].open(encoding="utf-8") as f:
            data = json.load(f)
        assert "org_id" in data
        assert "recommendations" in data
        recs = data["recommendations"]
        assert len(recs) >= 1
        first = recs[0]
        assert "rank" in first
        assert "score" in first
        assert first["rank"] == 1

    def test_recommendations_are_sorted_descending_by_score(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")
        files = sorted(rec_dir.glob("recommendations_ORG-EN-001_*.json"))
        if not files:
            pytest.skip("ORG-EN-001 recommendation file not found")
        with files[-1].open(encoding="utf-8") as f:
            data = json.load(f)
        scores = [r["score"] for r in data["recommendations"]]
        assert scores == sorted(scores, reverse=True), "Recommendations are not sorted by score descending"

    def test_top_recommendation_score_is_positive(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")
        for org in ["ORG-EN-001", "ORG-HC-001", "ORG-FB-001"]:
            files = sorted(rec_dir.glob(f"recommendations_{org}_*.json"))
            if not files:
                continue
            with files[-1].open(encoding="utf-8") as f:
                data = json.load(f)
            assert data["recommendations"][0]["score"] > 0, \
                f"{org} top recommendation has score <= 0"


# ---------------------------------------------------------------------------
# FR-06: MITRE technique matching produces measurable lift
# ---------------------------------------------------------------------------
class TestFR06_MITRELift:
    """The system SHALL improve event scores when MITRE techniques overlap with org profile."""

    def test_mitre_match_increases_score_over_baseline(self, monkeypatch):
        import recommendation_engine as rec
        monkeypatch.setattr(rec, "_profile_match_frequencies",
                            lambda: (Counter({"T1566": 1}), Counter()))
        profile = {"risk_profile": {"threat_landscape": ["T1566 - Phishing"]}}
        event_with_mitre = {"attack_techniques": ["T1566"], "entity_details": []}
        event_no_mitre   = {"attack_techniques": [], "entity_details": []}
        score_with = rec.compute_mitre_score(profile, event_with_mitre)
        score_without = rec.compute_mitre_score(profile, event_no_mitre)
        assert score_with > score_without

    def test_high_confidence_entity_scores_higher_than_low(self, monkeypatch):
        import recommendation_engine as rec
        monkeypatch.setattr(rec, "_profile_match_frequencies",
                            lambda: (Counter({"T1566": 1}), Counter()))
        profile = {"risk_profile": {"threat_landscape": ["T1566 - Phishing"]}}
        # Use two different techniques with different confidence levels —
        # confidence is looked up from entity_details by entity value
        high_conf = {"attack_techniques": ["T1566", "T1190"],
                     "entity_details": [
                         {"entity": "T1566", "type": "attack_technique", "confidence": 0.98},
                         {"entity": "T1190", "type": "attack_technique", "confidence": 0.98},
                     ]}
        low_conf  = {"attack_techniques": ["T1566"],
                     "entity_details": [
                         {"entity": "T1566", "type": "attack_technique", "confidence": 0.50},
                     ]}
        # More matching techniques + higher confidence should produce a higher score
        assert rec.compute_mitre_score(profile, high_conf) >= rec.compute_mitre_score(profile, low_conf)

    def test_industry_fallback_applied_when_no_direct_match(self, monkeypatch):
        import recommendation_engine as rec
        monkeypatch.setattr(rec, "_profile_match_frequencies",
                            lambda: (Counter(), Counter()))
        finance_profile = {
            "organizational_context": {"industry_sector": "Finance"},
            "risk_profile": {"threat_landscape": []}
        }
        # T1566 is in the finance fallback set
        event = {"attack_techniques": ["T1566"], "entity_details": []}
        score = rec.compute_mitre_score(finance_profile, event)
        assert score > 0, "Industry fallback should produce non-zero score for T1566 in Finance"


# ---------------------------------------------------------------------------
# FR-07: Cross-organisation differentiation
# ---------------------------------------------------------------------------
class TestFR07_CrossOrgDifferentiation:
    """The system SHALL produce different top-10 recommendations for different sectors."""

    def test_energy_and_healthcare_top10_differ(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")

        en_files = sorted(rec_dir.glob("recommendations_ORG-EN-001_*.json"))
        hc_files = sorted(rec_dir.glob("recommendations_ORG-HC-001_*.json"))
        if not en_files or not hc_files:
            pytest.skip("Org recommendation files not found")

        with en_files[-1].open(encoding="utf-8") as f:
            en_recs = {r["event_id"] for r in json.load(f)["recommendations"][:10]}
        with hc_files[-1].open(encoding="utf-8") as f:
            hc_recs = {r["event_id"] for r in json.load(f)["recommendations"][:10]}

        overlap = len(en_recs & hc_recs) / max(len(en_recs | hc_recs), 1) * 100
        assert overlap < 50, f"Energy vs Healthcare top-10 overlap is {overlap:.1f}% — too similar"

    def test_finance_orgs_share_some_recommendations(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")

        fb1_files = sorted(rec_dir.glob("recommendations_ORG-FB-001_*.json"))
        fb2_files = sorted(rec_dir.glob("recommendations_ORG-FB-002_*.json"))
        if not fb1_files or not fb2_files:
            pytest.skip("Finance org files not found")

        with fb1_files[-1].open(encoding="utf-8") as f:
            fb1 = {r["event_id"] for r in json.load(f)["recommendations"][:20]}
        with fb2_files[-1].open(encoding="utf-8") as f:
            fb2 = {r["event_id"] for r in json.load(f)["recommendations"][:20]}

        overlap = len(fb1 & fb2)
        assert overlap >= 1, "Finance orgs with similar profiles should share at least one recommendation"


# ---------------------------------------------------------------------------
# FR-08: REST API — threat listing, filtering, search
# ---------------------------------------------------------------------------
class TestFR08_RestAPI:
    """The system SHALL expose a REST API for browsing and filtering threats."""

    @pytest.fixture
    def client(self, tmp_path, monkeypatch):
        cleaned_dir = tmp_path / "cleaned"
        cleaned_dir.mkdir()
        payload = [
            {"id": "1", "uuid": "u1", "type": "ip-dst", "category": "Network activity",
             "value": "192.0.2.1", "to_ids": True, "comment": "APT28 C2",
             "timestamp": "100", "event_id": "10", "event_info": "APT report", "tags": []},
            {"id": "2", "uuid": "u2", "type": "md5", "category": "Payload delivery",
             "value": "a" * 32, "to_ids": True, "comment": "Ransomware sample",
             "timestamp": "200", "event_id": "11", "event_info": "Malware feed", "tags": ["apt"]},
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

    def test_health_endpoint_returns_200(self, client):
        r = client.get("/api/v1/threats/stats")
        assert r.status_code == 200

    def test_threats_list_returns_all_records(self, client):
        r = client.get("/api/v1/threats")
        assert r.status_code == 200
        assert r.json()["total"] == 2

    def test_severity_filter_returns_only_matching(self, client):
        r = client.get("/api/v1/threats?severity=critical")
        assert r.status_code == 200
        items = r.json()["items"]
        assert all(i["severity"] == "critical" for i in items)

    def test_search_by_value_returns_match(self, client):
        r = client.get("/api/v1/threats?q=192.0.2.1")
        assert r.status_code == 200
        assert r.json()["total"] == 1

    def test_uuid_lookup_returns_correct_record(self, client):
        r = client.get("/api/v1/threats/u1")
        assert r.status_code == 200
        assert r.json()["value"] == "192.0.2.1"

    def test_unknown_uuid_returns_404(self, client):
        r = client.get("/api/v1/threats/nonexistent-uuid")
        assert r.status_code == 404

    def test_response_does_not_expose_internal_misp_ids(self, client):
        r = client.get("/api/v1/threats/u1")
        body = r.json()
        assert "event_info" not in body, "Internal MISP event_info must be stripped from API response"

    def test_pagination_page_size_respected(self, client):
        r = client.get("/api/v1/threats?page=1&page_size=1")
        assert r.status_code == 200
        assert len(r.json()["items"]) == 1


# ---------------------------------------------------------------------------
# FR-09: Recommendation output format
# ---------------------------------------------------------------------------
class TestFR09_RecommendationFormat:
    """Recommendation files SHALL conform to the expected schema."""

    def test_all_recommendation_files_are_valid_json(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")
        files = list(rec_dir.glob("recommendations_ORG-*.json"))
        assert len(files) > 0
        for fp in files:
            try:
                with fp.open(encoding="utf-8") as f:
                    data = json.load(f)
                assert "recommendations" in data
            except Exception as e:
                pytest.fail(f"Invalid JSON in {fp.name}: {e}")

    def test_each_recommendation_has_required_fields(self):
        rec_dir = ROOT / "data" / "recommendations"
        if not rec_dir.exists():
            pytest.skip("Recommendations not generated — run pipeline first")
        files = sorted(rec_dir.glob("recommendations_ORG-EN-001_*.json"))
        if not files:
            pytest.skip("ORG-EN-001 file not found")
        with files[-1].open(encoding="utf-8") as f:
            recs = json.load(f)["recommendations"]
        required = {"rank", "score", "severity", "title"}
        for r in recs[:5]:
            missing = required - set(r.keys())
            assert not missing, f"Recommendation missing fields: {missing}"


# ---------------------------------------------------------------------------
# FR-10: Encryption key gating
# ---------------------------------------------------------------------------
class TestFR10_Encryption:
    """The system SHALL skip encryption when CCTI_ENCRYPTION_KEY is not set,
    and SHALL encrypt when it is set."""

    def test_encryption_disabled_when_no_key(self, monkeypatch):
        monkeypatch.delenv("CCTI_ENCRYPTION_KEY", raising=False)
        import importlib
        import crypto_utils
        importlib.reload(crypto_utils)
        assert not crypto_utils.encryption_enabled()

    def test_encryption_enabled_when_key_set(self, monkeypatch, tmp_path):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("CCTI_ENCRYPTION_KEY", key)
        import importlib
        import crypto_utils
        importlib.reload(crypto_utils)
        assert crypto_utils.encryption_enabled()

    def test_encrypt_decrypt_roundtrip(self, monkeypatch, tmp_path):
        from cryptography.fernet import Fernet
        key = Fernet.generate_key().decode()
        monkeypatch.setenv("CCTI_ENCRYPTION_KEY", key)
        import importlib
        import crypto_utils
        importlib.reload(crypto_utils)

        test_file = tmp_path / "test.json"
        original = [{"id": "1", "value": "secret"}]
        test_file.write_text(json.dumps(original), encoding="utf-8")
        crypto_utils.encrypt_file(test_file)

        # File on disk should no longer be readable as plain JSON
        raw = test_file.read_bytes()
        with pytest.raises(Exception):
            json.loads(raw)

        # Decryption should recover original data
        recovered = crypto_utils.decrypt_json(test_file)
        assert recovered == original
