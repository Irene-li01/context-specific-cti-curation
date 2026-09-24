"""Unit tests for ``filtering.cti_filter``.

These cover each of the four acceptance criteria from the user story:

  1. Irrelevant CTI records (whois / contact) are filtered out.
  2. Duplicates / low-quality entries are removed.
  3. Text fields are normalised.
  4. Cleaned dataset is generated and saved successfully.

Tests are intentionally hermetic — they build small in-memory inputs and
never read from the live ``data/`` folder, so they pass on a fresh clone.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

# When the package is checked out on a developer machine the tests are run
# from the repo root: ``pytest filtering/tests``. Importing via the package
# path keeps tests resilient to the CWD.
from filtering.cti_filter import CTIFilter, FilterConfig
from filtering import cti_filter as cf


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _attr(**overrides):
    """Return a canonical attribute with sensible defaults; overrides win."""
    base = {
        "id": "1", "uuid": "uuid-1",
        "type": "ip-dst", "category": "Network activity",
        "value": "8.8.8.8", "to_ids": True,
        "comment": "", "timestamp": "1700000000",
        "event_id": "1", "event_info": "feed",
        "tags": [],
    }
    base.update(overrides)
    return base


# --------------------------------------------------------------------------- #
# AC1 — irrelevant records filtered out
# --------------------------------------------------------------------------- #
class TestNoiseFiltering:
    def test_drops_person_category(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(category="Person", value="alice@x.com")])
        assert out == []
        assert cti.drop_counts["noise_category"] == 1

    def test_drops_whois_type(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(type="whois-registrant-email", value="x@y.com")])
        assert out == []
        assert cti.drop_counts["noise_type"] == 1

    def test_drops_email_subject_type(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(type="email-subject", value="Invoice attached")])
        assert out == []

    def test_keeps_legitimate_network_activity(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(type="ip-dst", value="8.8.8.8")])
        assert len(out) == 1


# --------------------------------------------------------------------------- #
# AC2 — duplicates / low-quality removed
# --------------------------------------------------------------------------- #
class TestDuplicates:
    def test_dedup_by_type_and_value(self):
        cti = CTIFilter()
        a = _attr(id="1", value="1.1.1.1", timestamp="100")
        b = _attr(id="2", value="1.1.1.1", timestamp="200")  # newer
        out = cti.filter_attributes([a, b])
        assert len(out) == 1
        assert out[0]["id"] == "2"  # latest wins by default
        assert cti.drop_counts["duplicate"] == 1

    def test_dedup_first_strategy(self):
        cti = CTIFilter(FilterConfig(dedup_strategy="first"))
        a = _attr(id="1", value="1.1.1.1", timestamp="100")
        b = _attr(id="2", value="1.1.1.1", timestamp="200")
        out = cti.filter_attributes([a, b])
        assert out[0]["id"] == "1"

    def test_drops_empty_value(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(value="")])
        assert out == []
        assert cti.drop_counts["empty_or_short_value"] == 1

    def test_drops_invalid_ip(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(type="ip-dst", value="not.an.ip")])
        assert out == []
        assert cti.drop_counts["invalid_ip"] == 1

    def test_drops_private_ip_by_default(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(type="ip-dst", value="192.168.1.1")])
        assert out == []
        assert cti.drop_counts["private_ip"] == 1

    def test_keeps_private_ip_when_disabled(self):
        cti = CTIFilter(FilterConfig(drop_private_ips=False))
        out = cti.filter_attributes([_attr(type="ip-dst", value="192.168.1.1")])
        assert len(out) == 1

    def test_drops_invalid_hash_length(self):
        cti = CTIFilter()
        out = cti.filter_attributes([_attr(type="md5", value="deadbeef")])  # too short
        assert out == []
        assert cti.drop_counts["invalid_hash"] == 1

    def test_to_ids_required_drops_false_flag(self):
        cti = CTIFilter(FilterConfig(require_to_ids=True))
        out = cti.filter_attributes([_attr(to_ids=False)])
        assert out == []


# --------------------------------------------------------------------------- #
# AC3 — text normalisation
# --------------------------------------------------------------------------- #
class TestNormalisation:
    def test_refangs_url(self):
        cti = CTIFilter()
        out = cti.filter_attributes([
            _attr(type="url", value="hxxps://EVIL[.]com/path  ")
        ])
        assert out[0]["value"] == "https://evil.com/path"

    def test_lowercases_hash(self):
        cti = CTIFilter()
        h = "A" * 32  # valid md5 length, mixed-case
        out = cti.filter_attributes([_attr(type="md5", value=h)])
        assert out[0]["value"] == h.lower()

    def test_strips_control_chars(self):
        cti = CTIFilter()
        nasty = "evil.com\x00\x01"
        out = cti.filter_attributes([_attr(type="domain", value=nasty)])
        assert out[0]["value"] == "evil.com"


# --------------------------------------------------------------------------- #
# AC4 — pipeline writes cleaned dataset to disk
# --------------------------------------------------------------------------- #
class TestPipelineRunner:
    def test_discover_inputs_loads_json_and_xlsx_together(self, tmp_path: Path):
        data_dir = tmp_path / "data"
        misp_dir = data_dir / "misp"
        misp_dir.mkdir(parents=True)

        root_json = data_dir / "feed.json"
        misp_json = misp_dir / "misp_raw_events.json"
        legacy_xlsx = misp_dir / "misp_categorized.xlsx"
        root_json.write_text("[]", encoding="utf-8")
        misp_json.write_text("[]", encoding="utf-8")
        legacy_xlsx.write_text("legacy", encoding="utf-8")

        inputs = cf.discover_inputs(tmp_path)

        # All sources are loaded together to maximise data coverage
        assert root_json in inputs
        assert misp_json in inputs
        assert legacy_xlsx in inputs

    def test_discover_inputs_uses_xlsx_when_no_json_exists(self, tmp_path: Path):
        misp_dir = tmp_path / "data" / "misp"
        misp_dir.mkdir(parents=True)
        legacy_xlsx = misp_dir / "misp_categorized.xlsx"
        legacy_xlsx.write_text("legacy", encoding="utf-8")

        assert cf.discover_inputs(tmp_path) == [legacy_xlsx]

    def test_writes_cleaned_and_stats_files(self, tmp_path: Path):
        # Build a minimal repo layout under tmp_path.
        data_dir = tmp_path / "data"
        data_dir.mkdir()
        feed = {
            "response": [{
                "Event": {
                    "id": "1", "info": "test feed",
                    "Attribute": [
                        {"id": "1", "type": "ip-dst", "category": "Network activity",
                         "value": "8.8.8.8", "to_ids": True, "uuid": "u1", "timestamp": "1"},
                        {"id": "2", "type": "whois-registrant-email", "category": "Person",
                         "value": "x@y.com", "to_ids": False, "uuid": "u2", "timestamp": "2"},
                    ],
                },
            }]
        }
        (data_dir / "feed.json").write_text(json.dumps(feed), encoding="utf-8")

        cleaned_path, stats_path = cf.run_pipeline(
            repo_root=tmp_path,
            today=date(2026, 4, 29),
        )
        assert cleaned_path.exists() and stats_path.exists()
        cleaned = json.loads(cleaned_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        assert len(cleaned) == 1               # only the IP survived
        assert cleaned[0]["value"] == "8.8.8.8"
        assert stats["seen"] == 2 and stats["kept"] == 1
        assert stats["inputs"]["mode"] == "json_preferred"
        assert any("feed.json" in f for f in stats["inputs"]["used_files"])
        assert "noise_category" in stats["drop_reasons"] or \
               "noise_type" in stats["drop_reasons"]

    def test_raises_when_no_inputs(self, tmp_path: Path):
        with pytest.raises(FileNotFoundError):
            cf.run_pipeline(repo_root=tmp_path)
