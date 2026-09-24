from collections import Counter

import pytest

import recommendation_engine as rec


def test_mitre_score_downweights_profile_wide_common_techniques(monkeypatch):
    monkeypatch.setattr(
        rec,
        "_profile_match_frequencies",
        lambda: (Counter({"T1566": 11, "T1190": 1}), Counter()),
    )

    common_profile = {"risk_profile": {"threat_landscape": ["T1566 - Phishing"]}}
    rare_profile = {"risk_profile": {"threat_landscape": ["T1190 - Exploit Public-Facing Application"]}}

    common_event = {"attack_techniques": ["T1566"], "entity_details": []}
    rare_event = {"attack_techniques": ["T1190"], "entity_details": []}

    common_score = rec.compute_mitre_score(common_profile, common_event)
    rare_score = rec.compute_mitre_score(rare_profile, rare_event)

    assert common_score < rare_score
    assert rare_score == pytest.approx(0.14)


def test_mitre_score_ignores_generic_actor_terms(monkeypatch):
    monkeypatch.setattr(
        rec,
        "_profile_match_frequencies",
        lambda: (Counter(), Counter({"fin7": 4})),
    )

    profile = {"risk_profile": {"threat_landscape": ["phishing", "FIN7"]}}
    event = {"threat_actors": ["phishing", "FIN7"], "attack_techniques": [], "entity_details": []}

    score = rec.compute_mitre_score(profile, event)

    assert score > 0
    assert score < 0.3
