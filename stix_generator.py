import json
import uuid
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("STIX_Generator")


def generate_stix_bundle(input_path, output_path):
    """Transform normalised NER results into a STIX 2.1 Bundle."""
    logger.info("Reading intelligence from %s...", input_path)
    try:
        with open(input_path, 'r') as f:
            events = json.load(f)
    except Exception as e:
        logger.error("Failed to load input: %s", e)
        return

    identity_id = f"identity--{uuid.uuid4()}"
    identity = {
        "type": "identity",
        "spec_version": "2.1",
        "id": identity_id,
        "created": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "modified": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "name": "CCTI AI-Pipeline",
        "description": "Automated CTI Curation using DeepSeek-R1 and spaCy NER.",
        "identity_class": "system"
    }

    stix_objects = [identity]

    for event in events:
        timestamp = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
        entities = event.get("entities", {})
        raw_text = event.get("raw_text", "")
        iocs = entities.get("iocs", {})

        ips    = iocs.get("ips", [])
        hashes = iocs.get("hashes", {})
        urls   = iocs.get("urls", [])

        if ips:
            pattern = f"[network-traffic:dst_ref.type = 'ipv4-addr' AND network-traffic:dst_ref.value = '{ips[0]}']"
        elif hashes.get("sha256"):
            pattern = f"[file:hashes.'SHA-256' = '{hashes['sha256'][0]}']"
        elif hashes.get("md5"):
            pattern = f"[file:hashes.'MD5' = '{hashes['md5'][0]}']"
        elif urls:
            pattern = f"[url:value = '{urls[0].replace(chr(39), chr(92) + chr(39))}']"
        else:
            pattern = "[network-traffic:dst_ref.type = 'ipv4-addr']"

        stix_objects.append({
            "type": "indicator",
            "spec_version": "2.1",
            "id": f"indicator--{uuid.uuid4()}",
            "created_by_ref": identity_id,
            "created": timestamp,
            "modified": timestamp,
            "name": event.get("title", "Threat Intelligence Entry"),
            "description": entities.get("summary", "No summary available."),
            "indicator_types": ["malicious-activity"],
            "pattern": pattern,
            "pattern_type": "stix",
            "pattern_version": "2.1",
            "valid_from": timestamp,
            "labels": entities.get("technologies", []) + entities.get("industries", []) + entities.get("attack_techniques", [])
        })

        if "CVE-" in raw_text:
            stix_objects.append({
                "type": "vulnerability",
                "spec_version": "2.1",
                "id": f"vulnerability--{uuid.uuid4()}",
                "created_by_ref": identity_id,
                "created": timestamp,
                "modified": timestamp,
                "name": "Extracted CVE Reference",
                "description": f"Vulnerability detected in event: {event.get('event_id')}"
            })

    bundle = {
        "type": "bundle",
        "id": f"bundle--{uuid.uuid4()}",
        "objects": stix_objects
    }

    with open(output_path, 'w') as f:
        json.dump(bundle, f, indent=4)

    logger.info("STIX bundle with %d objects -> %s", len(stix_objects), output_path)


if __name__ == "__main__":
    generate_stix_bundle(
        "ner/ner_results_v3_normalised.json",
        "cti_microservice_v1/curated_threats_stix3.json"
    )
