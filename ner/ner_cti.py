import argparse
import json
import re
import sys
from pathlib import Path
import spacy
from spacy.matcher import PhraseMatcher

# Path to the fine-tuned SecBERT model
_SECBERT_MODEL_PATH = str(
    Path(__file__).resolve().parent.parent / "Machine learning" / "secbert_cti_model_final"
)

# SecBERT label → our entity bucket mapping
_SECBERT_LABEL_MAP = {
    "ACTOR": "threat_actors",
    "IND":   "industries",
    "TECH":  "technologies",
    "TTP":   "attack_techniques",
}

# Lazy-loaded SecBERT model (only loaded when --use-secbert is passed)
_secbert_pipeline = None


def _load_secbert():
    """Load the fine-tuned SecBERT NER pipeline (once, on first call)."""
    global _secbert_pipeline
    if _secbert_pipeline is not None:
        return _secbert_pipeline
    try:
        from transformers import pipeline as hf_pipeline
        print(f"Loading SecBERT model from {_SECBERT_MODEL_PATH}...")
        _secbert_pipeline = hf_pipeline(
            "token-classification",
            model=_SECBERT_MODEL_PATH,
            tokenizer=_SECBERT_MODEL_PATH,
            aggregation_strategy="first",   # uses first subword token label per word
            device=0 if __import__("torch").cuda.is_available() else -1,  # GPU if available
        )
        print("SecBERT model loaded.")
        return _secbert_pipeline
    except Exception as e:
        print(f"WARNING: Could not load SecBERT model: {e}", file=sys.stderr)
        return None


def run_secbert_ner(text: str, entities: dict) -> None:
    """
    Run SecBERT inference on text and merge results into the entities dict.
    Confidence is set to 0.80 — above spaCy (0.75) but below hard dictionary (0.90)
    since the training data was derived from LLM output rather than ground truth.
    """
    model = _load_secbert()
    if model is None:
        return

    try:
        predictions = model(text[:512])  # BERT max 512 tokens
    except Exception as e:
        print(f"WARNING: SecBERT inference failed: {e}", file=sys.stderr)
        return

    for pred in predictions:
        raw_label = pred.get("entity_group", "")
        # Strip B- / I- prefix if aggregation_strategy didn't (safety net)
        label = raw_label.lstrip("BI-").strip()
        bucket = _SECBERT_LABEL_MAP.get(label)
        if not bucket:
            continue

        span = pred.get("word", "").strip()
        if len(span) < 2:
            continue

        score = float(pred.get("score", 0.80))
        confidence = round(min(max(score, 0.50), 0.95), 3)

        add_entity(entities, bucket, span, "secbert", confidence)

nlp = spacy.load("en_core_web_sm")

threat_actor_terms = [
    "APT29", "APT28", "Turla", "Sofacy", "Lazarus", "Black Vine", "Packrat",
    "FIN7", "Cobalt Group", "Carbanak"
]

malware_terms = [
    "Dridex", "Locky", "Fysbis", "PETYA", "WannaCry", "PlugX",
    "Bookworm", "TeslaCrypt", "BlackEnergy", "SeaDuke", "SAMSAM"
]

tool_terms = [
    "HammerToss", "EternalBlue"
]

technology_terms = [
    "SWIFT Gateway", "SWIFT Alliance Gateway", "Alliance Gateway",
    "Oracle Database", "Oracle Database Server", "Oracle WebLogic", "Apache Solr",
    "IBM Mainframe z15", "SAP Banking Core", "Palo Alto Firewall",
    "Microsoft Active Directory", "Siemens WinCC", "Power Grid Controller",
    "Rockwell Automation PLC", "SAP ERP", "Microsoft Windows Server",
    "Siemens S7 PLC", "NCR POS System", "SAP Retail", "Cisco Meraki Network",
    "Microsoft Azure", "Toast POS System", "Oracle MICROS POS",
    "Windows 10 Workstation", "Lightspeed Inventory", "Cerner EHR Platform",
    "MySQL Database", "VMware vSphere", "Splunk SIEM",
    "Stripe Payment Gateway", "PostgreSQL Database", "AWS Cloud Infrastructure",
    "Node.js API Server", "Siemens MES", "Microsoft SQL Server",
    "GE Grid Solutions EMS", "OSIsoft PI Historian", "Cisco Industrial Switch",
    "Epic EHR System", "Philips Patient Monitoring", "Microsoft Windows 10",
    "Cisco Network Switch", "SMA Solar Inverter", "Schneider Electric SCADA",
    "SCADA", "PLC", "Windows Server", "SQL Server", "Active Directory",
    "AWS", "Azure", "VMware", "Cisco", "Siemens", "SAP"
]

industry_terms = [
    "government", "healthcare", "finance", "energy", "aerospace", "military"
]

attack_terms = [
    "phishing", "ransomware", "watering hole", "backdoor",
    "malware injection", "privilege escalation", "credential dumping",
    "brute force", "credential access", "password spraying", "credential stuffing"
]

matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
term_lookup = {}

def add_patterns(label, terms):
    patterns = [nlp.make_doc(term) for term in terms]
    matcher.add(label, patterns)
    term_lookup[label] = {term.lower(): term for term in terms}

add_patterns("THREAT_ACTOR", threat_actor_terms)
add_patterns("MALWARE", malware_terms)
add_patterns("TOOL", tool_terms)
add_patterns("TECHNOLOGY", technology_terms)
add_patterns("INDUSTRY", industry_terms)
add_patterns("ATTACK_TECHNIQUE", attack_terms)

cve_pattern = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.IGNORECASE)
mitre_pattern = re.compile(r"\bT\d{4}(?:\.\d{3})?\b", re.IGNORECASE)
url_pattern = re.compile(r"\bhttps?://[^\s|,;\"']+", re.IGNORECASE)
domain_pattern = re.compile(
    r"\b(?![\w.-]*@)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}\b",
    re.IGNORECASE,
)
ip_pattern = re.compile(
    r"\b(?:(?:25[0-5]|2[0-4]\d|1?\d?\d)\.){3}(?:25[0-5]|2[0-4]\d|1?\d?\d)\b"
)
email_pattern = re.compile(r"\b[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}\b", re.IGNORECASE)
md5_pattern = re.compile(r"\b[a-f0-9]{32}\b", re.IGNORECASE)
sha1_pattern = re.compile(r"\b[a-f0-9]{40}\b", re.IGNORECASE)
sha256_pattern = re.compile(r"\b[a-f0-9]{64}\b", re.IGNORECASE)

LLM_PROMPT = """You are a cybersecurity threat intelligence NER assistant.
Extract only entities that are explicitly present in the text. Return JSON only.

Schema:
{{
  "technologies": [],
  "industries": [],
  "threat_actors": [],
  "malware": [],
  "tools": [],
  "attack_techniques": [],
  "mitre_techniques": [],
  "vulnerabilities": [],
  "severity": "unknown",
  "summary": ""
}}

Text:
{text}
"""


parser = argparse.ArgumentParser(description="Run CTI NER with optional LLM enrichment.")
parser.add_argument("--input", default="ner/cti_ner_input.json", help="NER input JSON path.")
parser.add_argument("--output", default="ner/ner_results_v3.json", help="NER output JSON path.")
parser.add_argument("--use-llm", action="store_true", help="Enable Ollama/DeepSeek enrichment.")
parser.add_argument("--llm-model", default="deepseek-r1:14b", help="Ollama model name.")
parser.add_argument("--llm-limit", type=int, default=25, help="Maximum records to enrich with LLM.")
parser.add_argument("--llm-all", action="store_true", help="Send every record to the LLM until --llm-limit.")
parser.add_argument("--use-secbert", action="store_true", help="Enable fine-tuned SecBERT NER as an additional extraction layer.")
args = parser.parse_args()


def add_unique(items, value):
    value = value.strip()
    if not value:
        return False
    existing = {item.lower() for item in items}
    if value.lower() not in existing:
        items.append(value)
        return True
    return False


def add_many_unique(items, values):
    for value in values:
        add_unique(items, value)


def add_detail(details, bucket, text, source, confidence):
    text = text.strip()
    if not text:
        return
    existing = {(d["text"].lower(), d["bucket"]) for d in details}
    key = (text.lower(), bucket)
    if key in existing:
        return
    details.append({
        "text": text,
        "bucket": bucket,
        "source": source,
        "confidence": confidence
    })


def add_entity(entities, bucket, text, source, confidence):
    if add_unique(entities[bucket], text):
        add_detail(entities["entity_details"], bucket, text, source, confidence)


def add_ioc(entities, ioc_bucket, text, source="regex_ioc", confidence=0.95):
    if add_unique(entities["iocs"][ioc_bucket], text):
        add_detail(entities["entity_details"], f"iocs.{ioc_bucket}", text, source, confidence)


def add_hash(entities, hash_type, text):
    if add_unique(entities["iocs"]["hashes"][hash_type], text):
        add_detail(entities["entity_details"], f"iocs.hashes.{hash_type}", text, "regex_hash", 0.98)


def strip_llm_json(raw):
    raw = raw.strip()
    if "<think>" in raw and "</think>" in raw:
        raw = raw.split("</think>", 1)[-1].strip()
    if "```" in raw:
        for part in raw.split("```"):
            part = part.strip().removeprefix("json").strip()
            if part.startswith("{"):
                raw = part
                break
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        raw = raw[start:end + 1]
    return raw


def run_llm_ner(text, model):
    try:
        import ollama
        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": LLM_PROMPT.format(text=text)}],
        )
        raw = response["message"]["content"]
        return json.loads(strip_llm_json(raw))
    except Exception as exc:
        return {"_llm_error": str(exc)}


def should_enrich_with_llm(entities, attr_type):
    if args.llm_all:
        return True
    semantic_signal = any([
        entities["threat_actors"],
        entities["malware"],
        entities["tools"],
        entities["industries"],
        entities["attack_techniques"],
        entities["vulnerabilities"],
    ])
    # LLM is most useful on descriptive text. It is usually wasteful for pure hashes.
    descriptive_types = {
        "text", "comment", "campaign-id", "campaign-name", "malware-sample",
        "snort", "sigma", "stix2-pattern", "target-external", "target-location",
        "target-machine", "target-org", "target-user", "vulnerability", "yara",
    }
    return attr_type in descriptive_types and not semantic_signal


def merge_llm_entities(entities, llm_result):
    if not isinstance(llm_result, dict) or llm_result.get("_llm_error"):
        if isinstance(llm_result, dict) and llm_result.get("_llm_error"):
            entities.setdefault("llm_errors", []).append(llm_result["_llm_error"])
        return

    for bucket in [
        "technologies", "industries", "threat_actors", "malware", "tools",
        "attack_techniques", "mitre_techniques", "vulnerabilities",
    ]:
        values = llm_result.get(bucket, [])
        if not isinstance(values, list):
            continue
        for value in values:
            text = str(value).strip()
            if len(text) < 2:
                continue
            add_entity(entities, bucket, text, "llm", 0.70)
            if bucket == "mitre_techniques":
                add_entity(entities, "attack_techniques", text, "llm", 0.70)

    severity = str(llm_result.get("severity", "")).lower()
    if severity in {"low", "medium", "high", "critical"}:
        entities["llm_severity"] = severity

    summary = str(llm_result.get("summary", "")).strip()
    if summary:
        entities["llm_summary"] = summary


with open(args.input, "r", encoding="utf-8") as f:
    data = json.load(f)

results = []
llm_used = 0

allowed_spacy_labels = {"ORG", "PRODUCT"}

for item in data:
    text = item["raw_text"]
    title = item.get("title") or item.get("event_info") or text
    description = item.get("description", "")
    attr_type = item.get("type", "").strip().lower()
    indicator = item.get("indicator", "").strip()
    category = item.get("category", "").strip()
    doc = nlp(text)

    seen = set()

    structured_entities = {
        "technologies": [],
        "malware": [],
        "tools": [],
        "industries": [],
        "threat_actors": [],
        "attack_techniques": [],
        "mitre_techniques": [],
        "vulnerabilities": [],
        "iocs": {
            "ips": [],
            "domains": [],
            "urls": [],
            "emails": [],
            "hashes": {
                "md5": [],
                "sha1": [],
                "sha256": []
            }
        },
        "entity_details": [],
        "severity": "unknown",
        "summary": ""
    }

    if attr_type == "threat-actor":
        add_entity(structured_entities, "threat_actors", indicator, "misp_type", 0.95)
    elif attr_type == "vulnerability":
        add_entity(structured_entities, "vulnerabilities", indicator, "misp_type", 0.95)

    # 1. spaCy generic entities
    for ent in doc.ents:
        if ent.label_ not in allowed_spacy_labels:
            continue

        text_clean = ent.text.strip()

        if len(text_clean) < 3:
            continue
        if text_clean.upper() == "OSINT":
            continue
        if " - " in text_clean:
            continue
        if ":" in text_clean:
            continue

        key = (text_clean, ent.label_)
        if key in seen:
            continue
        seen.add(key)

        if ent.label_ == "ORG":
            if text_clean in threat_actor_terms:
                add_entity(structured_entities, "threat_actors", text_clean, "spacy_org_dictionary", 0.75)

    # 2. custom CTI patterns
    for match_id, start, end in matcher(doc):
        label = nlp.vocab.strings[match_id]
        span = doc[start:end]
        text_clean = term_lookup.get(label, {}).get(span.text.strip().lower(), span.text.strip())

        key = (text_clean, label)
        if key in seen:
            continue
        seen.add(key)

        if label == "THREAT_ACTOR":
            add_entity(structured_entities, "threat_actors", text_clean, "dictionary", 0.90)
        elif label == "MALWARE":
            add_entity(structured_entities, "malware", text_clean, "dictionary", 0.90)
        elif label == "TOOL":
            add_entity(structured_entities, "tools", text_clean, "dictionary", 0.90)
        elif label == "TECHNOLOGY":
            add_entity(structured_entities, "technologies", text_clean, "dictionary", 0.90)
        elif label == "INDUSTRY":
            add_entity(structured_entities, "industries", text_clean, "dictionary", 0.90)
        elif label == "ATTACK_TECHNIQUE":
            add_entity(structured_entities, "attack_techniques", text_clean, "dictionary", 0.85)

    # 3. Regex extraction for structured CTI entities
    for cve in cve_pattern.findall(text):
        cve = cve.upper()
        add_entity(structured_entities, "vulnerabilities", cve, "regex_cve", 0.98)

    for technique in mitre_pattern.findall(text):
        technique = technique.upper()
        add_entity(structured_entities, "mitre_techniques", technique, "regex_mitre", 0.98)
        add_entity(structured_entities, "attack_techniques", technique, "regex_mitre", 0.98)

    urls = url_pattern.findall(text)
    emails = email_pattern.findall(text)
    for url in urls:
        add_ioc(structured_entities, "urls", url)
    for email in emails:
        add_ioc(structured_entities, "emails", email)
    for ip in ip_pattern.findall(text):
        add_ioc(structured_entities, "ips", ip)

    hash_text = text
    for noisy_token in urls + emails:
        hash_text = hash_text.replace(noisy_token, " ")
    for value in md5_pattern.findall(hash_text):
        add_hash(structured_entities, "md5", value)
    for value in sha1_pattern.findall(hash_text):
        add_hash(structured_entities, "sha1", value)
    for value in sha256_pattern.findall(hash_text):
        add_hash(structured_entities, "sha256", value)

    # Domains are useful, but avoid duplicating domains already inside URLs/emails.
    url_domains = set()
    for url in structured_entities["iocs"]["urls"]:
        url_domains.update(domain_pattern.findall(url))
    email_domains = {email.rsplit("@", 1)[-1].lower() for email in structured_entities["iocs"]["emails"]}
    for domain in domain_pattern.findall(text):
        if domain.lower() in url_domains or domain.lower() in email_domains:
            continue
        add_ioc(structured_entities, "domains", domain, "regex_domain", 0.85)

    if args.use_llm and llm_used < args.llm_limit and should_enrich_with_llm(structured_entities, attr_type):
        llm_used += 1
        llm_result = run_llm_ner(text, args.llm_model)
        merge_llm_entities(structured_entities, llm_result)

    # 3b. SecBERT NER (optional, enabled with --use-secbert)
    if args.use_secbert:
        run_secbert_ner(text, structured_entities)

    # 4. deduplicate
    for k in [
        "technologies",
        "malware",
        "tools",
        "industries",
        "threat_actors",
        "attack_techniques",
        "mitre_techniques",
        "vulnerabilities"
    ]:
        structured_entities[k] = list(dict.fromkeys(structured_entities[k]))

    # 5. rule-based severity
    if structured_entities.get("llm_severity") in {"critical", "high"}:
        structured_entities["severity"] = structured_entities["llm_severity"]
    elif structured_entities["threat_actors"] and structured_entities["attack_techniques"]:
        structured_entities["severity"] = "high"
    elif structured_entities["threat_actors"] or structured_entities["technologies"]:
        structured_entities["severity"] = "medium"
    else:
        structured_entities["severity"] = "low"

    # 6. summary
    summary_parts = []
    if structured_entities["threat_actors"]:
        summary_parts.append(f"Threat actors: {', '.join(structured_entities['threat_actors'])}")
    if structured_entities["malware"]:
        summary_parts.append(f"Malware: {', '.join(structured_entities['malware'])}")
    if structured_entities["tools"]:
        summary_parts.append(f"Tools: {', '.join(structured_entities['tools'])}")
    if structured_entities["technologies"]:
        summary_parts.append(f"Technologies: {', '.join(structured_entities['technologies'])}")
    if structured_entities["attack_techniques"]:
        summary_parts.append(f"Techniques: {', '.join(structured_entities['attack_techniques'])}")
    if structured_entities["vulnerabilities"]:
        summary_parts.append(f"Vulnerabilities: {', '.join(structured_entities['vulnerabilities'])}")
    if structured_entities["industries"]:
        summary_parts.append(f"Industries: {', '.join(structured_entities['industries'])}")

    structured_entities["summary"] = "; ".join(summary_parts)

    results.append({
        "event_id": item["event_id"],
        "attr_id": item.get("attr_id", ""),
        "category": category,
        "type": attr_type,
        "indicator": indicator,
        "title": title,
        "description": description,
        "embedding_text": item.get("embedding_text", text),
        "raw_text": text,
        "entities": structured_entities
    })

with open(args.output, "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2, ensure_ascii=False)

print("NER finished!")
print(f"Saved to {args.output}")
if args.use_llm:
    print(f"LLM enrichment used on {llm_used} record(s).")
