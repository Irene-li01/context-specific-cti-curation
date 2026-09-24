import json
import re

# Constant definitions for the input corpus and the output serialized dataset
INPUT_FILE = "cti_ner_results.json"
OUTPUT_FILE = "secbert_train_data.json"

def build_bio_dataset():
    print(f"Initializing character-level alignment pipeline for Named Entity Recognition (NER). Reading corpus from: {INPUT_FILE}...\n")
    
    bio_dataset = []
    
    # Mapping schema for entity categorization
    tag_map = {
        "technologies": "TECH",
        "threat_actors": "ACTOR",
        "attack_techniques": "TTP",
        "industries": "IND"
    }

    success_count = 0
    total_entities_found = 0
    seen_event_ids = set() # Utilized to enforce deduplication of instances

    # Load the comprehensive JSON array into memory
    with open(INPUT_FILE, 'r', encoding='utf-8') as f:
        data = json.load(f)
        
    for item in data:
        # ==========================================
        # Implement event deduplication to mitigate potential redundancy 
        # introduced during the automated data collection phase
        # ==========================================
        event_id = item.get("event_id")
        if event_id in seen_event_ids:
            continue
        seen_event_ids.add(event_id)

        entities = item.get("entities", {})
        if entities.get("parse_error"):
            continue

        # Synthesize the contextual sequence by concatenating the title and summary fields
        title = item.get("title", "").strip()
        summary = entities.get("summary", "").strip()
        text = f"{title}. {summary}".strip()
        
        if not text or text == ".":
            continue

        # Phase 1: Character-level annotation 
        # Incorporates a mutual-exclusion mechanism to prevent overlapping entity annotations
        char_tags = ["O"] * len(text)
        
        for category, tag_name in tag_map.items():
            entity_list = entities.get(category, [])
            for ent in entity_list:
                if not ent.strip(): 
                    continue
                
                pattern = re.compile(re.escape(ent), re.IGNORECASE)
                
                for match in pattern.finditer(text):
                    start, end = match.span()
                    
                    # Validate that the current character span has not been previously annotated
                    if all(char_tags[i] == "O" for i in range(start, end)):
                        char_tags[start] = f"B-{tag_name}"
                        for i in range(start + 1, end):
                            char_tags[i] = f"I-{tag_name}"
                        total_entities_found += 1

        # Phase 2: Token-level projection
        # Aligns character-level annotations with the tokenized sequence
        tokens = []
        tags = []
        
        for match in re.finditer(r'\w+|[^\w\s]', text):
            start, end = match.span()
            tokens.append(match.group())
            # Assign the corresponding BIO tag based on the initial character's annotation
            tags.append(char_tags[start])
            
        bio_dataset.append({
            "tokens": tokens,
            "ner_tags": tags
        })
        success_count += 1
        
    print(f"\nData processing sequence finalized. Successfully generated {success_count} aligned training instances (Total entities successfully mapped: {total_entities_found}).")
    print(f"Serializing output to: {OUTPUT_FILE}...")
    
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(bio_dataset, f, indent=2, ensure_ascii=False)
        
    print("Serialization complete. The dataset is structured and prepared for model training.")

if __name__ == "__main__":
    build_bio_dataset()