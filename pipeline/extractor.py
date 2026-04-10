import os
import json
import hashlib
import time
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.knowledge_graph import KnowledgeGraph, Entity, Relation

load_dotenv()

ENTITY_TYPES = [
    "Drug", "Condition", "Procedure", "Symptom",
    "AnatomicalStructure", "DosageRegimen", "LabTest",
    "Organization", "Other",
]

RELATION_TYPES = [
    "treats", "causes", "contraindicated_for",
    "part_of", "diagnosed_by", "dosage_is", "related_to",
]

EXTRACTION_PROMPT = """You are a medical knowledge extraction system.
Extract all entities and relations from the given text.

Entity types: {entity_types}
Relation types: {relation_types}

Return ONLY valid JSON:
{{
  "entities": [
    {{"label": "entity name", "type": "EntityType", "description": "brief description"}}
  ],
  "relations": [
    {{"subject": "label", "predicate": "relation_type", "object": "label", "confidence": 0.9}}
  ]
}}

Rules:
- Use exact type names from the lists
- Keep descriptions under 20 words
- confidence is float 0.0-1.0

Text:
{text}"""


class ExtractionAgent:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = model
        self._cache = {}

    def extract(self, text, chunk_id=""):
        h = hashlib.md5(text.encode()).hexdigest()
        if h in self._cache:
            return self._cache[h]

        prompt = EXTRACTION_PROMPT.format(
            entity_types=", ".join(ENTITY_TYPES),
            relation_types=", ".join(RELATION_TYPES),
            text=text[:3000],
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
        except Exception as e:
            print(f"  [Extraction] Error on chunk {chunk_id}: {e}")
            return [], []

        entities = []
        label_to_id = {}
        for item in data.get("entities", []):
            label = item.get("label", "").strip()
            if not label:
                continue
            eid = hashlib.md5(label.lower().encode()).hexdigest()[:12]
            label_to_id[label.lower()] = eid
            entities.append(Entity(
                id=eid,
                label=label,
                entity_type=item.get("type", "Other"),
                description=item.get("description", ""),
                source_chunks=[chunk_id] if chunk_id else [],
            ))

        relations = []
        for item in data.get("relations", []):
            subj_id = label_to_id.get(
                item.get("subject", "").strip().lower()
            )
            obj_id = label_to_id.get(
                item.get("object", "").strip().lower()
            )
            if subj_id and obj_id and subj_id != obj_id:
                relations.append(Relation(
                    subject_id=subj_id,
                    predicate=item.get("predicate", "related_to"),
                    object_id=obj_id,
                    confidence=float(item.get("confidence", 0.8)),
                    source_chunk=chunk_id,
                ))

        self._cache[h] = (entities, relations)
        return entities, relations


def chunk_text(text, chunk_size=800, overlap=100):
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunks.append(" ".join(words[start:end]))
        if end == len(words):
            break
        start += chunk_size - overlap
    return chunks


def build_graph_from_text(text, agent):
    kg = KnowledgeGraph()
    chunks = chunk_text(text)
    print(f"    {len(chunks)} chunks...")
    for i, chunk in enumerate(chunks):
        entities, relations = agent.extract(chunk, f"chunk_{i}")
        for e in entities:
            kg.add_entity(e)
        for r in relations:
            kg.add_relation(r)
        if i % 5 == 4:
            time.sleep(0.5)
    return kg