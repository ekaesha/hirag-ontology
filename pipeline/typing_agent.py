import os
import json
from openai import OpenAI
from dotenv import load_dotenv
from pipeline.knowledge_graph import KnowledgeGraph

load_dotenv()

from pipeline.ontology_loader import (
    get_classes, get_properties,
    get_valid_types, get_valid_predicates,
    load_ontology
)

# Загружаем из ontology.json
ONTOLOGY_CLASSES    = get_classes()
ONTOLOGY_PROPERTIES = get_properties()
VALID_TYPES         = get_valid_types()

TYPING_PROMPT = """You are a medical ontology typing system.

Given an entity label and description, assign it to exactly ONE class.

Ontology classes:
{classes}

Entity label: "{label}"
Entity description: "{description}"

Return ONLY valid JSON:
{{"class": "<ClassName>", "confidence": <0.0-1.0>}}

Rules:
- Use ONLY class names from the list
- If unsure, use "Other"
- confidence reflects your certainty"""


class TypingAgent:
    def __init__(self, model="deepseek-chat"):
        self.client = OpenAI(
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com",
        )
        self.model = model
        self._cache = {}

    def _format_classes(self):
        return "\n".join(
            f"  {name}: {desc}"
            for name, desc in ONTOLOGY_CLASSES.items()
        )

    def type_entity(self, label, description=""):
        cache_key = label.lower()
        if cache_key in self._cache:
            return self._cache[cache_key]

        prompt = TYPING_PROMPT.format(
            classes=self._format_classes(),
            label=label,
            description=description or "No description available",
        )

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
            )
            data = json.loads(response.choices[0].message.content)
            cls = data.get("class", "Other")
            conf = float(data.get("confidence", 0.8))
            if cls not in ONTOLOGY_CLASSES:
                cls = "Other"
            self._cache[cache_key] = (cls, conf)
            return cls, conf
        except Exception as e:
            print(f"  [Typing] Error for '{label}': {e}")
            return "Other", 0.0

    def type_graph(self, kg):
        print(f"  [Typing] Typing {len(kg.entities)} entities...")
        type_distribution = {cls: 0 for cls in ONTOLOGY_CLASSES}
        typed_count = 0

        for i, (eid, entity) in enumerate(kg.entities.items()):
            cls, conf = self.type_entity(
                entity.label, entity.description
            )
            entity.entity_type = cls
            type_distribution[cls] = type_distribution.get(cls, 0) + 1
            typed_count += 1
            if (i + 1) % 20 == 0:
                print(f"  [Typing] {i+1}/{len(kg.entities)} done")

        print(f"  [Typing] Done. Distribution:")
        for cls, count in type_distribution.items():
            if count > 0:
                print(f"    {cls}: {count}")

        return {
            "typed": typed_count,
            "type_distribution": type_distribution,
        }