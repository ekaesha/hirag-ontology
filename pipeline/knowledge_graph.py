import json
from dataclasses import dataclass, field
from typing import Optional
import networkx as nx


@dataclass
class Entity:
    id: str
    label: str
    entity_type: str
    description: str = ""
    aliases: list = field(default_factory=list)
    source_chunks: list = field(default_factory=list)
    embedding: Optional[list] = None


@dataclass
class Relation:
    subject_id: str
    predicate: str
    object_id: str
    confidence: float = 1.0
    source_chunk: str = ""


class KnowledgeGraph:
    def __init__(self):
        self.entities = {}
        self.relations = []
        self._graph = nx.DiGraph()
        self._pagerank = {}

    def add_entity(self, entity):
        if entity.id in self.entities:
            existing = self.entities[entity.id]
            for alias in entity.aliases:
                if alias not in existing.aliases:
                    existing.aliases.append(alias)
            return
        self.entities[entity.id] = entity
        self._graph.add_node(entity.id)

    def add_relation(self, relation):
        if relation.subject_id not in self.entities:
            return
        if relation.object_id not in self.entities:
            return
        self.relations.append(relation)
        self._graph.add_edge(
            relation.subject_id, relation.object_id,
            predicate=relation.predicate, weight=relation.confidence,
        )

    def stats(self):
        return {
            "nodes": len(self.entities),
            "edges": len(self.relations),
        }

    def compute_pagerank(self, damping=0.85):
        if not self._graph.nodes:
            return
        self._pagerank = nx.pagerank(self._graph, alpha=damping, max_iter=200)

    def get_pagerank(self, entity_id):
        return self._pagerank.get(entity_id, 0.0)

    def get_context_for_entities(self, entity_ids):
        lines = []
        seen = set()
        for eid in entity_ids:
            if eid not in self.entities:
                continue
            e = self.entities[eid]
            lines.append(f"Entity: {e.label} (type: {e.entity_type})")
            if e.description:
                lines.append(f"  Description: {e.description}")
            if e.aliases:
                lines.append(f"  Also known as: {', '.join(e.aliases)}")
            for rel in self.relations:
                key = (rel.subject_id, rel.predicate, rel.object_id)
                if key in seen:
                    continue
                if rel.subject_id == eid or rel.object_id == eid:
                    subj = self.entities.get(rel.subject_id)
                    obj = self.entities.get(rel.object_id)
                    if subj and obj:
                        lines.append(
                            f"  Relation: {subj.label} --[{rel.predicate}]--> {obj.label}"
                        )
                        seen.add(key)
        return "\n".join(lines) if lines else "No relevant entities found."

    def save(self, path):
        data = {
            "entities": [
                {"id": e.id, "label": e.label, "entity_type": e.entity_type,
                 "description": e.description, "aliases": e.aliases,
                 "source_chunks": e.source_chunks}
                for e in self.entities.values()
            ],
            "relations": [
                {"subject_id": r.subject_id, "predicate": r.predicate,
                 "object_id": r.object_id, "confidence": r.confidence,
                 "source_chunk": r.source_chunk}
                for r in self.relations
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"[KG] Saved: {len(self.entities)} entities, {len(self.relations)} relations")

    @classmethod
    def load(cls, path):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        kg = cls()
        for e in data["entities"]:
            kg.add_entity(Entity(**e))
        for r in data["relations"]:
            kg.add_relation(Relation(**r))
        print(f"[KG] Loaded: {len(kg.entities)} entities, {len(kg.relations)} relations")
        return kg