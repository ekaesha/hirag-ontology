import re
from difflib import SequenceMatcher
from pipeline.knowledge_graph import KnowledgeGraph, Entity

import sys
IN_COLAB = 'google.colab' in sys.modules


def _normalize(text):
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def _token_sort_ratio(a, b):
    a_sorted = " ".join(sorted(_normalize(a).split()))
    b_sorted = " ".join(sorted(_normalize(b).split()))
    return SequenceMatcher(None, a_sorted, b_sorted).ratio()


def _cosine_similarity(vec_a, vec_b):
    if not vec_a or not vec_b:
        return 0.0
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    norm_a = sum(a ** 2 for a in vec_a) ** 0.5
    norm_b = sum(b ** 2 for b in vec_b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _share_token(label_a, label_b):
    STOPWORDS = {"the", "a", "an", "of", "in", "for", "and", "or",
                 "с", "в", "и", "для", "при", "на", "по", "из"}
    tokens_a = set(_normalize(label_a).split()) - STOPWORDS
    tokens_b = set(_normalize(label_b).split()) - STOPWORDS
    return bool(tokens_a & tokens_b)


class DeduplicationAgent:
    def __init__(self, alpha=0.6, threshold=0.85, use_embeddings=True):
        self.alpha = alpha
        self.threshold = threshold
        self.use_embeddings = use_embeddings
        self._encoder = None

    def _get_encoder(self):
        if self._encoder is None:
            try:
                import os
                # In Colab: allow downloading from HuggingFace
                # Locally: use cached model only
                if not IN_COLAB:
                    #
                    #
                else:
                    #
                    #

                from sentence_transformers import SentenceTransformer
                print("  [Dedup] Loading model...")
                self._encoder = SentenceTransformer(
                    "paraphrase-multilingual-MiniLM-L12-v2",
                    local_files_only=False,
                )
                print("  [Dedup] Model loaded.")
            except Exception as e:
                print(f"  [Dedup] No embeddings model: {e}")
                self.use_embeddings = False
        return self._encoder

    def _compute_embeddings(self, kg):
        if not self.use_embeddings:
            return
        encoder = self._get_encoder()
        if encoder is None:
            return
        missing = [e for e in kg.entities.values() if e.embedding is None]
        if not missing:
            return
        labels = [e.label for e in missing]
        print(f"  [Dedup] Computing embeddings for {len(labels)} entities...")
        embeddings = encoder.encode(labels, batch_size=64, show_progress_bar=False)
        for entity, emb in zip(missing, embeddings):
            entity.embedding = emb.tolist()

    def similarity(self, entity_a, entity_b):
        sim_lex = _token_sort_ratio(entity_a.label, entity_b.label)
        if self.use_embeddings and entity_a.embedding and entity_b.embedding:
            sim_sem = _cosine_similarity(entity_a.embedding, entity_b.embedding)
        else:
            sim_sem = sim_lex
        return self.alpha * sim_sem + (1 - self.alpha) * sim_lex

    def find_duplicate_pairs(self, kg):
        self._compute_embeddings(kg)
        entities = list(kg.entities.values())
        pairs = []
        for i in range(len(entities)):
            for j in range(i + 1, len(entities)):
                a, b = entities[i], entities[j]
                if not _share_token(a.label, b.label):
                    continue
                score = self.similarity(a, b)
                if score >= self.threshold:
                    pairs.append((a.id, b.id, score))
        return pairs

    def deduplicate(self, kg):
        stats_before = kg.stats()
        print(f"  [Dedup] Before: {stats_before['nodes']} entities, "
              f"{stats_before['edges']} edges")

        pairs = self.find_duplicate_pairs(kg)
        print(f"  [Dedup] Found {len(pairs)} duplicate pairs")

        parent = {eid: eid for eid in kg.entities}

        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(x, y):
            px, py = find(x), find(y)
            if px != py:
                deg_x = kg._graph.degree(px) if px in kg._graph else 0
                deg_y = kg._graph.degree(py) if py in kg._graph else 0
                if deg_x >= deg_y:
                    parent[py] = px
                else:
                    parent[px] = py

        for id_a, id_b, _ in pairs:
            union(id_a, id_b)

        canonical_map = {}
        for eid in list(kg.entities.keys()):
            canon = find(eid)
            if canon != eid:
                canonical_map[eid] = canon

        for merged_id, canon_id in canonical_map.items():
            if merged_id not in kg.entities or canon_id not in kg.entities:
                continue
            merged = kg.entities[merged_id]
            canon = kg.entities[canon_id]
            if merged.label not in canon.aliases and merged.label != canon.label:
                canon.aliases.append(merged.label)
            for alias in merged.aliases:
                if alias not in canon.aliases:
                    canon.aliases.append(alias)
            for chunk in merged.source_chunks:
                if chunk not in canon.source_chunks:
                    canon.source_chunks.append(chunk)

        new_relations = []
        for rel in kg.relations:
            new_subj = canonical_map.get(rel.subject_id, rel.subject_id)
            new_obj = canonical_map.get(rel.object_id, rel.object_id)
            if new_subj == new_obj:
                continue
            rel.subject_id = new_subj
            rel.object_id = new_obj
            new_relations.append(rel)
        kg.relations = new_relations

        for merged_id in canonical_map:
            if merged_id in kg.entities:
                del kg.entities[merged_id]
            if merged_id in kg._graph:
                kg._graph.remove_node(merged_id)

        kg._graph.clear()
        for eid in kg.entities:
            kg._graph.add_node(eid)
        for rel in kg.relations:
            if rel.subject_id in kg.entities and rel.object_id in kg.entities:
                kg._graph.add_edge(
                    rel.subject_id, rel.object_id,
                    predicate=rel.predicate, weight=rel.confidence
                )

        stats_after = kg.stats()
        reduction = stats_before['nodes'] - stats_after['nodes']
        pct = reduction / max(stats_before['nodes'], 1) * 100
        print(f"  [Dedup] After:  {stats_after['nodes']} entities, "
              f"{stats_after['edges']} edges")
        print(f"  [Dedup] Removed: {reduction} entities ({pct:.1f}%)")

        return {
            "pairs_found": len(pairs),
            "entities_before": stats_before["nodes"],
            "entities_after": stats_after["nodes"],
            "edges_before": stats_before["edges"],
            "edges_after": stats_after["edges"],
            "reduction_pct": round(pct, 2),
        }
