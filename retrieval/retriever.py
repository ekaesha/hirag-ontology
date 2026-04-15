import re
import sys
from enum import Enum
from pipeline.knowledge_graph import KnowledgeGraph

IN_COLAB = 'google.colab' in sys.modules

_GLOBAL_ENCODER = None


def _get_global_encoder():
    global _GLOBAL_ENCODER
    if _GLOBAL_ENCODER is None:
        import os
        from sentence_transformers import SentenceTransformer

        if not IN_COLAB:
            #
            #
            print("  [Encoder] Loading BERT model from cache...")
        else:
            #
            #
            print("  [Encoder] Downloading BERT model...")

        _GLOBAL_ENCODER = SentenceTransformer(
            "paraphrase-multilingual-MiniLM-L12-v2",
            local_files_only=False,
        )
        print("  [Encoder] Model ready.")
    return _GLOBAL_ENCODER


class RetrievalMode(Enum):
    SEMANTIC_ONLY   = "semantic_only"
    LEXICAL_ONLY    = "lexical_only"
    STRUCTURAL_ONLY = "structural_only"
    HYBRID_RRF      = "hybrid_rrf"


def _cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x**2 for x in a)**0.5
    nb = sum(x**2 for x in b)**0.5
    return dot / (na * nb) if na and nb else 0.0


def _tokenize(text):
    return re.sub(r"[^\w\s]", " ", text.lower()).split()


def build_bm25_index(kg):
    from rank_bm25 import BM25Okapi
    entity_ids = list(kg.entities.keys())
    if not entity_ids:
        return None, []
    corpus = []
    for eid in entity_ids:
        e = kg.entities[eid]
        doc = e.label + " " + e.description + " " + " ".join(e.aliases)
        corpus.append(_tokenize(doc))
    bm25 = BM25Okapi(corpus)
    return bm25, entity_ids


def rrf_fuse(ranked_lists, k=60):
    scores = {}
    for ranked in ranked_lists:
        for rank, (eid, _) in enumerate(ranked, start=1):
            scores[eid] = scores.get(eid, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)


class HybridRetriever:
    def __init__(self, kg, mode=RetrievalMode.HYBRID_RRF):
        self.kg = kg
        self.mode = mode
        self._encoder = None
        self._bm25 = None
        self._bm25_ids = None

    def _load_encoder(self):
        if self._encoder is None:
            self._encoder = _get_global_encoder()
        return self._encoder

    def _ensure_embeddings(self):
        encoder = self._load_encoder()
        missing = [e for e in self.kg.entities.values()
                   if e.embedding is None]
        if missing:
            labels = [e.label for e in missing]
            embs = encoder.encode(
                labels, batch_size=64, show_progress_bar=False
            )
            for entity, emb in zip(missing, embs):
                entity.embedding = emb.tolist()

    def _ensure_bm25(self):
        if self._bm25 is None:
            self._bm25, self._bm25_ids = build_bm25_index(self.kg)

    def _ensure_pagerank(self):
        if not self.kg._pagerank:
            self.kg.compute_pagerank()

    def _semantic_ranked(self, query, top_k):
        encoder = self._load_encoder()
        q_emb = encoder.encode(
            [query], show_progress_bar=False
        )[0].tolist()
        scores = []
        for eid, entity in self.kg.entities.items():
            if entity.embedding is None:
                continue
            scores.append((eid, _cosine(q_emb, entity.embedding)))
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    def _lexical_ranked(self, query, top_k):
        if self._bm25 is None or not self._bm25_ids:
            return []
        tokens = _tokenize(query)
        raw = self._bm25.get_scores(tokens)
        ranked = sorted(
            zip(self._bm25_ids, raw),
            key=lambda x: x[1], reverse=True
        )
        return [(eid, float(s)) for eid, s in ranked[:top_k]]

    def _structural_ranked(self, top_k):
        scored = [
            (eid, self.kg.get_pagerank(eid))
            for eid in self.kg.entities
        ]
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored[:top_k]

    def retrieve(self, query, top_k=10):
        if self.mode == RetrievalMode.SEMANTIC_ONLY:
            self._ensure_embeddings()
            ranked = self._semantic_ranked(query, top_k * 5)
            return [eid for eid, _ in ranked[:top_k]]

        elif self.mode == RetrievalMode.LEXICAL_ONLY:
            self._ensure_bm25()
            ranked = self._lexical_ranked(query, top_k * 5)
            return [eid for eid, _ in ranked[:top_k]]

        elif self.mode == RetrievalMode.STRUCTURAL_ONLY:
            self._ensure_pagerank()
            ranked = self._structural_ranked(top_k * 5)
            return [eid for eid, _ in ranked[:top_k]]

        elif self.mode == RetrievalMode.HYBRID_RRF:
            self._ensure_embeddings()
            self._ensure_bm25()
            self._ensure_pagerank()
            sem = self._semantic_ranked(query, 200)
            lex = self._lexical_ranked(query, 200)
            struct = self._structural_ranked(200)
            fused = rrf_fuse([sem, lex, struct], k=60)
            return [eid for eid, _ in fused[:top_k]]

        return []
