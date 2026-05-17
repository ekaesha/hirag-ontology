"""
evaluation/generation_eval.py
==============================
Computes RAGAS-aligned generation quality metrics.

AUTHOR: Eva Karimova (E. Karimova, Individual Contribution)

WHAT THIS FILE DOES:
    For each benchmark question, generates an answer using the full RAG pipeline
    and evaluates it on 4 dimensions using DeepSeek-Chat as the judge:

      1. Faithfulness      — are all answer claims grounded in the retrieved context?
                             Score: 0–1 (fraction of claims supported)
      2. Answer Relevance  — is the answer actually addressing the question?
                             Score: 0–1 (embedding cosine similarity)
      3. Context Precision — what fraction of retrieved entities are actually relevant?
                             Score: 0–1
      4. Context Recall    — what fraction of ground-truth entities were retrieved?
                             Score: 0–1

    These metrics correspond to the RAGAS framework (Es et al., 2023).

HOW TO RUN:
    python -m evaluation.generation_eval
    # or
    python evaluation/generation_eval.py --n 50

    Requires: DEEPSEEK_API_KEY in .env

OUTPUT:
    - Console table
    - results/generation_metrics.json
    - results/generation_metrics_per_question.json
"""

import json
import os
import sys
import time
import re
from pathlib import Path

import openai
import numpy as np
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv()

from retrieval.retriever import HybridRetriever, RetrievalMode
from pipeline.knowledge_graph import KnowledgeGraph


# ── constants ─────────────────────────────────────────────────────────────────

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"  # same as pipeline

FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to the provided context.

CONTEXT (retrieved entities):
{context}

QUESTION: {question}

ANSWER: {answer}

Task: List all factual claims made in the ANSWER. For each claim, state whether it is
SUPPORTED or NOT SUPPORTED by the CONTEXT. Then output a JSON object with this exact format:
{{
  "claims": [
    {{"claim": "...", "supported": true}},
    {{"claim": "...", "supported": false}}
  ],
  "faithfulness_score": <fraction of supported claims, 0.0 to 1.0>
}}

Output ONLY the JSON object, no other text."""


CONTEXT_PRECISION_PROMPT = """\
You are evaluating the precision of retrieved context for a RAG system.

QUESTION: {question}

RETRIEVED ENTITIES (context):
{context}

GROUND TRUTH relevant entity labels: {gt_labels}

Task: For each retrieved entity, state whether it is RELEVANT to answering the question.
Output a JSON object:
{{
  "entity_relevance": [
    {{"entity": "...", "relevant": true}},
    {{"entity": "...", "relevant": false}}
  ],
  "context_precision": <fraction of relevant entities, 0.0 to 1.0>
}}

Output ONLY the JSON object."""


GENERATION_PROMPT = """\
You are a clinical decision support assistant specialising in oncology.
Use the following retrieved entities and their relations to answer the question.
Answer in English.

Retrieved context:
{context}

Question: {question}

Provide a comprehensive answer based on the retrieved context."""


def cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 == 0 or norm2 == 0:
        return 0.0
    return float(np.dot(v1, v2) / (norm1 * norm2))


def format_context(retrieved_entities, max_entities: int = 10) -> str:
    lines = []
    for ent in retrieved_entities[:max_entities]:
        desc = getattr(ent, "description", "") or ""
        line = f"[{ent.entity_type}] {ent.label}: {desc[:150]}"
        lines.append(line)
    return "\n".join(lines)


def safe_json_parse(text: str) -> dict:
    """
    Tries to parse JSON from LLM output.
    LLMs sometimes wrap JSON in ```json ... ``` code blocks, so we strip those first.
    """
    # Remove markdown code fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        # Try to find JSON object in the text
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group())
            except json.JSONDecodeError:
                pass
    return {}


class GenerationEvaluator:
    """
    Evaluates RAG generation quality using LLM-as-judge and embedding similarity.

    Usage:
        evaluator = GenerationEvaluator(kg, retriever, llm_client)
        result = evaluator.evaluate_question(question, gt_labels)
    """

    def __init__(self, kg: KnowledgeGraph, retriever: HybridRetriever, llm_client, top_k: int = 10):
        self.kg = kg
        self.retriever = retriever
        self.llm = llm_client
        self.top_k = top_k

        print("[generation_eval] Loading sentence transformer for Answer Relevance ...")
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)

    def _llm_call(self, prompt: str, max_tokens: int = 1024) -> str:
        """Makes one LLM call and returns the response text."""
        response = self.llm.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=0.0,  # deterministic for evaluation
        )
        return response.choices[0].message.content or ""

    def compute_faithfulness(self, question: str, context_str: str, answer: str) -> float:
        """
        Asks the LLM to verify each claim in the answer against the context.
        Returns the fraction of claims that are grounded in the context.
        """
        prompt = FAITHFULNESS_PROMPT.format(
            context=context_str, question=question, answer=answer
        )
        raw = self._llm_call(prompt)
        parsed = safe_json_parse(raw)

        score = parsed.get("faithfulness_score")
        if score is not None:
            return float(score)

        # Fallback: compute manually from claims list
        claims = parsed.get("claims", [])
        if not claims:
            return 0.5  # unknown → neutral
        supported = sum(1 for c in claims if c.get("supported", False))
        return supported / len(claims)

    def compute_answer_relevance(self, question: str, answer: str) -> float:
        """
        Answer Relevance = cosine similarity between question embedding and answer embedding.
        High similarity = the answer is talking about the same topic as the question.

        This is a proxy metric — a proper implementation would generate multiple
        synthetic questions from the answer and compute similarity, but cosine
        similarity of embeddings is a good approximation.
        """
        q_emb = self.embedder.encode(question, convert_to_numpy=True)
        a_emb = self.embedder.encode(answer, convert_to_numpy=True)
        return cosine_similarity(q_emb, a_emb)

    def compute_context_precision(self, question: str, context_str: str, retrieved_entities) -> float:
        """
        Context Precision = fraction of retrieved entities that are relevant to the question.
        Uses LLM-as-judge.
        """
        entity_list = "\n".join(f"- {ent.label}" for ent in retrieved_entities[:self.top_k])
        prompt = CONTEXT_PRECISION_PROMPT.format(
            question=question, context=context_str, gt_labels=entity_list
        )
        raw = self._llm_call(prompt)
        parsed = safe_json_parse(raw)

        score = parsed.get("context_precision")
        if score is not None:
            return float(score)

        # Fallback from entity_relevance list
        relevance = parsed.get("entity_relevance", [])
        if not relevance:
            return 0.5
        relevant = sum(1 for r in relevance if r.get("relevant", False))
        return relevant / len(relevance)

    def compute_context_recall(self, retrieved_entities, ground_truth_labels: list[str]) -> float:
        """
        Context Recall = fraction of ground-truth entities that were retrieved.
        This is computed exactly (no LLM needed), by checking entity label matching.

        Formula: |retrieved ∩ ground_truth| / |ground_truth|
        """
        if not ground_truth_labels:
            return 1.0

        retrieved_set = set()
        for ent in retrieved_entities:
            retrieved_set.add(ent.label.strip().lower())
            for alias in getattr(ent, "aliases", []):
                retrieved_set.add(alias.strip().lower())

        gt_normalised = {l.strip().lower() for l in ground_truth_labels}
        found = gt_normalised & retrieved_set
        return len(found) / len(gt_normalised)

    def evaluate_question(self, question: str, gt_labels: list[str]) -> dict:
        """
        Full evaluation pipeline for one question.
        Returns a dict with all 4 metric scores.
        """
        # Step 1: Retrieve
        retrieved = self.retriever.retrieve(question, k=self.top_k, mode=RetrievalMode.HYBRID_RRF)
        context_str = format_context(retrieved, max_entities=self.top_k)

        # Step 2: Generate answer
        gen_prompt = GENERATION_PROMPT.format(context=context_str, question=question)
        answer = self._llm_call(gen_prompt, max_tokens=512)

        # Step 3: Compute metrics
        faithfulness     = self.compute_faithfulness(question, context_str, answer)
        answer_relevance = self.compute_answer_relevance(question, answer)
        context_precision = self.compute_context_precision(question, context_str, retrieved)
        context_recall   = self.compute_context_recall(retrieved, gt_labels)

        return {
            "faithfulness":       round(faithfulness, 4),
            "answer_relevance":   round(answer_relevance, 4),
            "context_precision":  round(context_precision, 4),
            "context_recall":     round(context_recall, 4),
            "answer_preview":     answer[:200],
        }


# ── main runner ───────────────────────────────────────────────────────────────

def run_generation_eval(
    kg_path: str = "results/knowledge_graph_final.json",
    gt_path: str = "evaluation/ground_truth.json",
    n_questions: int = 50,
) -> dict:
    """Runs generation evaluation on the first n_questions from the benchmark."""

    # Load KG
    print(f"[generation_eval] Loading KG ...")
    kg = KnowledgeGraph()
    kg.load(kg_path)

    # Load ground truth
    with open(gt_path, "r", encoding="utf-8") as f:
        gt_data = json.load(f)
    questions = gt_data["questions"][:n_questions]

    # LLM client
    api_key = os.getenv("DEEPSEEK_API_KEY") or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("Set DEEPSEEK_API_KEY in your .env file.")
    llm = openai.OpenAI(api_key=api_key, base_url="https://api.deepseek.com/v1")

    # Build evaluator
    retriever = HybridRetriever(kg)
    evaluator = GenerationEvaluator(kg, retriever, llm)

    # Evaluate
    per_question = []
    agg = {"faithfulness": [], "answer_relevance": [], "context_precision": [], "context_recall": []}

    for i, q in enumerate(questions):
        print(f"  [{i+1}/{len(questions)}] {q['question'][:70]}...")
        result = evaluator.evaluate_question(q["question"], q["relevant_entity_labels"])
        per_question.append({"id": q["id"], "question": q["question"], **result})

        for metric in agg:
            agg[metric].append(result[metric])

    # Aggregate
    summary = {k: round(sum(v) / len(v), 4) for k, v in agg.items() if v}

    return {"summary": summary, "per_question": per_question, "n": len(questions)}


def print_generation_table(summary: dict) -> None:
    print("\n" + "=" * 55)
    print("Generation Quality Metrics (HiRAG-Ontology, Hybrid RRF)")
    print("=" * 55)
    labels = {
        "faithfulness":      "Faithfulness",
        "answer_relevance":  "Answer Relevance",
        "context_precision": "Context Precision",
        "context_recall":    "Context Recall",
    }
    for key, label in labels.items():
        print(f"  {label:<25} {summary.get(key, 0.0):.4f}")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="RAGAS-style generation evaluation.")
    parser.add_argument("--n",  type=int, default=50, help="Number of questions (default: 50)")
    parser.add_argument("--kg", default="results/knowledge_graph_final.json")
    parser.add_argument("--gt", default="evaluation/ground_truth.json")
    args = parser.parse_args()

    results = run_generation_eval(kg_path=args.kg, gt_path=args.gt, n_questions=args.n)
    print_generation_table(results["summary"])

    out_dir = Path("results")
    out_dir.mkdir(exist_ok=True)

    with open(out_dir / "generation_metrics.json", "w", encoding="utf-8") as f:
        json.dump(results["summary"], f, ensure_ascii=False, indent=2)

    with open(out_dir / "generation_metrics_per_question.json", "w", encoding="utf-8") as f:
        json.dump(results["per_question"], f, ensure_ascii=False, indent=2)

    print(f"[generation_eval] Saved to results/generation_metrics*.json")
