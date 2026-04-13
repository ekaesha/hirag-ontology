import os
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))

from openai import OpenAI
from dotenv import load_dotenv
from pipeline.knowledge_graph import KnowledgeGraph
from retrieval.retriever import HybridRetriever, RetrievalMode

load_dotenv()

client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com",
)

kg_raw   = KnowledgeGraph.load("results/knowledge_graph_raw.json")
kg_final = KnowledgeGraph.load("results/knowledge_graph_final.json")
kg_raw.compute_pagerank()
kg_final.compute_pagerank()

QUESTION = "What is the treatment protocol for acute lymphoblastic leukemia in children, and how is Ph+ ALL managed differently?"

def answer(question, kg, mode, label):
    retriever = HybridRetriever(kg, mode=mode)
    entity_ids = retriever.retrieve(question, top_k=10)
    context = kg.get_context_for_entities(entity_ids)

    prompt = f"""You are a medical information system.
Answer based ONLY on the knowledge graph context below.
Be specific and comprehensive.

Context:
{context}

Question: {question}
Answer:"""

    resp = client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.1,
        max_tokens=500,
    )
    text = resp.choices[0].message.content.strip()
    print(f"\n{'='*60}")
    print(f"SYSTEM: {label}")
    print(f"{'='*60}")
    print(f"CONTEXT ENTITIES: {len(entity_ids)}")
    print(f"ANSWER:\n{text}")
    return text

print(f"QUESTION: {QUESTION}\n")
answer(QUESTION, kg_raw,   RetrievalMode.SEMANTIC_ONLY, "Naive RAG (semantic only)")
answer(QUESTION, kg_final, RetrievalMode.HYBRID_RRF,   "HiRAG-Ontology (our system)")