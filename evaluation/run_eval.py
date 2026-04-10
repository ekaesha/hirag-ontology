import os
import sys
import json
import time
import csv
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from openai import OpenAI
from dotenv import load_dotenv
from pipeline.knowledge_graph import KnowledgeGraph, Entity, Relation
from pipeline.typing_agent import TypingAgent
from pipeline.deduplication import DeduplicationAgent
from pipeline.validator import ValidationAgent, ReasoningAgent
from pipeline.quality import compute_quality
from retrieval.retriever import HybridRetriever, RetrievalMode
from evaluation.judge import LLMJudge

load_dotenv()

QUESTIONS = [
    # Acute lymphoblastic leukemia
    "What is the treatment protocol for acute lymphoblastic leukemia in children?",
    "What induction therapy is used for ALL with BCR-ABL1 mutation?",
    "How is minimal residual disease monitored in acute lymphoblastic leukemia?",
    # Acute myeloid leukemia
    "What is the standard induction chemotherapy for acute myeloid leukemia?",
    "How is acute promyelocytic leukemia treated differently from other AML subtypes?",
    # Lymphomas
    "What is the first-line treatment for aggressive non-follicular lymphoma?",
    "How is Hodgkin lymphoma staged and treated?",
    "What treatment is recommended for follicular lymphoma stage III-IV?",
    # Solid tumors
    "What systemic therapy is used for adrenocortical carcinoma?",
    "How is anal squamous cell carcinoma treated with chemoradiation?",
    "What is the recommended treatment for basal cell skin cancer?",
    "How is biliary tract cancer managed with chemotherapy?",
    "What is the standard treatment protocol for bladder cancer?",
    "How are bone and cartilage sarcomas treated?",
    # Supportive care
    "How is febrile neutropenia managed during chemotherapy?",
    "What antiemetic regimens are used with highly emetogenic chemotherapy?",
    "How is tumor lysis syndrome prevented and treated?",
    "What supportive care is recommended during intensive chemotherapy?",
    # Cross-document
    "Which chemotherapy drugs require dose adjustment for renal impairment?",
    "What biomarkers are used to stratify risk in hematological malignancies?",
]


def create_demo_graph():
    kg = KnowledgeGraph()
    entities = [
        Entity("e001", "Paclitaxel", "Drug",
               "Chemotherapy drug for cancer treatment", ["Taxol", "PTX"]),
        Entity("e002", "Non-small cell lung cancer", "Condition",
               "Most common type of lung cancer", ["NSCLC"]),
        Entity("e003", "Bevacizumab", "Drug",
               "Anti-VEGF monoclonal antibody", ["Avastin"]),
        Entity("e004", "Hypertension", "Condition", "High blood pressure"),
        Entity("e005", "Carboplatin", "Drug",
               "Platinum-based chemotherapy agent"),
        Entity("e006", "Febrile neutropenia", "Condition",
               "Fever with low neutrophil count after chemotherapy"),
        Entity("e007", "G-CSF", "Drug",
               "Granulocyte colony-stimulating factor",
               ["filgrastim", "pegfilgrastim"]),
        Entity("e008", "PET-CT", "Procedure",
               "Positron emission tomography combined with CT"),
        Entity("e009", "EGFR mutation", "Condition",
               "Epidermal growth factor receptor mutation"),
        Entity("e010", "Erlotinib", "Drug",
               "EGFR tyrosine kinase inhibitor", ["Tarceva"]),
        Entity("e011", "Cisplatin", "Drug",
               "Platinum compound chemotherapy drug"),
        Entity("e012", "Breast cancer", "Condition",
               "Malignant tumor of breast tissue"),
        Entity("e013", "HER2-positive breast cancer", "Condition",
               "Breast cancer with HER2 overexpression"),
        Entity("e014", "Trastuzumab", "Drug",
               "Anti-HER2 monoclonal antibody", ["Herceptin"]),
        Entity("e015", "Ondansetron", "Drug",
               "5-HT3 antagonist antiemetic"),
        Entity("e016", "Nausea", "Symptom",
               "Feeling of sickness, urge to vomit"),
        Entity("e017", "Colorectal cancer", "Condition",
               "Cancer of colon or rectum"),
        Entity("e018", "FOLFOX", "DosageRegimen",
               "Regimen: 5-FU, leucovorin, oxaliplatin"),
        Entity("e019", "Bone marrow", "AnatomicalStructure",
               "Tissue inside bones that produces blood cells"),
        Entity("e020", "Stem cell transplantation", "Procedure",
               "Transplantation of hematopoietic stem cells"),
        Entity("e021", "Tamoxifen", "Drug",
               "Selective estrogen receptor modulator", ["SERM"]),
        Entity("e022", "Hormone receptor positive breast cancer",
               "Condition",
               "Breast cancer expressing estrogen receptors",
               ["HR+ breast cancer"]),
        Entity("e023", "Palliative care", "Procedure",
               "Supportive care focused on quality of life"),
        Entity("e024", "TNM staging", "Procedure",
               "Tumor Node Metastasis cancer staging system"),
        Entity("e025", "Creatinine clearance", "LabTest",
               "Measure of kidney function for dose adjustment"),
    ]
    for e in entities:
        kg.add_entity(e)

    relations = [
        Relation("e001", "treats", "e002"),
        Relation("e005", "treats", "e002"),
        Relation("e011", "treats", "e002"),
        Relation("e003", "contraindicated_for", "e004"),
        Relation("e007", "treats", "e006"),
        Relation("e010", "treats", "e009"),
        Relation("e010", "treats", "e002"),
        Relation("e014", "treats", "e013"),
        Relation("e015", "treats", "e016"),
        Relation("e001", "causes", "e016"),
        Relation("e011", "causes", "e006"),
        Relation("e005", "causes", "e006"),
        Relation("e008", "diagnosed_by", "e002"),
        Relation("e018", "treats", "e017"),
        Relation("e011", "treats", "e017"),
        Relation("e020", "part_of", "e019"),
        Relation("e013", "related_to", "e012"),
        Relation("e009", "related_to", "e002"),
        Relation("e021", "treats", "e022"),
        Relation("e023", "related_to", "e002"),
        Relation("e024", "diagnosed_by", "e002"),
        Relation("e025", "related_to", "e005"),
        Relation("e025", "related_to", "e011"),
    ]
    for r in relations:
        kg.add_relation(r)

    print(f"[Demo] Graph: {kg.stats()}")
    return kg


def build_graph_from_dataset(docs_folder, n_docs=10):
    from pipeline.extractor import ExtractionAgent, build_graph_from_text
    md_files = sorted(Path(docs_folder).glob("*.md"))
    print(f"  Found {len(md_files)} documents, using first {n_docs}")
    agent = ExtractionAgent(model="deepseek-chat")
    kg = KnowledgeGraph()
    selected = md_files[:n_docs]
    for i, doc_path in enumerate(selected):
        print(f"\n  [{i+1}/{len(selected)}] {doc_path.name}")
        with open(doc_path, encoding="utf-8") as f:
            text = f.read()
        doc_kg = build_graph_from_text(text, agent)
        for entity in doc_kg.entities.values():
            kg.add_entity(entity)
        for relation in doc_kg.relations:
            kg.add_relation(relation)
        print(f"  Total: {kg.stats()}")
    return kg


def generate_answer(question, kg, retriever, client, top_k=10):
    # Получаем сущности из графа
    entity_ids = retriever.retrieve(question, top_k=top_k)
    graph_context = kg.get_context_for_entities(entity_ids)

    # Если граф нашёл что-то полезное — используем его
    # Если нет — честно говорим что информации нет
    if not graph_context.strip() or graph_context == "No relevant entities found.":
        graph_context = "No relevant entities found in knowledge graph."

    prompt = f"""You are a medical information system with access to a structured knowledge graph.

Knowledge graph context (entities and relations extracted from clinical guidelines):
{graph_context}

Based on the above knowledge graph context, answer the following medical question.
Be specific, comprehensive, and clinically accurate.
If the knowledge graph does not contain sufficient information, say what you know from the context provided.

Question: {question}

Answer:"""

    try:
        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=600,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: {e}"


def run_evaluation(kg, mode, questions, judge, client, label):
    print(f"\n{'='*55}")
    print(f"  {label}")
    print(f"{'='*55}")
    retriever = HybridRetriever(kg, mode=mode)
    results = []
    for i, question in enumerate(questions):
        print(f"  Q{i+1}/{len(questions)}: {question[:55]}...")
        answer = generate_answer(question, kg, retriever, client)
        scores = judge.evaluate(question, answer)
        results.append({
            "question_id": i + 1,
            "question": question,
            "label": label,
            "mode": mode.value,
            "answer": answer,
            "comprehensiveness": scores.get("comprehensiveness", 0),
            "empowerment": scores.get("empowerment", 0),
            "diversity": scores.get("diversity", 0),
            "overall": scores.get("overall", 0),
            "comment": scores.get("comment", ""),
        })
        time.sleep(0.3)

    avg = {m: sum(r[m] for r in results) / len(results)
           for m in ["comprehensiveness", "empowerment",
                     "diversity", "overall"]}
    print(f"\n  Comprehensiveness: {avg['comprehensiveness']:.2f}")
    print(f"  Empowerment:       {avg['empowerment']:.2f}")
    print(f"  Diversity:         {avg['diversity']:.2f}")
    print(f"  Overall:           {avg['overall']:.2f}")
    return results


def save_csv(rows, path):
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")


def print_summary(all_results, title):
    modes = {}
    for r in all_results:
        lbl = r["label"]
        if lbl not in modes:
            modes[lbl] = {"comprehensiveness": [],
                          "empowerment": [],
                          "diversity": [],
                          "overall": []}
        for m in ["comprehensiveness", "empowerment",
                  "diversity", "overall"]:
            modes[lbl][m].append(r[m])

    print(f"\n{'='*65}")
    print(f"  {title}")
    print(f"{'='*65}")
    print(f"{'System':<32} {'Comp':>5} {'Emp':>5} "
          f"{'Div':>5} {'Overall':>8}")
    print("-" * 65)

    rows = []
    for label, scores in modes.items():
        avg = {m: sum(scores[m]) / len(scores[m]) for m in scores}
        print(f"{label:<32} {avg['comprehensiveness']:>5.2f} "
              f"{avg['empowerment']:>5.2f} "
              f"{avg['diversity']:>5.2f} "
              f"{avg['overall']:>8.2f}")
        rows.append({
            "System": label,
            "Comprehensiveness": f"{avg['comprehensiveness']:.2f}",
            "Empowerment": f"{avg['empowerment']:.2f}",
            "Diversity": f"{avg['diversity']:.2f}",
            "Overall": f"{avg['overall']:.2f}",
        })
    print(f"{'='*65}")
    return rows


def save_summary_csv(rows, path):
    fields = ["System", "Comprehensiveness",
              "Empowerment", "Diversity", "Overall"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[Saved] {path}")


def main():
    print("\nHiRAG-Ontology — Full Multi-Agent Pipeline")
    print("=" * 50)
    print("A1→A2→A3→A4→A5→A6→Evaluation\n")

    if not os.getenv("DEEPSEEK_API_KEY"):
        print("ERROR: DEEPSEEK_API_KEY not found in .env")
        return

    client = OpenAI(
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
    )
    os.makedirs("results", exist_ok=True)

    # ── A1: Extraction ────────────────────────────────────────────
    print("[A1] Extraction Agent")
    graph_path = "results/knowledge_graph_raw.json"

    if os.path.exists(graph_path):
        print("  Loading existing raw graph...")
        kg_raw = KnowledgeGraph.load(graph_path)
    else:
        docs_folder = Path("data/documents/minzdrav_dataset")
        if docs_folder.exists():
            print("  Building graph from Minzdrav dataset...")
            kg_raw = build_graph_from_dataset(docs_folder, n_docs=10)
        else:
            print("  No dataset found, using demo graph...")
            kg_raw = create_demo_graph()
        kg_raw.save(graph_path)

    stats_a1 = kg_raw.stats()
    print(f"  Graph after A1: {stats_a1}")

    # ── A2: Typing ────────────────────────────────────────────────
    print("\n[A2] Typing Agent — τ: V → C")
    typing_agent = TypingAgent(model="deepseek-chat")
    typing_stats = typing_agent.type_graph(kg_raw)
    kg_raw.save("results/knowledge_graph_typed.json")

    # ── A3: Deduplication ─────────────────────────────────────────
    print("\n[A3] Deduplication Agent — π: V → V_canon")
    kg_dedup = KnowledgeGraph.load("results/knowledge_graph_typed.json")
    dedup_agent = DeduplicationAgent(alpha=0.6, threshold=0.85)
    dedup_stats = dedup_agent.deduplicate(kg_dedup)
    kg_dedup.save("results/knowledge_graph_dedup.json")

    # ── A4: Validation ────────────────────────────────────────────
    print("\n[A4] Validation Agent — Cons(G) = 1 - violations/|A|")
    validator = ValidationAgent()

    val_before = validator.validate(kg_raw)
    print(f"  Cons(G) before: {val_before['consistency_score']:.3f}")

    val_after = validator.validate(kg_dedup)
    print(f"  Cons(G) after dedup: {val_after['consistency_score']:.3f}")

    repaired = validator.auto_repair(kg_dedup, val_after)
    if repaired > 0:
        val_final = validator.validate(kg_dedup)
        print(f"  Cons(G) after repair: {val_final['consistency_score']:.3f}")
    else:
        val_final = val_after

    kg_dedup.save("results/knowledge_graph_validated.json")

    # ── A5: Reasoning ─────────────────────────────────────────────
    print("\n[A5] Reasoning Agent — T_missing = T* - T_hat")
    reasoner = ReasoningAgent()
    suggestions = reasoner.find_missing_relations(kg_dedup)
    added = reasoner.apply_suggestions(
        kg_dedup, suggestions, max_apply=10
    )

    # ── A6: Graph Update ──────────────────────────────────────────
    print("\n[A6] Graph Update — PageRank + save final graph")
    kg_dedup.compute_pagerank()
    kg_raw.compute_pagerank()
    kg_dedup.save("results/knowledge_graph_final.json")

    # ── Q(G) ──────────────────────────────────────────────────────
    print("\n[Q(G)] Computing quality functional...")
    q_before = compute_quality(kg_raw, val_before)
    q_after = compute_quality(kg_dedup, val_final)

    quality_report = {
        "before_pipeline": q_before,
        "after_pipeline": q_after,
        "improvement": round(q_after["Q"] - q_before["Q"], 4),
    }
    with open("results/quality_report.json", "w") as f:
        json.dump(quality_report, f, indent=2)
    print(f"  Q(G) before: {q_before['Q']:.3f}")
    print(f"  Q(G) after:  {q_after['Q']:.3f}")
    print(f"  Delta:       +{quality_report['improvement']:.3f}")
    print("[Saved] results/quality_report.json")

    # ── Pipeline stats ────────────────────────────────────────────
    pipeline_stats = {
        "A1_extraction": stats_a1,
        "A2_typing": typing_stats,
        "A3_deduplication": {
            k: v for k, v in dedup_stats.items()
            if k != "canonical_map"
        },
        "A4_validation": {
            "cons_before": val_before["consistency_score"],
            "cons_after": val_final["consistency_score"],
            "violations_repaired": repaired,
        },
        "A5_reasoning": {
            "suggestions_found": len(suggestions),
            "relations_added": added,
        },
        "A6_update": {
            "final_nodes": kg_dedup.stats()["nodes"],
            "final_edges": kg_dedup.stats()["edges"],
        },
    }
    with open("results/pipeline_stats.json", "w") as f:
        json.dump(pipeline_stats, f, indent=2)
    print("[Saved] results/pipeline_stats.json")

    # ── Evaluation — Table 5.1 ────────────────────────────────────
    print("\n[Eval] RAG Evaluation — Table 5.1")
    judge = LLMJudge(model="deepseek-chat")
    all_results = []

    configs = [
        (kg_raw,   RetrievalMode.SEMANTIC_ONLY, "Naive RAG"),
        (kg_raw,   RetrievalMode.HYBRID_RRF,    "HiRAG (baseline)"),
        (kg_dedup, RetrievalMode.SEMANTIC_ONLY, "HiRAG + Dedup"),
        (kg_dedup, RetrievalMode.HYBRID_RRF,    "HiRAG-Ontology (Ours)"),
    ]

    for kg, mode, label in configs:
        results = run_evaluation(
            kg, mode, QUESTIONS, judge, client, label
        )
        all_results.extend(results)
        time.sleep(1)

    save_csv(all_results, "results/evaluation_results.csv")
    summary = print_summary(all_results, "TABLE 5.1 — Main Results")
    save_summary_csv(summary, "results/summary_table51.csv")

    # ── Ablation — Table 5.2 ──────────────────────────────────────
    print("\n[Eval] Ablation Study — Table 5.2")
    ablation_results = []

    ablation_configs = [
        (kg_raw,   RetrievalMode.SEMANTIC_ONLY, "Baseline"),
        (kg_dedup, RetrievalMode.SEMANTIC_ONLY, "w/o Hybrid Retriever"),
        (kg_raw,   RetrievalMode.HYBRID_RRF,    "w/o Deduplication"),
        (kg_dedup, RetrievalMode.HYBRID_RRF,    "Full System (Ours)"),
    ]

    for kg, mode, label in ablation_configs:
        results = run_evaluation(
            kg, mode, QUESTIONS[:10], judge, client, label
        )
        ablation_results.extend(results)
        time.sleep(1)

    save_csv(ablation_results, "results/ablation_results.csv")
    ablation_summary = print_summary(
        ablation_results, "TABLE 5.2 — Ablation Study"
    )
    save_summary_csv(ablation_summary, "results/ablation_table52.csv")

    # ── Done ──────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("PIPELINE COMPLETE")
    print("=" * 55)
    print("results/summary_table51.csv  → Table 5.1")
    print("results/ablation_table52.csv → Table 5.2")
    print("results/quality_report.json  → Q(G) values")
    print("results/pipeline_stats.json  → agent stats")
    print("=" * 55)


if __name__ == "__main__":
    main()