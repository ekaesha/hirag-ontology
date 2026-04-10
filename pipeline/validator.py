"""
Validation Agent (A4) + Reasoning Agent (A5).

A4 — проверяет логическую согласованность графа:
     Cons(G) = 1 - |violations| / |A|

A5 — находит пропущенные связи:
     T_missing = T* - T_hat
"""

from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.typing_agent import ONTOLOGY_PROPERTIES, ONTOLOGY_CLASSES


# Аксиомы онтологии A — правила которым должен удовлетворять граф
AXIOMS = [
    # (axiom_id, description, check_function)
    # Аксиома 1: домен предиката должен совпадать с типом субъекта
    "domain_constraint",
    # Аксиома 2: ренж предиката должен совпадать с типом объекта
    "range_constraint",
    # Аксиома 3: сущность должна иметь известный тип
    "valid_type",
    # Аксиома 4: предикат должен быть из онтологии
    "valid_predicate",
    # Аксиома 5: нет петель (сущность не связана сама с собой)
    "no_self_loops",
]

VALID_PREDICATES = set(ONTOLOGY_PROPERTIES.keys())
VALID_TYPES = set(ONTOLOGY_CLASSES.keys())


class ValidationAgent:
    """
    A4 — Validation Agent.
    Проверяет выполнение аксиом онтологии.
    Реализует Cons(G) = 1 - |violations(G)| / |A|
    """

    def check_domain_constraint(self, kg: KnowledgeGraph) -> list:
        """Домен предиката должен совпадать с типом субъекта."""
        violations = []
        for rel in kg.relations:
            if rel.predicate not in ONTOLOGY_PROPERTIES:
                continue
            expected_domain = ONTOLOGY_PROPERTIES[rel.predicate]["domain"]
            if expected_domain == "Other":
                continue
            subj = kg.entities.get(rel.subject_id)
            if subj and subj.entity_type != expected_domain:
                violations.append({
                    "axiom": "domain_constraint",
                    "predicate": rel.predicate,
                    "entity": subj.label,
                    "expected_type": expected_domain,
                    "actual_type": subj.entity_type,
                    "description": (
                        f"'{subj.label}' has type '{subj.entity_type}' "
                        f"but predicate '{rel.predicate}' "
                        f"requires domain '{expected_domain}'"
                    )
                })
        return violations

    def check_range_constraint(self, kg: KnowledgeGraph) -> list:
        """Ренж предиката должен совпадать с типом объекта."""
        violations = []
        for rel in kg.relations:
            if rel.predicate not in ONTOLOGY_PROPERTIES:
                continue
            expected_range = ONTOLOGY_PROPERTIES[rel.predicate]["range"]
            if expected_range == "Other":
                continue
            obj = kg.entities.get(rel.object_id)
            if obj and obj.entity_type != expected_range:
                violations.append({
                    "axiom": "range_constraint",
                    "predicate": rel.predicate,
                    "entity": obj.label,
                    "expected_type": expected_range,
                    "actual_type": obj.entity_type,
                    "description": (
                        f"'{obj.label}' has type '{obj.entity_type}' "
                        f"but predicate '{rel.predicate}' "
                        f"requires range '{expected_range}'"
                    )
                })
        return violations

    def check_valid_types(self, kg: KnowledgeGraph) -> list:
        """Все сущности должны иметь известный тип."""
        violations = []
        for entity in kg.entities.values():
            if entity.entity_type not in VALID_TYPES:
                violations.append({
                    "axiom": "valid_type",
                    "entity": entity.label,
                    "actual_type": entity.entity_type,
                    "description": (
                        f"Entity '{entity.label}' has unknown "
                        f"type '{entity.entity_type}'"
                    )
                })
        return violations

    def check_valid_predicates(self, kg: KnowledgeGraph) -> list:
        """Все предикаты должны быть из онтологии."""
        violations = []
        for rel in kg.relations:
            if rel.predicate not in VALID_PREDICATES:
                subj = kg.entities.get(rel.subject_id)
                obj = kg.entities.get(rel.object_id)
                violations.append({
                    "axiom": "valid_predicate",
                    "predicate": rel.predicate,
                    "subject": subj.label if subj else rel.subject_id,
                    "object": obj.label if obj else rel.object_id,
                    "description": (
                        f"Predicate '{rel.predicate}' is not "
                        f"defined in the ontology"
                    )
                })
        return violations

    def check_no_self_loops(self, kg: KnowledgeGraph) -> list:
        """Сущность не должна быть связана сама с собой."""
        violations = []
        for rel in kg.relations:
            if rel.subject_id == rel.object_id:
                entity = kg.entities.get(rel.subject_id)
                violations.append({
                    "axiom": "no_self_loops",
                    "entity": entity.label if entity else rel.subject_id,
                    "predicate": rel.predicate,
                    "description": (
                        f"Self-loop detected: '{entity.label if entity else rel.subject_id}' "
                        f"--[{rel.predicate}]--> itself"
                    )
                })
        return violations

    def validate(self, kg: KnowledgeGraph) -> dict:
        """
        Запустить все проверки и посчитать Cons(G).

        Возвращает:
        {
          'consistency_score': float,   — Cons(G) в [0,1]
          'total_axioms': int,          — |A|
          'total_violations': int,      — |violations(G)|
          'violations': list,           — детальный список нарушений
          'violations_by_type': dict    — сгруппированные нарушения
        }
        """
        print(f"  [Validation] Checking {len(kg.entities)} entities, "
              f"{len(kg.relations)} relations...")

        all_violations = []
        all_violations.extend(self.check_domain_constraint(kg))
        all_violations.extend(self.check_range_constraint(kg))
        all_violations.extend(self.check_valid_types(kg))
        all_violations.extend(self.check_valid_predicates(kg))
        all_violations.extend(self.check_no_self_loops(kg))

        # Считаем Cons(G) = 1 - |violations| / |A|
        # |A| = количество аксиом × количество триплетов
        total_checks = len(AXIOMS) * max(len(kg.relations), 1)
        num_violations = len(all_violations)
        consistency = 1.0 - (num_violations / total_checks)
        consistency = max(0.0, min(1.0, consistency))

        # Группируем по типу
        violations_by_type = {}
        for v in all_violations:
            axiom = v["axiom"]
            if axiom not in violations_by_type:
                violations_by_type[axiom] = []
            violations_by_type[axiom].append(v)

        print(f"  [Validation] Violations found: {num_violations}")
        print(f"  [Validation] Cons(G) = {consistency:.3f}")

        return {
            "consistency_score": round(consistency, 4),
            "total_axioms": len(AXIOMS),
            "total_checks": total_checks,
            "total_violations": num_violations,
            "violations": all_violations,
            "violations_by_type": {
                k: len(v) for k, v in violations_by_type.items()
            },
        }

    def auto_repair(self, kg: KnowledgeGraph,
                    validation_result: dict) -> int:
        """
        Автоматически исправить простые нарушения.
        Возвращает количество исправленных нарушений.
        """
        repaired = 0
        for v in validation_result["violations"]:
            # Исправляем неизвестные типы → Other
            if v["axiom"] == "valid_type":
                for entity in kg.entities.values():
                    if entity.label == v["entity"]:
                        entity.entity_type = "Other"
                        repaired += 1
                        break
            # Удаляем петли
            elif v["axiom"] == "no_self_loops":
                kg.relations = [
                    r for r in kg.relations
                    if r.subject_id != r.object_id
                ]
                repaired += 1

        if repaired > 0:
            print(f"  [Validation] Auto-repaired {repaired} violations")
        return repaired


class ReasoningAgent:
    """
    A5 — Reasoning Agent.
    Находит пропущенные связи: T_missing = T* - T_hat

    Логика: если Drug treats Condition и Drug causes Symptom,
    то вероятно Condition related_to Symptom.
    """

    # Правила вывода: (тип_субъекта, предикат1, тип_объекта1) →
    #                  предлагаем (тип_объекта1, новый_предикат, ?)
    INFERENCE_RULES = [
        # Если Drug treats Condition → ищем другие Drug для той же Condition
        {
            "if_subject_type": "Drug",
            "if_predicate": "treats",
            "if_object_type": "Condition",
            "suggest_predicate": "related_to",
            "suggest_between": ("Condition", "Condition"),
            "description": "Conditions treated by same drug may be related",
        },
        # Если Drug causes Symptom → Symptom related_to Drug's Condition
        {
            "if_subject_type": "Drug",
            "if_predicate": "causes",
            "if_object_type": "Symptom",
            "suggest_predicate": "related_to",
            "suggest_between": ("Symptom", "Condition"),
            "description": "Symptoms caused by drug may relate to its indications",
        },
    ]

    def find_missing_relations(self, kg: KnowledgeGraph) -> list:
        """
        Найти потенциально пропущенные связи T_missing.
        Возвращает список предлагаемых триплетов.
        """
        suggestions = []

        # Группируем связи по типу
        by_predicate: dict = {}
        for rel in kg.relations:
            if rel.predicate not in by_predicate:
                by_predicate[rel.predicate] = []
            by_predicate[rel.predicate].append(rel)

        # Простое правило: Drug treats Condition
        # Если два разных Drug treat одну Condition — они related_to
        treats_rels = by_predicate.get("treats", [])
        condition_to_drugs: dict = {}
        for rel in treats_rels:
            obj = kg.entities.get(rel.object_id)
            subj = kg.entities.get(rel.subject_id)
            if obj and subj and obj.entity_type == "Condition":
                if obj.id not in condition_to_drugs:
                    condition_to_drugs[obj.id] = []
                condition_to_drugs[obj.id].append(subj.id)

        # Предлагаем related_to между препаратами одной нозологии
        existing_pairs = {
            (r.subject_id, r.object_id) for r in kg.relations
        }
        for condition_id, drug_ids in condition_to_drugs.items():
            if len(drug_ids) > 1:
                condition = kg.entities.get(condition_id)
                for i in range(len(drug_ids)):
                    for j in range(i + 1, len(drug_ids)):
                        d1, d2 = drug_ids[i], drug_ids[j]
                        if (d1, d2) not in existing_pairs:
                            drug1 = kg.entities.get(d1)
                            drug2 = kg.entities.get(d2)
                            if drug1 and drug2:
                                suggestions.append({
                                    "subject": drug1.label,
                                    "predicate": "related_to",
                                    "object": drug2.label,
                                    "reason": (
                                        f"Both treat "
                                        f"'{condition.label if condition else condition_id}'"
                                    ),
                                    "subject_id": d1,
                                    "object_id": d2,
                                })

        print(f"  [Reasoning] Found {len(suggestions)} missing relations")
        return suggestions

    def apply_suggestions(
        self, kg: KnowledgeGraph, suggestions: list,
        max_apply: int = 10
    ) -> int:
        """
        Добавить предложенные связи в граф.
        Возвращает количество добавленных связей.
        """
        from pipeline.knowledge_graph import Relation
        added = 0
        existing = {
            (r.subject_id, r.predicate, r.object_id)
            for r in kg.relations
        }

        for s in suggestions[:max_apply]:
            key = (s["subject_id"], s["predicate"], s["object_id"])
            if key not in existing:
                kg.add_relation(Relation(
                    subject_id=s["subject_id"],
                    predicate=s["predicate"],
                    object_id=s["object_id"],
                    confidence=0.7,
                    source_chunk="reasoning_agent",
                ))
                existing.add(key)
                added += 1

        if added > 0:
            print(f"  [Reasoning] Added {added} inferred relations")
        return added