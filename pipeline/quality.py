"""
Quality Functional — Q(G) из формальной модели диплома.

Q(G) = λ1·Coverage + λ2·Consistency + λ3·Precision - λ4·Redundancy

Это целевая функция оптимизации всей системы:
G* = argmax Q(G)
"""

from pipeline.knowledge_graph import KnowledgeGraph
from pipeline.typing_agent import ONTOLOGY_CLASSES, VALID_TYPES


def coverage(kg: KnowledgeGraph) -> float:
    """
    Coverage(G) — доля классов онтологии представленных в графе.

    Coverage = |{C in C_domain : exists v, τ(v) = C}| / |C_domain|
    """
    if not kg.entities:
        return 0.0
    represented = {e.entity_type for e in kg.entities.values()}
    represented = represented & VALID_TYPES
    return len(represented) / len(ONTOLOGY_CLASSES)


def consistency(validation_result: dict) -> float:
    """
    Consistency(G) — из результата ValidationAgent.
    Cons(G) = 1 - |violations| / |A|
    """
    return validation_result.get("consistency_score", 0.0)


def precision(kg: KnowledgeGraph) -> float:
    """
    Precision(G) — доля триплетов с высокой уверенностью.

    Аппроксимируем через confidence scores извлечённых триплетов.
    В реальной системе требует аннотированный gold standard.
    """
    if not kg.relations:
        return 0.0
    high_conf = sum(
        1 for r in kg.relations if r.confidence >= 0.7
    )
    return high_conf / len(kg.relations)


def redundancy(kg: KnowledgeGraph) -> float:
    """
    Redundancy(G) — доля дублирующихся сущностей.

    Redundancy = 1 - |V_canon| / |V|
    Низкое значение = хорошо (мало дублей).
    """
    if not kg.entities:
        return 0.0
    # Считаем сущности с одинаковыми нормализованными метками
    labels = [e.label.lower().strip() for e in kg.entities.values()]
    unique = len(set(labels))
    total = len(labels)
    return 1.0 - (unique / total) if total > 0 else 0.0


def compute_quality(
    kg: KnowledgeGraph,
    validation_result: dict,
    lambda1: float = 0.3,
    lambda2: float = 0.3,
    lambda3: float = 0.2,
    lambda4: float = 0.2,
) -> dict:
    """
    Вычислить Q(G) = λ1·Cov + λ2·Cons + λ3·Prec - λ4·Red

    Параметры:
        lambda1 — вес покрытия онтологии
        lambda2 — вес логической согласованности
        lambda3 — вес точности триплетов
        lambda4 — штраф за избыточность

    Возвращает словарь со всеми компонентами и итоговым Q(G).
    """
    cov  = coverage(kg)
    cons = consistency(validation_result)
    prec = precision(kg)
    red  = redundancy(kg)

    q = lambda1 * cov + lambda2 * cons + lambda3 * prec - lambda4 * red

    result = {
        "coverage":      round(cov,  4),
        "consistency":   round(cons, 4),
        "precision":     round(prec, 4),
        "redundancy":    round(red,  4),
        "Q":             round(q,    4),
        "lambdas": {
            "lambda1": lambda1,
            "lambda2": lambda2,
            "lambda3": lambda3,
            "lambda4": lambda4,
        }
    }

    print(f"\n  [Quality] Q(G) components:")
    print(f"    Coverage     = {cov:.3f}  (λ1={lambda1})")
    print(f"    Consistency  = {cons:.3f}  (λ2={lambda2})")
    print(f"    Precision    = {prec:.3f}  (λ3={lambda3})")
    print(f"    Redundancy   = {red:.3f}  (λ4={lambda4})")
    print(f"    Q(G)         = {q:.3f}")

    return result