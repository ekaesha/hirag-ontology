"""
ontology_loader.py — загружает онтологию из JSON файла.

Заменяет захардкоженные словари в typing_agent.py и validator.py.
Теперь онтологию можно менять без изменения кода.
"""

import json
from pathlib import Path


_ONTOLOGY_CACHE = None
_ONTOLOGY_PATH = Path(__file__).parent.parent / "ontology.json"


def load_ontology(path: str = None) -> dict:
    """
    Загрузить онтологию из JSON файла.
    Результат кэшируется — повторные вызовы не читают файл.
    """
    global _ONTOLOGY_CACHE
    if _ONTOLOGY_CACHE is not None:
        return _ONTOLOGY_CACHE

    ontology_path = Path(path) if path else _ONTOLOGY_PATH

    if not ontology_path.exists():
        raise FileNotFoundError(
            f"Ontology file not found: {ontology_path}\n"
            f"Create ontology.json in the project root."
        )

    with open(ontology_path, encoding="utf-8") as f:
        _ONTOLOGY_CACHE = json.load(f)

    print(f"[Ontology] Loaded: {len(_ONTOLOGY_CACHE['classes'])} classes, "
          f"{len(_ONTOLOGY_CACHE['properties'])} properties, "
          f"{len(_ONTOLOGY_CACHE['axioms'])} axioms "
          f"← {ontology_path.name}")
    return _ONTOLOGY_CACHE


def get_classes(path: str = None) -> dict:
    """Вернуть словарь классов {ClassName: description}."""
    return load_ontology(path)["classes"]


def get_properties(path: str = None) -> dict:
    """Вернуть словарь свойств {predicate: {domain, range, description}}."""
    return load_ontology(path)["properties"]


def get_axioms(path: str = None) -> list:
    """Вернуть список аксиом."""
    return load_ontology(path)["axioms"]


def get_valid_types(path: str = None) -> set:
    """Вернуть множество допустимых типов сущностей."""
    return set(get_classes(path).keys())


def get_valid_predicates(path: str = None) -> set:
    """Вернуть множество допустимых предикатов."""
    return set(get_properties(path).keys())


def reload_ontology():
    """Сбросить кэш и перечитать файл."""
    global _ONTOLOGY_CACHE
    _ONTOLOGY_CACHE = None
    return load_ontology()