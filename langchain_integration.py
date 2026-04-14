"""
langchain_integration.py — LangChain-compatible wrapper для HiRAG-Ontology.

Позволяет использовать нашу систему как стандартный LangChain Retriever
и интегрировать с любым LangChain pipeline без переписывания основного кода.

Использование:
    from langchain_integration import HiRAGOntologyRetriever
    retriever = HiRAGOntologyRetriever.from_graph("results/knowledge_graph_final.json")
    docs = retriever.invoke("What treats acute lymphoblastic leukemia?")
"""

import os
import sys
from pathlib import Path
from typing import List
from dotenv import load_dotenv

load_dotenv()

# Добавляем корневую папку в path
sys.path.insert(0, str(Path(__file__).parent))

try:
    from langchain_core.retrievers import BaseRetriever
    from langchain_core.documents import Document
    from langchain_core.callbacks import CallbackManagerForRetrieverRun
    from langchain_core.language_models import BaseLLM
    from langchain_openai import ChatOpenAI
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.runnables import RunnablePassthrough
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("[LangChain] Not installed. Run: pip install langchain langchain-openai langchain-core")

from pipeline.knowledge_graph import KnowledgeGraph
from retrieval.retriever import HybridRetriever, RetrievalMode


class HiRAGOntologyRetriever:
    """
    LangChain-compatible retriever для HiRAG-Ontology.

    Оборачивает HybridRetriever в интерфейс совместимый с LangChain,
    возвращая List[Document] для интеграции с любым LangChain chain.
    """

    def __init__(
        self,
        kg: KnowledgeGraph,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        top_k: int = 10,
    ):
        self.kg = kg
        self.retriever = HybridRetriever(kg, mode=mode)
        self.top_k = top_k

    @classmethod
    def from_graph(
        cls,
        graph_path: str,
        mode: RetrievalMode = RetrievalMode.HYBRID_RRF,
        top_k: int = 10,
    ) -> "HiRAGOntologyRetriever":
        """Создать ретривер из JSON файла графа."""
        kg = KnowledgeGraph.load(graph_path)
        kg.compute_pagerank()
        return cls(kg, mode=mode, top_k=top_k)

    def get_relevant_documents(self, query: str) -> list:
        """
        Совместимый с LangChain метод получения документов.
        Возвращает List[Document] с контекстом из графа.
        """
        entity_ids = self.retriever.retrieve(query, top_k=self.top_k)
        context = self.kg.get_context_for_entities(entity_ids)

        if not LANGCHAIN_AVAILABLE:
            return [{"page_content": context,
                     "metadata": {"entity_count": len(entity_ids),
                                  "retrieval_mode": self.retriever.mode.value}}]

        return [Document(
            page_content=context,
            metadata={
                "entity_count": len(entity_ids),
                "retrieval_mode": self.retriever.mode.value,
                "top_entities": [
                    self.kg.entities[e].label
                    for e in entity_ids
                    if e in self.kg.entities
                ][:5],
            }
        )]

    def invoke(self, query: str) -> list:
        """Alias для совместимости с новым LangChain API."""
        return self.get_relevant_documents(query)


def build_langchain_rag_chain(graph_path: str):
    """
    Построить полный RAG chain используя LangChain LCEL.

    Архитектура:
        query → HiRAGOntologyRetriever → context
        (query, context) → ChatPromptTemplate → LLM → answer

    Возвращает runnable chain который принимает вопрос и возвращает ответ.
    """
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not installed. Run: pip install langchain langchain-openai")

    # Инициализируем компоненты
    retriever = HiRAGOntologyRetriever.from_graph(graph_path)

    llm = ChatOpenAI(
        model="deepseek-chat",
        api_key=os.getenv("DEEPSEEK_API_KEY"),
        base_url="https://api.deepseek.com",
        temperature=0.1,
        max_tokens=600,
    )

    prompt = ChatPromptTemplate.from_template("""You are a medical information system
with access to a structured knowledge graph of oncological clinical guidelines.

Knowledge graph context:
{context}

Question: {question}

Answer based ONLY on the provided context. Be specific and comprehensive.""")

    def format_docs(docs):
        return "\n\n".join(d.page_content for d in docs)

    # LCEL chain: query → retrieve → format → prompt → llm → parse
    chain = (
        {
            "context": lambda q: format_docs(retriever.invoke(q)),
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return chain


def demo_langchain():
    """Демонстрация LangChain интеграции."""
    print("\n=== HiRAG-Ontology LangChain Integration Demo ===\n")

    graph_path = "results/knowledge_graph_final.json"
    if not Path(graph_path).exists():
        print(f"Graph not found: {graph_path}")
        print("Run: py -m evaluation.run_eval")
        return

    # Простой retriever (без LangChain зависимости)
    retriever = HiRAGOntologyRetriever.from_graph(graph_path)
    query = "What is the treatment for acute lymphoblastic leukemia?"
    docs = retriever.invoke(query)
    print(f"Query: {query}")
    print(f"Retrieved {len(docs)} document(s)")
    if docs:
        content = docs[0].page_content if LANGCHAIN_AVAILABLE else docs[0]['page_content']
        print(f"Context preview:\n{content[:300]}...")

    # Полный LangChain chain (если установлен)
    if LANGCHAIN_AVAILABLE and os.getenv("DEEPSEEK_API_KEY"):
        print("\n--- Full LangChain RAG Chain ---")
        chain = build_langchain_rag_chain(graph_path)
        answer = chain.invoke(query)
        print(f"Answer:\n{answer}")
    else:
        print("\nTo use full LangChain chain:")
        print("  pip install langchain langchain-openai")
        print("  Set DEEPSEEK_API_KEY in .env")


if __name__ == "__main__":
    demo_langchain()
