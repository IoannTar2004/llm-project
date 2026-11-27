"""Инструменты агента. Требование FR-6."""
import re
from collections.abc import Callable
from typing import Any

from assistant.graphrag import search
from assistant.infra.erp_client import ErpClient, ErpNotFound, ErpRejected
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
warehouses = ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"]

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_knowledge",
            "description": (
                "Ищет ответ на вопрос в базе знаний сервисных бюллетеней. "
               "Возвращает answer, facts, sources и признак refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Вопрос пользователя на естественном языке."
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "summarize_topic",
            "description": (
                "Собирает сводку по теме из релевантных сообществ базы знаний. "
                "Возвращает answer, проверяемые facts, sources и признак refused."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {
                        "type": "string",
                        "description": "Тема или вопрос для сводки."
                    }
                },
                "required": ["question"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_stock",
            "description": (
                "Проверяет остаток детали на складе. "
                "Возвращает остаток и признак discontinued."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "Номер детали вида P-1042 или P-1042B."
                    },
                    "warehouse": {
                        "type": "string",
                        "enum": warehouses,
                        "description": "Склад, на котором проверяется остаток.",
                    }
                },
                "required": ["part_id", "warehouse"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": (
                "Возвращает цену детали в рублях и признак discontinued."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "Номер детали вида P-1042 или P-1042B."
                    }
                },
                "required": ["part_id"],
            }
        }
    },
{
        "type": "function",
        "function": {
            "name": "create_service_order",
            "description": (
                "Создаёт заказ на обслуживание: деталь, склад, количество, модель. "
                "Возвращает номер созданного заказа."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "part_id": {
                        "type": "string",
                        "description": "Номер детали вида P-1042 или P-1042B.",
                    },
                    "warehouse": {
                        "type": "string",
                        "enum": warehouses,
                        "description": "Склад, с которого списывается деталь.",
                    },
                    "qty": {
                        "type": "integer",
                        "minimum": 1,
                        "description": "Количество деталей, больше нуля.",
                    },
                    "model": {
                        "type": "string",
                        "description": "Код модели, для которой создаётся заказ, например AR-210.",
                    },
                },
                "required": ["part_id", "warehouse", "qty", "model"],
            },
        },
    }
]


def build_tools(kb: KnowledgeBase, llm: LLMClient, erp: ErpClient) -> dict[str, Callable[..., dict]]:
    """Словарь «имя инструмента -> функция». Каждая функция возвращает словарь.

    Функция проверяет свои аргументы до обращения к ERP: склад из списка, номер детали вида P-1042
    или P-1042B, количество больше нуля. При ошибке возвращает {"error": "..."} и ничего не выполняет.
    `ErpNotFound` и `ErpRejected` превращаются в {"error": "..."} с разным текстом, чтобы исходы
    оставались различимыми. `ErpUnavailable` передаётся дальше: повторы делает агент (FR-8).
    """
    def search_knowledge(question: str) -> dict[str, Any]:
        if not question:
            return {"error": "Пустой вопрос"}

        search_result = search.local_search(question, kb, llm)
        return {
            "answer": search_result.text,
            "facts": search_result.facts,
            "sources": search_result.sources,
            "refused": search_result.refused
        }

    def summarize_topic(question: str) -> dict[str, Any]:
        if not question:
            return {"error": "Пустой вопрос"}

        search_result = search.global_search(question, kb, llm)
        return {
            "answer": search_result.text,
            "facts": search_result.facts,
            "sources": search_result.sources,
            "refused": search_result.refused
        }

    def check_stock(part_id: str, warehouse: str) -> Any:
        if not re.match(r"P-\d{4}[A-Z]?", part_id):
            return {"error": "Невалидный код детали"}
        if warehouse not in warehouses:
            return {"error": "Неизвестный склад"}

        try:
            response = erp.get_stock(part_id, warehouse)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена в ERP: {e}"}
        except ErpRejected as e:
            return {"error": f"ERP отклонил операцию: {e}"}

        return {"qty": response["qty"], "discontinued": response["discontinued"]}

    def get_price(part_id: str) -> Any:
        if not re.match(r"P-\d{4}[A-Z]?", part_id):
            return {"error": "Невалидный код детали"}

        try:
            response = erp.get_price(part_id)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена в ERP: {e}"}
        except ErpRejected as e:
            return {"error": f"ERP отклонил операцию: {e}"}

        return {"price_rub": response["price_rub"], "discontinued": response["discontinued"]}

    def create_service_order(part_id: str, warehouse: str, qty: int, model: str) -> Any:
        if not re.match(r"P-\d{4}[A-Z]?", part_id):
            return {"error": "Невалидный код детали"}
        if warehouse not in warehouses:
            return {"error": "Неизвестный склад"}
        if qty <= 0:
            return {"error": "Число деталей должно быть >= 1"}
        if not re.match(r"[A-Z]{2}-\d{3}", model):
            return {"error": "Невалидный тип модели"}

        try:
            response = erp.create_order(part_id, warehouse, qty, model)
        except ErpNotFound as e:
            return {"error": f"Деталь не найдена в ERP: {e}"}
        except ErpRejected as e:
            return {"error": f"ERP отклонил операцию: {e}"}

        return {"order_id": response["order_id"]}

    return {
        "search_knowledge": search_knowledge,
        "summarize_topic": summarize_topic,
        "check_stock": check_stock,
        "get_price": get_price,
        "create_service_order": create_service_order,
    }