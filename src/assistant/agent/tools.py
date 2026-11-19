"""Инструменты агента. Требование FR-6."""
import re
from collections.abc import Callable

from assistant.graphrag import search
from assistant.infra.erp_client import ErpClient
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import KnowledgeBase, Answer

# Описания пяти инструментов в формате function calling:
# search_knowledge, summarize_topic, check_stock, get_price, create_service_order.
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
                "Собирает сводку по теме из нескольких документов базы знаний. "
                "Возвращает answer, sources и признак refused."
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
                        "enum": ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"],
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
                        "enum": ["Москва", "Санкт-Петербург", "Казань", "Новосибирск"],
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
    # TODO: пять функций поверх local_search, global_search и клиента ERP
