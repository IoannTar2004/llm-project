"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""
import json
import re

from collections.abc import Callable
from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import normalize_code, find_codes, _CODE
from assistant.infra.llm import LLMClient, ToolCall
from assistant.infra.schemas import AgentResult, ToolStep


numbers_pattern = r"\b\d+\b"


def run_agent(
    question: str,
    llm: LLMClient,
    tools: dict[str, Callable[..., dict]],
    *,
    max_steps: int = 6,
    confirm: bool = False,
) -> AgentResult:
    """Ведёт диалог с моделью в цикле «рассуждение, вызов инструмента, наблюдение».

    Модель получает `TOOL_SCHEMAS` через `llm.chat(..., tools=...)`. Каждый вызов записывается
    в `steps` как `ToolStep`. Агент сам проверяет номер детали в аргументах (он должен встречаться
    в вопросе или в результатах прошлых инструментов), количество в заказе (должно быть названо
    пользователем) и повторные вызовы. При `ErpUnavailable` вызов повторяется до двух раз.
    Заказ создаётся только при `confirm=True`, иначе агент возвращает `needs_confirmation=True`
    и `pending_action`. Перед выдачей ответа коды, цены и остатки сверяются с вопросом
    и результатами инструментов.
    """
    # TODO: цикл, проверки аргументов, повторы, подтверждение, проверка итогового ответа'

    system_prompt = """
    Ты экспертный ИИ-ассистент, интегрированный с ERP-системой. Твоя задача - помогать пользователям находить 
    информацию о деталях и оформлять заказы. 
    
    Ограничения:
    1. Запрещено придумывать коды деталей, моделей, складов. Все данные берутся из вопроса или из истории. 
    2. Запрещено придумывать количество. Если его нет в вопросе - спроси пользователя.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": question}
    ]

    tool_steps = []
    calls = {}
    retries = 0

    for _ in range(max_steps):
        llm_response = llm.chat(messages, tools=TOOL_SCHEMAS)
        if not llm_response.tool_calls and llm_response.content:
            if not verify_answer(llm_response.content, question, tool_steps):
                if retries < 1:
                    messages.append({
                        "role": "user",
                        "content":  (
                            "Твой ответ содержит неподтверждённые данные."
                            "Исправь его, используя только вопрос и результаты инструментов."
                        )
                    })
                    retries += 1
                    continue

                else:
                    return AgentResult("В базе знаний не найдено.", tool_steps, get_all_sources(tool_steps), True)
            else:
                return AgentResult(llm_response.content, tool_steps, get_all_sources(tool_steps))

        for tool_call in llm_response.tool_calls:
            key = (tool_call.name, frozenset(tool_call.arguments))

            if key not in calls:
                error = is_valid_function(tools, tool_call)
                if error is not None:
                    add_message_with_tool(messages, tool_call, {"error": error})
                    continue

                if tool_call.name in ("get_price", "create_service_order"):
                    if not verify_answer(tool_call.arguments["part_id"], question, tool_steps):
                        add_message_with_tool(messages, tool_call,
                                          {"error":
                                                        f'Деталь {tool_call.arguments["part_id"]} '
                                                        f'не упоминается ни в вопросе, ни в результатах вызовов инструментов'
                                                    })
                        continue
    raise NotImplementedError