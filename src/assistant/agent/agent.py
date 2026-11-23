"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""
import json
import re

from collections.abc import Callable
from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import find_codes, _CODE
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
                            "В твоём ответе есть неподтверждённые коды, числа, цены или количества. "
                            "Пересмотри ответ, используя только данные из вопроса и результатов инструментов. "
                            "Удали все неподтверждённые сведения. Если ответить достоверно невозможно, "
                            "сообщи, что в базе знаний не найдено."
                        )
                    })
                    retries += 1
                    continue

                else:
                    return AgentResult("В базе знаний не найдено.", tool_steps, get_all_sources(tool_steps), True)
            else:
                return AgentResult(llm_response.content, tool_steps, get_all_sources(tool_steps))

        for tool_call in llm_response.tool_calls:
            key = (tool_call.name, frozenset(tool_call.arguments.items()))

            if key not in calls:
                error = is_valid_function(tools, tool_call)
                if error is not None:
                    add_message_with_tool(messages, tool_call, {"error": error})
                    tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, error))

                    continue

                if tool_call.name in ("get_price", "create_service_order"):
                    if not verify_answer(tool_call.arguments["part_id"], question, tool_steps):
                        error = (f'Деталь {tool_call.arguments["part_id"]} '
                                 f'не упоминается ни в вопросе, ни в результатах вызовов инструментов')
                        add_message_with_tool(messages, tool_call,{"error": error})
                        tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, error))

                        continue

                if tool_call.name == "create_service_order":
                    if not verify_answer(str(tool_call.arguments["qty"]), question, tool_steps):
                        error = "Неизвестно количество деталей, которое требуется заказать"
                        add_message_with_tool(messages, tool_call, {"error": error})
                        tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, error))

                        continue

                func = tools[tool_call.name]
                if tool_call.name == "create_service_order" and not confirm:
                    return AgentResult("Жду подтверждения.", tool_steps, get_all_sources(tool_steps), False, True,
                                       {"tool": "create_service_order", "args": tool_call.arguments})

                for _ in range(3):
                    try:
                        result = func(**tool_call.arguments)
                        break
                    except ErpUnavailable:
                        pass
                else:
                    result = {"error": "Данные ERP получить не удалось."}

            else:
                result = calls[key]

            if "error" in result:
                tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, result["error"]))
            else:
                tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, result))

            if result.get("refused"):
                return AgentResult(
                    "В базе знаний не найдено.",
                    tool_steps,
                    get_all_sources(tool_steps),
                    True
                )

            add_message_with_tool(messages, tool_call, result)
            calls[key] = result

    return AgentResult("Работа прервана. Достигнут лимит шагов.", tool_steps, get_all_sources(tool_steps))


def add_message_with_tool(messages: list[dict], tool_call: ToolCall, message: dict) -> None:
    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call],
    })

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": message
    })


def is_valid_function(tools: dict[str, Callable[..., dict]], tool_call: ToolCall) -> str | None:
    if tool_call.name not in tools:
        return f"Отсутствует инструмент {tool_call.name}."

    if "_raw" in tool_call.arguments:
        return "Аргументы не являются валидным JSON"

    schema = next(s["function"] for s in TOOL_SCHEMAS if s["function"]["name"] == tool_call.name)
    for arg in tool_call.arguments:
        if arg not in schema["parameters"]["properties"]:
            return f"Неизвестный аргумент функции {tool_call.name} - {arg}."

    return None


def qty_or_price_is_mentioned(target: int, question: str) -> str | None:
    cleaned = re.sub(_CODE, "", question)
    numbers = {int(n) for n in re.findall(r"\b\d+\b", cleaned)}
    if len(numbers) == 0:
        return "Отсутствует информация, сколько штук заказать. Спроси пользователя об этом."
    if len(numbers) > 1:
        return "В вопросе обнаружена двусмысленность: непонятно, сколько требуется заказать. Спроси про точное количество."
    if numbers != {target}:
        return f"Неверный аргумент qty: пользователь запросил {target}, а ты указал qty={numbers.pop()}."

    return None


def get_all_sources(tool_steps: list[ToolStep]) -> list[str]:
    sources = []
    for tool_step in tool_steps:
        if tool_step.result and "sources" in tool_step.result:
            sources.extend(tool_step.result["sources"])

    return list(dict.fromkeys(sources))


def extract_codes_and_numbers(raw: str) -> set:
    codes = find_codes(raw)
    cleaned = re.sub(_CODE, "", raw)
    cleaned = re.sub(r"SO-\d+", "", cleaned)
    numbers = re.findall(numbers_pattern, cleaned)
    return set(codes) | set(numbers)


def verify_answer(answer: str, question: str, tool_steps: list[ToolStep]) -> bool:
    all_mentions = extract_codes_and_numbers(question)

    for tool_step in tool_steps:
        if tool_step.result:
            all_mentions.update(extract_codes_and_numbers(tool_step.result.get("answer", "")))

            facts_text = " ".join(tool_step.result.get("facts", []))
            all_mentions.update(extract_codes_and_numbers(facts_text))

            if "price_rub" in tool_step.result:
                all_mentions.add(str(tool_step.result["price_rub"]))
            if "qty" in tool_step.result:
                all_mentions.add(str(tool_step.result["qty"]))

    llm_mentions = extract_codes_and_numbers(answer)

    return llm_mentions.issubset(all_mentions)