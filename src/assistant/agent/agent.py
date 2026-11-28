"""Цикл агента. Требования FR-7, FR-8, FR-9 и FR-10."""
import json
import re
from collections.abc import Callable

from assistant.agent.tools import TOOL_SCHEMAS
from assistant.infra.erp_client import ErpUnavailable
from assistant.infra.grounding import _CODE, find_codes
from assistant.infra.llm import LLMClient, ToolCall
from assistant.infra.schemas import AgentResult, ToolStep

number_pattern = re.compile(r"(?<![\w-])(?:\d{1,3}(?:[ \u00a0\u202f]\d{3})+|\d+)(?![\w-])")


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
    system_prompt = """
    Ты экспертный ИИ-ассистент, работающий с базой знаний и ERP.

    Порядок работы:
    1. Если для ответа нужны сведения из базы знаний, сначала вызови search_knowledge с исходным
       вопросом. Это правило действует независимо от того, указан ли в вопросе код детали P-..., 
       модели, код ошибки или название узла. Наличие кода детали в вопросе не заменяет поиск и не означает, 
       что сведения о ней уже известны.
    2. Для check_stock, get_price и create_service_order используй только Part-код, который прямо
       указан пользователем или присутствует в фактах, возвращённых search_knowledge.
    3. Если деталь указана через модель, ошибку или название узла, сначала найди связь с Part
       через search_knowledge, затем обращайся к ERP.
    4. При цепочке замен используй последнюю деталь в цепочке как актуальную. Не запрашивай ERP
       для снятой с производства детали, если в фактах есть её замена.
    5. Количество для заказа должно быть явно указано пользователем в исходном вопросе.
       Не выводи количество из остатка склада или из других результатов инструментов.
    6. Не придумывай коды деталей, моделей, склады, цены и количества. Для фактического ответа
       используй результаты инструментов и проверяемые факты, а не неподтверждённые догадки.

    Завершение запроса:
    Перед формированием окончательного ответа определи, все ли сведения и действия, запрошенные пользователем, 
    выполнены. Если пользователь запросил несколько сведений, выполни все необходимые обращения к инструментам для 
    получения этих сведений. Результаты search_knowledge используются для получения технических сведений и определения 
    кодов деталей. Они не заменяют проверку данных в ERP. Если по результатам search_knowledge определена деталь, а 
    пользователь запросил её наличие, цену или складской остаток, используй соответствующие ERP-инструменты для 
    проверки этих сведений. Не завершай ответ сообщением о том, что данные необходимо дополнительно проверить, 
    если соответствующий инструмент доступен и проверка ещё не выполнена. Формируй окончательный ответ после выполнения 
    всех запрошенных проверок либо после выяснения, что выполнить их невозможно.
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
            # Если данные в ответе не соответствуют данным из вопроса и из результатов инструментов, просим модель
            # перепроверить данные и повторно сгенерировать ответ (FR-10).
            if not verify_answer(llm_response.content, question, tool_steps):
                if retries < 1:
                    messages.append({
                        "role": "user",
                        "content": (
                            "В твоём ответе есть неподтверждённые коды, числа, цены или количества. "
                            "Пересмотри ответ, используя только данные из вопроса и результатов инструментов. "
                            "Удали все неподтверждённые сведения. Если ответить достоверно невозможно, "
                            "сообщи, что в базе знаний не найдено."
                        )
                    })
                    retries += 1
                    continue

                else:
                    # Если после повторной генерации верификация ответа не прошла, то возвращаем ответ, что в базе
                    #  знаний не найдено (FR-10).
                    return AgentResult("В базе знаний не найдено.", tool_steps, get_all_sources(tool_steps), True)
            else:
                return AgentResult(llm_response.content, tool_steps, get_all_sources(tool_steps))

        for tool_call in llm_response.tool_calls:
            key = (tool_call.name, frozenset(tool_call.arguments.items()))

            if key not in calls:
                # Валидация инструмента и его аргументов (FR-7).
                error = is_valid_function(tools, tool_call)
                if error is not None:
                    add_message_with_tool(messages, tool_call, {"error": error})
                    tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, error))
                    continue

                # Проверка того, что part_id содержится в вопросе и/или в результатах вызванных инструментов (FR-7).
                if tool_call.name in ("get_price", "create_service_order"):
                    if not verify_answer(tool_call.arguments["part_id"], question, tool_steps):
                        error = (f'Деталь {tool_call.arguments["part_id"]} '
                                 f'не упоминается ни в вопросе, ни в результатах вызовов инструментов')
                        add_message_with_tool(messages, tool_call, {"error": error})
                        tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, error))
                        continue

                # Проверка того, что qty содержится в вопросе и/или в результатах вызванных инструментов (FR-7).
                if tool_call.name == "create_service_order":
                    if not verify_answer(str(tool_call.arguments["qty"]), question, tool_steps):
                        error = "Неизвестно количество деталей, которое требуется заказать. Уточни у пользователя."
                        add_message_with_tool(messages, tool_call, {"error": error})
                        tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, error))

                        continue

                func = tools[tool_call.name]

                # Если confirm - False, возвращаем ответ, требующий подтверждения создания заказа от пользователя
                # (FR-9).
                if tool_call.name == "create_service_order" and not confirm:
                    answer = (
                        f"Подготовлен заказ на {tool_call.arguments['qty']} шт. детали "
                        f"{tool_call.arguments['part_id']} для модели {tool_call.arguments['model']}, "
                        f"склад: {tool_call.arguments['warehouse']}. "
                        "Жду подтверждения."
                    )
                    return AgentResult(answer, tool_steps, get_all_sources(tool_steps), False, True,
                                       {"tool": "create_service_order", "args": tool_call.arguments})

                # В случае возникновения ошибки ErpUnavailable повторяем еще 2 раза, иначе возвращаем ошибку, что данные
                # erp получить не удалось (FR-8).
                for _ in range(3):
                    try:
                        result = func(**tool_call.arguments)
                        break
                    except ErpUnavailable:
                        pass
                else:
                    result = {"error": "Данные ERP получить не удалось."}

            else:
                # Если ранее инструмент с такими аргументами вызывался, то извлекаем сохраненный ответ (FR-7).
                result = calls[key]

            # Передача LLM результата ERP 404 как отказа с refused=True.
            if result.get("error_type") == "not_found":
                error = result.get("error", "Деталь не найдена в ERP.")
                tool_steps.append(ToolStep(tool_call.name, tool_call.arguments, None, error))
                return AgentResult(
                    "В базе знаний не найдено.",
                    tool_steps,
                    get_all_sources(tool_steps),
                    True,
                )

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

    # Лимит шагов исчерпан, агент возвращает то, что успел узнать (FR-7).
    return AgentResult("Работа прервана. Достигнут лимит шагов.", tool_steps, get_all_sources(tool_steps))


def add_message_with_tool(messages: list[dict], tool_call: ToolCall, message: dict) -> None:
    # Превращаем объект ToolCall в словарь для JSON
    tool_call_dict = {
        "id": tool_call.id,
        "type": "function",
        "function": {
            "name": tool_call.name,
            # Аргументы должны быть строкой JSON!
            "arguments": json.dumps(tool_call.arguments, ensure_ascii=False)
        }
    }

    messages.append({
        "role": "assistant",
        "content": None,
        "tool_calls": [tool_call_dict],
    })

    messages.append({
        "role": "tool",
        "tool_call_id": tool_call.id,
        "content": json.dumps(message, ensure_ascii=False)
    })


def is_valid_function(tools: dict[str, Callable[..., dict]], tool_call: ToolCall) -> str | None:
    """Проверка имени инструмента, обязательных аргументов, типов и ограничений схемы."""
    if tool_call.name not in tools:
        return f"Отсутствует инструмент {tool_call.name}."

    arguments = tool_call.arguments
    if "_raw" in arguments:
        return "Аргументы не являются валидным JSON"

    schema = next(
        (item["function"] for item in TOOL_SCHEMAS if item["function"]["name"] == tool_call.name),
        None,
    )
    if schema is None:
        return f"Для инструмента {tool_call.name} отсутствует схема аргументов."
    parameters = schema["parameters"]
    properties = parameters["properties"]

    for argument in arguments:
        if argument not in properties:
            return f"Неизвестный аргумент функции {tool_call.name} — {argument}."

    missing = [argument for argument in parameters.get("required", []) if argument not in arguments]
    if missing:
        return f"Не переданы обязательные аргументы функции {tool_call.name}: {', '.join(missing)}."

    for argument, value in arguments.items():
        property_schema = properties[argument]
        expected_type = property_schema.get("type")
        if expected_type == "string" and not isinstance(value, str):
            return f"Аргумент {argument} функции {tool_call.name} должен быть строкой."
        if expected_type == "integer" and (isinstance(value, bool) or not isinstance(value, int)):
            return f"Аргумент {argument} функции {tool_call.name} должен быть целым числом."
        if expected_type == "integer" and value < property_schema.get("minimum", value):
            return f"Аргумент {argument} функции {tool_call.name} должен быть не меньше {property_schema['minimum']}."
        if "enum" in property_schema and value not in property_schema["enum"]:
            return f"Недопустимое значение аргумента {argument} функции {tool_call.name}."

    return None


def get_all_sources(tool_steps: list[ToolStep]) -> list[str]:
    """
    Собирает все источники, к которым обращалась модель.
    """
    sources = []
    for tool_step in tool_steps:
        if tool_step.result and "sources" in tool_step.result:
            sources.extend(tool_step.result["sources"])

    return list(dict.fromkeys(sources))


def extract_codes_and_numbers(raw: str) -> set[str]:
    """Извлечение кодов и чисел с нормализацией разделителей тысяч."""
    codes = set(find_codes(raw))
    cleaned = re.sub(_CODE, " ", raw)
    cleaned = re.sub(r"SO-\d+", " ", cleaned)
    numbers = {
        re.sub(r"\D", "", match.group())
        for match in number_pattern.finditer(cleaned)
    }
    return codes | numbers


def verify_answer(answer: str, question: str, tool_steps: list[ToolStep]) -> bool:
    """Проверка кодов и чисел по вопросу, фактам графа и структурированным ответам ERP."""
    allowed_mentions = extract_codes_and_numbers(question)

    for tool_step in tool_steps:
        result = tool_step.result
        if not result:
            continue

        # Текст answer у search_knowledge/summarize_topic создан LLM и не считается доказательством.
        facts = result.get("facts") or []
        if isinstance(facts, list):
            allowed_mentions.update(extract_codes_and_numbers(" ".join(map(str, facts))))

        # Структурированные поля ERP являются проверяемыми источниками числовых значений.
        for field in ("price_rub", "qty"):
            if field in result:
                allowed_mentions.add(str(result[field]))

    answer_mentions = extract_codes_and_numbers(answer)
    return answer_mentions.issubset(allowed_mentions)