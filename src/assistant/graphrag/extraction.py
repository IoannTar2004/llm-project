"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json
from typing import Any

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, find_codes, normalize_code
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import (
    EXTRACTION_SCHEMA,
    RELATION_SIGNATURES,
    Chunk,
    Entity,
    ExtractionResult,
    Relation,
)

code_types = ("Part", "Model", "ErrorCode", "Procedure")
named_types = ("Component", "Supplier", "ProductLine")

def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """

    system_prompt = """
    Роль: Ты извлекаешь сущности и связи из текста сервисного бюллетеня.
    
    Список допустимых типов сущностей: 
    1. ProductLine - название линейки. Пример: Арктика.
    2. Model - код модели. Пример: AR-210. 
    3. Component - название узла строчными буквами. Пример: пусковое реле.
    4. Part - номер детали. Пример: P-1042.
    5. ErrorCode - код ошибки. Пример: E-17.
    6. Procedure - номер регламента. Пример: SP-104.
    7. Supplier - название поставщика. Пример: КомпрессорМаш.
    
    Список допустимых типов связей с направлением:
    1. BELONGS_TO (Model → ProductLine) - модель входит в линейку.
    2. HAS_PART (Model → Part) - в модели установлена деталь.
    3. IS_A (Part → Component) - деталь является узлом этого вида.
    4. INDICATES (ErrorCode → Component) - ошибка указывает на отказ узла.
    5. FIXED_BY (ErrorCode → Procedure) - ошибка устраняется по регламенту.
    6. REPLACED_BY (Part → Part) - деталь снята с производства и заменена. Важно: REPLACED_BY идёт от старой детали 
    к новой, а не наоборот.
    7. SUPPLIED_BY (Part → Supplier) - деталь поставляет поставщик.
    8. FAILED (Model → Component) - в отчёте о ремонте зафиксирован отказ узла.
    
    Важно: 
    Связь должна быть такой (Subject → Object).
    Правила направления связей:
    Направление связи строго определяется её типом. Порядок упоминания сущностей в тексте не определяет направление связи.
    Например:
    «Поставщик Х поставляет деталь P-1042» → P-1042 SUPPLIED_BY Х.
    «Линейка Бриз включает модель BZ-220» → BZ-220 BELONGS_TO Бриз.
    «Деталь P-1042 установлена в модели AR-210» → AR-210 HAS_PART P-1042.
    «Узел пусковое реле установлен в модели AR-210 (деталь P-1042)» → P-1042 IS_A пусковое реле; AR-210 HAS_PART P-1042.
    
    Аналогично с другими связями.
    Перед выдачей JSON проверяй направление каждой связи по её допустимой схеме.
    
    Код ошибки означает одно и то же во всех моделях. В каждой модели на один 
    вид узла приходится одна деталь. Замены бывают в два шага: P-2054 -> P-2054B -> P-2054C. Если после кода детали в 
    скобках указан узел (например, P-2054 (нагреватель оттайки)), ты обязан извлечь этот узел как сущность Component 
    и создать связь IS_A от детали к этому узлу. Извлекай коды/названия, которые указаны в запросе: ничего не
    выдумывай!
    
    1. Пример для REPLACED_BY:
    Текст: Деталь P-2054 (нагреватель оттайки) снята с производства. Вместо неё устанавливается деталь P-2054B. 
    Поставщик детали P-2054B – «ТермоПрибор». Детали взаимозаменяемы по посадочным местам и разъёмам, доработка 
    оборудования не требуется. Новые заказы оформляются только на P-2054B.
    
    Правильный ответ:
    {'entities': [{'name': 'нагреватель оттайки', 'type': 'Component'}, {'name': 'P-2054', 'type': 'Part'}, 
    {'name': 'P-2054B', 'type': 'Part'}, {'name': 'ТермоПрибор', 'type': 'Supplier'}], 
    'relations': [{'object': 'нагреватель оттайки', 'object_type': 'Component', 'subject': 'P-2054', 
    'subject_type': 'Part', 'type': 'IS_A'}, {'object': 'P-2054B', 'object_type': 'Part', 'subject': 'P-2054', 
    'subject_type': 'Part', 'type': 'REPLACED_BY'}, {'object': 'ТермоПрибор', 'object_type': 'Supplier', 'subject': 
    'P-2054B', 'subject_type': 'Part', 'type': 'SUPPLIED_BY'}]}
    
    2. Пример для INDICATES и FIXED_BY:
    Текст: При ошибке E-11 отказал компрессор, ремонт по регламенту SP-101.
    
    Правильный ответ:
    {"entities": [{"name": "E-11", "type": "ErrorCode"}, {"name": "компрессор", "type": "Component"}, {"name": "SP-101", 
    "type": "Procedure"}], "relations": [{"subject": "E-11", "subject_type": "ErrorCode", "type": "INDICATES", "object":
    "компрессор", "object_type": "Component"}, {"subject": "E-11", "subject_type": "ErrorCode", "type": "FIXED_BY", 
    "object": "SP-101", "object_type": "Procedure"}]}
    
    3. Пример для HAS_PART, IS_A и SUPPLIED_BY (из списка узлов):
    Текст: В AR-210 установлены следующие узлы. пусковое реле – деталь P-1042, поставщик «ХолодКомплект».
    
    Правильный ответ:
    {"entities": [{"name": "AR-210", "type": "Model"}, {"name": "пусковое реле", "type": "Component"}, 
    {"name": "P-1042", "type": "Part"}, {"name": "ХолодКомплект", "type": "Supplier"}], "relations": [
    {"subject": "AR-210", "subject_type": "Model", "type": "HAS_PART", "object": "P-1042", "object_type": "Part"}, 
    {"subject": "P-1042", "subject_type": "Part", "type": "IS_A", "object": "пусковое реле", 
    "object_type": "Component"}, {"subject": "P-1042", "subject_type": "Part", "type": "SUPPLIED_BY", 
    "object": "ХолодКомплект", "object_type": "Supplier"}]}
    
    4. Пример для BELONGS_TO (из руководства):
    Текст: Моноблок BZ-220 входит в линейку «Бриз» компании «ПолярТех».
    
    Правильный ответ:
    {"entities": [{"name": "BZ-220", "type": "Model"}, {"name": "Бриз", "type": "ProductLine"}], 
    "relations": [{"subject": "BZ-220", "subject_type": "Model", "type": "BELONGS_TO", "object": "Бриз", 
    "object_type": "ProductLine"}]}
    
    5. Пример для FAILED (из отчета о выезде!):
    Текст: Оборудование: витрина AR-210. Диагностика показала отказ узла: пусковое реле.
    
    Правильный ответ:
    {"entities": [{"name": "AR-210", "type": "Model"}, {"name": "пусковое реле", "type": "Component"}], 
    "relations": [{"subject": "AR-210", "subject_type": "Model", "type": "FAILED", "object": "пусковое реле", 
    "object_type": "Component"}]}
    
    Требование: Возвращай только JSON по схеме, без пояснений.
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk.text}
    ]

    # Делаем две попытки на генерацию валидного json
    for _ in range(2):
        llm_response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
        content = get_extracted_json(llm_response.content)
        if content != {}:
            break
    else:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id, [], [], 0)

    dropped = 0
    _vocabulary = vocabulary()
    all_codes = find_codes(chunk.text)
    # обработка сущностей
    for entity in content["entities"]:
        entity_type = entity["type"]

        # приведение названий деталей к каноническому виду
        entity["name"] = canonicalize(entity["type"], entity["name"], _vocabulary)
        if entity["name"] is None or entity_type in code_types and entity["name"] not in all_codes:
            entity["dropped"] = True
            dropped += 1
            continue

    entities = [Entity(e["type"], e["name"]) for e in content["entities"] if "dropped" not in e]

    # обработка связей
    relations = []
    for i, relation in enumerate(content["relations"]):
        type = relation["type"]

        # Проверка, что связь корректна
        if type not in RELATION_SIGNATURES:
            dropped += 1
            relation["dropped"] = True
            continue

        signature = (relation["subject_type"], relation["object_type"])
        if signature != RELATION_SIGNATURES[type]:
            relation = try_to_swap_connection(relation)
            signature = (relation["subject_type"], relation["object_type"])

            if signature != RELATION_SIGNATURES[type]:
                dropped += 1
                relation["dropped"] = True
                continue
            else:
                content["relations"][i] = relation

        # приведение связей к каноническому виду
        ok = True
        for side in ("subject", "object"):
            relation[side] = canonicalize(relation[f"{side}_type"], relation[side], _vocabulary)
            if not relation[side]:
                ok = False
                break
        if not ok:
            dropped += 1
            relation["dropped"] = True
            continue

        # Проверяем, что все сущности в связях есть в тексте чанка
        subject = relation["subject"]
        object = relation["object"]
        object_type = relation["object_type"]
        subject_type = relation["subject_type"]

        if subject_type in code_types and subject not in all_codes:
            dropped += 1
            continue

        if object_type in code_types and object not in all_codes:
            dropped += 1
            continue

        relations.append(Relation(Entity(subject_type, subject), type, Entity(object_type, object)))

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, entities, relations, dropped)


def get_extracted_json(llm_response: str) -> Any:
    """
    Валидирует сгенерированный моделью json согласно схеме.
    """
    try:
        json_response = json.loads(llm_response)
    except json.decoder.JSONDecodeError:
        return {}

    if not assert_fields(json_response, ["entities", "relations"]):
        return {}

    entities = json_response["entities"]
    relations = json_response["relations"]

    for e in entities:
        if not assert_fields(e, ["name", "type"]):
            return {}
    for r in relations:
        if not assert_fields(r, ["subject", "object", "subject_type", "object_type", "type"]):
            return {}

    return json_response

def assert_fields(json_response: dict, fields: list) -> bool:
    for field in fields:
        if field not in json_response:
            return False

    return True


def canonicalize(stype: str, sname: str, vocabulary: dict) -> str | None:
    """
    Приведение кода/названия к каноническому виду
    """
    if stype in code_types:
        return normalize_code(sname)
    if stype in named_types:
        canonical = canonical_name(stype, sname, vocabulary)
        return canonical

    return None


def try_to_swap_connection(relation: dict[str, str]) -> dict[str, Any]:
    relation_type = relation["type"]
    signature = (relation["subject_type"], relation["object_type"])

    if signature[0] != signature[1] and signature[::-1] == RELATION_SIGNATURES[relation_type]:
        return {
            "subject": relation["object"],
            "object": relation["subject"],
            "subject_type": relation["object_type"],
            "object_type": relation["subject_type"],
            "type": relation_type
        }

    return relation