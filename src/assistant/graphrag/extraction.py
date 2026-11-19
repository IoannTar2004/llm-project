"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json
from typing import Any

from assistant.infra.config import vocabulary
from assistant.infra.grounding import normalize_code, find_codes, canonical_name
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Chunk, ExtractionResult, EXTRACTION_SCHEMA, ENTITY_TYPES, RELATION_SIGNATURES, \
    Entity, Relation


def extract_chunk(chunk: Chunk, llm: LLMClient) -> ExtractionResult:
    """Возвращает сущности и связи чанка строго по схеме графа.

    Модель вызывается через `llm.chat(..., json_schema=EXTRACTION_SCHEMA)`. Запрос описывает схему
    с направлением каждой связи и содержит хотя бы один пример. Всё, что не подтверждается текстом,
    отбрасывается и считается в `dropped`: код, которого нет в чанке, название не из словаря,
    связь с отброшенным концом. Пригодятся `find_codes` и `canonical_name` из `assistant.infra.grounding`.
    Функция не падает ни на каком ответе модели: одна повторная попытка, затем пустой результат.
    """
    # TODO: запрос к модели, разбор JSON, проверки по тексту чанка и словарю

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
    3. IS_A (Part → Node) - деталь является узлом этого вида.
    4. INDICATES (ErrorCode → Node) - ошибка указывает на отказ узла.
    5. FIXED_BY (ErrorCode -> Procedure) - ошибка устраняется по регламенту.
    6. REPLACED_BY (Part → Part) - деталь снята с производства и заменена. Важно: REPLACED_BY идёт от старой детали 
    к новой, а не наоборот.
    7. SUPPLIED_BY (Part → Supplier) - деталь поставляет поставщик.
    8. FAILED (Model → Component) - в отчёте о ремонте зафиксирован отказ узла.
    
    Важно: Связь должна быть такой Subject → Object.
    
    Код ошибки означает одно и то же во всех моделях. В каждой модели на один вид узла приходится одна деталь. 
    Замены бывают в два шага: P-2054 -> P-2054B -> P-2054C.
    
    Пример для REPLACED_BY:
    Текст: «Деталь P-1 снята с производства, вместо неё устанавливается P-2.»
    Правильный ответ:
    {"subject": "P-1", "subject_type": "Part", "type": "REPLACED_BY", "object": "P-2", "object_type": "Part"}
    
    Требование: «Возвращай только JSON по схеме, без пояснений».
    """

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk.text}
    ]

    for attempt in range(2):
        llm_response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
        content = get_extracted_json(llm_response.content)
        if content != {}:
            break
    else:
        return ExtractionResult(chunk.chunk_id, chunk.doc_id, [], [], 0)

    dropped = 0
    _vocabulary = vocabulary()
    parts = find_codes(chunk.text)

    for entity in content["entities"]:
        type = entity["type"]
        if type not in ENTITY_TYPES:
            entity["dropped"] = True
            dropped += 1
            continue

        # приведение названий деталей к каноническому виду
        name = entity["name"]
        if type == "Part":
            name = normalize_code(name)
            if name not in parts:
                dropped += 1
                entity["dropped"] = True
                continue
            entity["name"] = name

        # Проверка того, что компонент/поставщик/продуктовая линия есть в vocabulary.json
        elif type in ["Component", "Supplier", "ProductLine"]:
            name = canonical_name(type, name, _vocabulary)
            if not name:
                dropped += 1
                entity["dropped"] = True

    for relation in content["relations"]:
        type = relation["type"]
        if type not in RELATION_SIGNATURES:
            dropped += 1
            relation["dropped"] = True
            continue

        signature = (relation["subject_type"], relation["object_type"])
        if signature != RELATION_SIGNATURES[type]:
            dropped += 1
            relation["dropped"] = True
            continue

        # приведение названий деталей к каноническому виду
        name = relation["object"]
        if relation["object_type"] == "Part":
            name = normalize_code(name)
            if name not in parts:
                dropped += 1
                relation["dropped"] = True
                continue
            relation["object"] = name

        name = relation["subject"]
        if relation["subject_type"] == "Part":
            name = normalize_code(name)
            if name not in parts:
                dropped += 1
                relation["dropped"] = True
                continue
            relation["subject"] = name

    entities = [Entity(e["type"], e["name"]) for e in content["entities"] if "dropped" not in e]
    relations = [Relation(r["subject"], r["type"], r["object"]) for r in content["relations"] if "dropped" not in r]

    return ExtractionResult(chunk.chunk_id, chunk.doc_id, entities, relations, dropped)


def get_extracted_json(llm_response: str) -> Any:
    try:
        json_response = json.loads(llm_response)
    except json.decoder.JSONDecodeError as e:
        return {}

    if not assert_fields("json_response", json_response, ["entities", "relations"]):
        return {}

    entities = json_response["entities"]
    relations = json_response["relations"]

    for i, e in enumerate(entities):
        if not assert_fields(f"entities[{i}]", e, ["name", "type"]):
            return {}
    for i, r in enumerate(relations):
        if not assert_fields(f"relations[{i}]", r, ["subject", "object", "subject_type", "object_type", "type"]):
            return {}

    return json_response

def assert_fields(name:str, json_response: dict, fields: list) -> bool:
    for field in fields:
        if field not in json_response:
            return False

    return True