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
        if not entity["name"] or entity_type in code_types and entity["name"] not in all_codes:
            entity["dropped"] = True
            dropped += 1
            continue

    entities = [Entity(e["type"], e["name"]) for e in content["entities"] if "dropped" not in e]

    # обработка связей
    relations = []
    for relation in content["relations"]:
        type = relation["type"]
        object_type = relation["object_type"]
        subject_type = relation["subject_type"]

        if type not in RELATION_SIGNATURES:
            dropped += 1
            relation["dropped"] = True
            continue

        if (subject_type, object_type) != RELATION_SIGNATURES[type]:
            dropped += 1
            relation["dropped"] = True
            continue

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


def canonicalize(stype: str, sname: str, vocabulary: dict) -> str:
    """
    Приведение кода/названия к каноническому виду
    """
    if stype in code_types:
        return normalize_code(sname)
    if stype in named_types:
        canonical = canonical_name(stype, sname, vocabulary)
        return canonical

    return ""
