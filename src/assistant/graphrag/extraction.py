"""Извлечение сущностей и связей из чанка. Требование FR-1."""
import json
import re
from typing import Any

from assistant.infra.config import vocabulary
from assistant.infra.grounding import canonical_name, code_type, find_codes
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
    Направление связи строго определяется её типом. Порядок упоминания сущностей в тексте
    не определяет направление связи.
    Например:
    «Поставщик Х поставляет деталь P-1042» → P-1042 SUPPLIED_BY Х.
    «Линейка Бриз включает модель BZ-220» → BZ-220 BELONGS_TO Бриз.
    «Деталь P-1042 установлена в модели AR-210» → AR-210 HAS_PART P-1042.
    «Узел пусковое реле установлен в модели AR-210 (деталь P-1042)» → P-1042 IS_A пусковое реле; AR-210 HAS_PART P-1042.
    
    Аналогично с другими связями.
    Перед выдачей JSON проверяй направление каждой связи по её допустимой схеме.
    
    Код ошибки означает одно и то же во всех моделях. В каждой модели на один вид узла приходится одна деталь. Замены 
    бывают в два шага: P-2054 -> P-2054B -> P-2054C. Если после кода детали в скобках указан узел (например, P-2054 
    (нагреватель оттайки)), ты обязан извлечь этот узел как сущность Component и создать связь IS_A от детали к этому 
    узлу.
    
    Не ограничивайся наиболее очевидными связями и не пропускай связи между сущностями, упомянутыми в разных 
    предложениях. Не пропускай связь только потому, что она не описана отдельным предложением или упоминается косвенно. 
    При этом не создавай связи, которые не подтверждаются текстом. Не дублируй одну и ту же связь.
    
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
    {"entities": [
    {"name": "E-11", "type": "ErrorCode"},
    {"name": "компрессор", "type": "Component"},
    {"name": "SP-101", "type": "Procedure"}
    ], "relations": [
    {"subject": "E-11", "subject_type": "ErrorCode", "type": "INDICATES",
    "object": "компрессор", "object_type": "Component"},
    {"subject": "E-11", "subject_type": "ErrorCode", "type": "FIXED_BY",
    "object": "SP-101", "object_type": "Procedure"}
    ]}
    
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

    _vocabulary = vocabulary()
    all_codes = find_codes(chunk.text)
    inventory_entities, inventory_relations = extract_inventory_facts(
        chunk, all_codes, _vocabulary
    )
    report_entities, report_relations = extract_supplier_report_facts(
        chunk, all_codes, _vocabulary
    )
    deterministic_entities = {
        entity.node_id: entity for entity in (*inventory_entities, *report_entities)
    }
    deterministic_relations = {
        (relation.subject.node_id, relation.type, relation.object.node_id): relation
        for relation in (*inventory_relations, *report_relations)
    }

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": chunk.text}
    ]

    # Две попытки с отдельным корректирующим сообщением при невалидном JSON.
    for attempt in range(2):
        llm_response = llm.chat(messages, json_schema=EXTRACTION_SCHEMA)
        content = get_extracted_json(llm_response.content)
        if content != {}:
            break
        if attempt == 0:
            messages.append({
                "role": "user",
                "content": (
                    "Предыдущий ответ не прошёл проверку структуры JSON. "
                    "Повтори извлечение и верни корректный JSON с обязательными массивами "
                    "entities и relations, сохранив все найденные факты."
                ),
            })
    else:
        # После двух неудачных попыток возврат пустого результата по FR-1.
        return ExtractionResult(chunk.chunk_id, chunk.doc_id, [], [], 0)

    dropped = 0
    accepted_entities: dict[str, Entity] = {}

    # Канонизация и проверка сущностей, возвращённых моделью.
    for raw_entity in content["entities"]:
        entity_type = raw_entity["type"]
        canonical = canonicalize(entity_type, raw_entity["name"], _vocabulary)
        if canonical is None or (entity_type in code_types and canonical not in all_codes):
            dropped += 1
            continue

        entity = Entity(entity_type, canonical)
        accepted_entities[entity.node_id] = entity

    # Добавление сущностей из явно структурированных строк корпуса.
    for entity in deterministic_entities.values():
        accepted_entities[entity.node_id] = entity

    accepted_relations: dict[tuple[str, str, str], Relation] = {}

    # Канонизация и проверка отношений, включая наличие обоих концов среди принятых сущностей.
    for raw_relation in content["relations"]:
        relation_type = raw_relation["type"]
        if relation_type not in RELATION_SIGNATURES:
            dropped += 1
            continue

        relation_data = raw_relation
        signature = (relation_data["subject_type"], relation_data["object_type"])
        if signature != RELATION_SIGNATURES[relation_type]:
            relation_data = try_to_swap_connection(relation_data)
            signature = (relation_data["subject_type"], relation_data["object_type"])
            if signature != RELATION_SIGNATURES[relation_type]:
                dropped += 1
                continue

        subject_type = relation_data["subject_type"]
        object_type = relation_data["object_type"]
        subject_name = canonicalize(subject_type, relation_data["subject"], _vocabulary)
        object_name = canonicalize(object_type, relation_data["object"], _vocabulary)

        if subject_name is None or object_name is None:
            dropped += 1
            continue
        if subject_type in code_types and subject_name not in all_codes:
            dropped += 1
            continue
        if object_type in code_types and object_name not in all_codes:
            dropped += 1
            continue

        subject_entity = Entity(subject_type, subject_name)
        object_entity = Entity(object_type, object_name)

        # По FR-1 отношение сохраняется только при наличии обоих принятых концов.
        relation = Relation(subject_entity, relation_type, object_entity)
        if not relation.matches_schema():
            dropped += 1
            continue

        # Добавление проверенных концов отношения в список принятых сущностей.
        accepted_entities.setdefault(subject_entity.node_id, subject_entity)
        accepted_entities.setdefault(object_entity.node_id, object_entity)

        relation_id = (subject_entity.node_id, relation_type, object_entity.node_id)
        accepted_relations[relation_id] = relation

    # Добавление детерминированных отношений после проверки их схемы.
    for relation_id, relation in deterministic_relations.items():
        if relation.matches_schema():
            accepted_entities.setdefault(relation.subject.node_id, relation.subject)
            accepted_entities.setdefault(relation.object.node_id, relation.object)
            accepted_relations.setdefault(relation_id, relation)

    return ExtractionResult(
        chunk.chunk_id,
        chunk.doc_id,
        list(accepted_entities.values()),
        list(accepted_relations.values()),
        dropped,
    )

_INVENTORY_LINE = re.compile(
    r'^\s*[-*]\s*(?P<component>.+?)\s+[–—-]\s*деталь\s+'
    r'(?P<part>P-\d{4}[A-Z]?)\s*,\s*поставщик\s*[«"]'
    r'(?P<supplier>[^»"]+)[»"]\s*$',
    re.IGNORECASE | re.MULTILINE,
)


def extract_inventory_facts(
    chunk: Chunk,
    all_codes: list[str],
    vocabulary: dict[str, list[str]],
) -> tuple[list[Entity], list[Relation]]:
    """Извлечение явных соответствий модель–узел–деталь из списка состава узлов."""
    if not re.search(r"^##\s+Состав узлов\s*$", chunk.text, re.IGNORECASE | re.MULTILINE):
        return [], []

    model_codes = [code for code in all_codes if code_type(code) == "Model"]
    if len(model_codes) != 1:
        return [], []

    model = canonical_name("Model", model_codes[0], vocabulary)
    if model is None:
        return [], []

    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}
    model_entity = Entity("Model", model)
    entities[model_entity.node_id] = model_entity

    for match in _INVENTORY_LINE.finditer(chunk.text):
        component = canonical_name("Component", match.group("component"), vocabulary)
        part = canonical_name("Part", match.group("part"), vocabulary)
        supplier = canonical_name("Supplier", match.group("supplier"), vocabulary)

        if not component or not part or not supplier or part not in all_codes:
            continue

        part_entity = Entity("Part", part)
        component_entity = Entity("Component", component)
        supplier_entity = Entity("Supplier", supplier)

        for entity in (part_entity, component_entity, supplier_entity):
            entities[entity.node_id] = entity

        inventory_relations = (
            Relation(model_entity, "HAS_PART", part_entity),
            Relation(part_entity, "IS_A", component_entity),
            Relation(part_entity, "SUPPLIED_BY", supplier_entity),
        )
        for relation in inventory_relations:
            if relation.matches_schema():
                relation_id = (relation.subject.node_id, relation.type, relation.object.node_id)
                relations[relation_id] = relation

    return list(entities.values()), list(relations.values())


_SUPPLIER_REPORT = re.compile(
    r'Поставщик\s*[«"](?P<supplier>[^»"]+)[»"]\s+сообщил(?:а|и)?\s+о\s+дефекте\s+'
    r'партии\s+детали\s+(?P<part>P-\d{4}[A-Z]?)',
    re.IGNORECASE,
)


def extract_supplier_report_facts(
    chunk: Chunk,
    all_codes: list[str],
    vocabulary: dict[str, list[str]],
) -> tuple[list[Entity], list[Relation]]:
    """Извлечение явной связи поставщика с деталью из сообщения поставщика."""
    entities: dict[str, Entity] = {}
    relations: dict[tuple[str, str, str], Relation] = {}

    for match in _SUPPLIER_REPORT.finditer(chunk.text):
        part = canonical_name("Part", match.group("part"), vocabulary)
        supplier = canonical_name("Supplier", match.group("supplier"), vocabulary)
        if not part or part not in all_codes or not supplier:
            continue

        part_entity = Entity("Part", part)
        supplier_entity = Entity("Supplier", supplier)
        entities[part_entity.node_id] = part_entity
        entities[supplier_entity.node_id] = supplier_entity

        relation = Relation(part_entity, "SUPPLIED_BY", supplier_entity)
        if relation.matches_schema():
            relation_id = (part_entity.node_id, relation.type, supplier_entity.node_id)
            relations[relation_id] = relation

    return list(entities.values()), list(relations.values())


def get_extracted_json(llm_response: str) -> Any:
    """
    Валидирует сгенерированный моделью json согласно схеме.
    """
    try:
        json_response = json.loads(llm_response)
    except json.decoder.JSONDecodeError:
        return {}

    if not isinstance(json_response, dict) or not assert_fields(json_response, ["entities", "relations"]):
        return {}

    entities = json_response["entities"]
    relations = json_response["relations"]
    if not isinstance(entities, list) or not isinstance(relations, list):
        return {}

    for entity in entities:
        if not isinstance(entity, dict) or not assert_fields(entity, ["name", "type"]):
            return {}
        if not isinstance(entity["name"], str) or not isinstance(entity["type"], str):
            return {}
    for relation in relations:
        if not isinstance(relation, dict) or not assert_fields(
            relation, ["subject", "object", "subject_type", "object_type", "type"]
        ):
            return {}
        if not all(isinstance(relation[field], str) for field in (
            "subject", "object", "subject_type", "object_type", "type"
        )):
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
    if stype in code_types or stype in named_types:
        return canonical_name(stype, sname, vocabulary)
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