"""Сборка графа знаний из результатов извлечения. Требование FR-2."""

import networkx as nx

from assistant.infra.schemas import ExtractionResult


def build_graph(results: list[ExtractionResult]) -> nx.DiGraph:
    """Объединяет результаты по всем чанкам в один направленный граф.

    Узел: идентификатор `Entity.node_id`, атрибуты `type`, `name`, `sources` (список chunk_id).
    Ребро: атрибуты `type`, `weight` (число разных чанков, где связь найдена), `sources`.
    Связи, не соответствующие схеме (`Relation.matches_schema`), отбрасываются. Если между двумя
    деталями есть `REPLACED_BY` в обе стороны, остаётся связь с большим весом, при равенстве
    отбрасываются обе. Число таких конфликтов записывается в `graph.graph["conflicts"]`.
    """
    # узлы, рёбра с весами, проверка схемы, разбор встречных замен
    graph = nx.DiGraph()

    node_sources: dict[str, set[str]] = {}
    edge_data: dict[tuple[str, str], dict] = {}

    for result in results:
        chunk_id = result.chunk_id

        def add_node(entity, source_chunk_id: str) -> None:
            node_id = entity.node_id

            if node_id not in node_sources:
                node_sources[node_id] = set()

            node_sources[node_id].add(source_chunk_id)

        # Добавление всех извлечённых сущностей, в том числе без связей
        for entity in result.entities:
            add_node(entity, chunk_id)

        # Сбор только связей, соответствующих схеме графа
        for relation in result.relations:
            if not relation.matches_schema():
                continue

            subject_id = relation.subject.node_id
            object_id = relation.object.node_id

            # Самоссылочная замена не описывает замену одной детали другой.
            if relation.type == "REPLACED_BY" and subject_id == object_id:
                continue

            # Концы допустимой связи также должны присутствовать в графе
            add_node(relation.subject, chunk_id)
            add_node(relation.object, chunk_id)

            edge_key = (subject_id, object_id)

            if edge_key not in edge_data:
                edge_data[edge_key] = {
                    "type": relation.type,
                    "sources": set(),
                }

            edge_data[edge_key]["sources"].add(chunk_id)

    # Перенос собранных узлов в граф
    for node_id, sources in node_sources.items():
        entity_type, name = node_id.split(":", maxsplit=1)

        graph.add_node(
            node_id,
            type=entity_type,
            name=name,
            sources=sorted(sources),
        )

    # Перенос рёбер и вычисление их весов
    for (subject_id, object_id), data in edge_data.items():
        sources = data["sources"]

        graph.add_edge(
            subject_id,
            object_id,
            type=data["type"],
            weight=len(sources),
            sources=sorted(sources),
        )

    # Разбор встречных отношений REPLACED_BY
    conflicts = 0
    processed_pairs: set[frozenset[str]] = set()

    for subject_id, object_id, data in list(graph.edges(data=True)):
        if data["type"] != "REPLACED_BY":
            continue

        pair = frozenset((subject_id, object_id))

        if pair in processed_pairs:
            continue

        processed_pairs.add(pair)

        # Самоссылка не образует конфликтующей пары двух разных деталей.
        if subject_id == object_id:
            continue

        if not graph.has_edge(object_id, subject_id):
            continue

        reverse_data = graph.edges[object_id, subject_id]

        if reverse_data["type"] != "REPLACED_BY":
            continue

        conflicts += 1

        if data["weight"] > reverse_data["weight"]:
            graph.remove_edge(object_id, subject_id)
        elif data["weight"] < reverse_data["weight"]:
            graph.remove_edge(subject_id, object_id)
        else:
            graph.remove_edge(subject_id, object_id)
            graph.remove_edge(object_id, subject_id)

    graph.graph["conflicts"] = conflicts

    return graph