"""Сообщества графа и их сводки. Требование FR-3."""

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным `seed`.

    Каждый узел входит ровно в одно сообщество. Поле `summary` здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []

    # Создание неориентированной проекции с сохранением узлов.
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(sorted(graph.nodes))

    # Суммирование весов рёбер, направленных между одной парой узлов.
    edges = sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1]))

    for source, target, attributes in edges:
        weight = attributes.get("weight", 1)

        if undirected_graph.has_edge(source, target):
            undirected_graph[source][target]["weight"] += weight
        else:
            undirected_graph.add_edge(source, target, weight=weight)

    # Разбиение графа алгоритмом Лувена.
    detected = nx.community.louvain_communities(
        undirected_graph,
        weight="weight",
        seed=LOUVAIN_SEED,
    )

    # Сортировка узлов и сообществ для воспроизводимой нумерации.
    sorted_communities = sorted(
        (sorted(nodes) for nodes in detected if nodes),
        key=lambda nodes: nodes[0],
    )

    return [
        Community(community_id=index, nodes=nodes)
        for index, nodes in enumerate(sorted_communities)
    ]

def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов: какие сущности входят, как связаны, что встречается чаще.

    Все числа считает код по весам рёбер и передаёт модели готовыми. Модель их только формулирует.
    """
    # Отбор узлов сообщества, присутствующих в графе.
    nodes = sorted({node for node in community.nodes if node in graph})
    node_set = set(nodes)

    node_labels: dict[str, str] = {}
    entity_type_counts: Counter[str] = Counter()

    for node in nodes:
        attributes = graph.nodes[node]
        fallback_type, separator, fallback_name = node.partition(":")

        entity_type = attributes.get("type") or fallback_type or "Unknown"
        entity_name = attributes.get("name") or (fallback_name if separator else node)

        node_labels[node] = f"{entity_type}:{entity_name}"
        entity_type_counts[entity_type] += 1

    # Отбор внутренних связей и подготовка статистики по их весам.
    internal_edges = sorted(
        graph.subgraph(node_set).edges(data=True),
        key=lambda edge: (edge[0], edge[1], str(edge[2].get("type", ""))),
    )

    relation_weight_counts: Counter[str] = Counter()
    incident_weight_counts: Counter[str] = Counter()
    edge_lines: list[str] = []
    total_weight = 0

    for source, target, attributes in internal_edges:
        relation_type = str(attributes.get("type") or "UNKNOWN")
        weight = attributes.get("weight", 1)

        relation_weight_counts[relation_type] += weight
        incident_weight_counts[source] += weight

        if source != target:
            incident_weight_counts[target] += weight

        total_weight += weight

        edge_lines.append(
            f"- {node_labels[source]} -{relation_type}-> {node_labels[target]} "
            f"(вес: {weight})"
        )

    # Подсчёт наиболее связанных сущностей и распределения отношений.
    entity_type_summary = "; ".join(
        f"{entity_type}: {count}"
        for entity_type, count in sorted(entity_type_counts.items())
    ) or "нет"

    relation_summary = "; ".join(
        f"{relation_type}: {weight}"
        for relation_type, weight in sorted(
            relation_weight_counts.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ) or "нет"

    most_connected_entities = sorted(
        incident_weight_counts.items(),
        key=lambda item: (-item[1], node_labels[item[0]]),
    )[:10]

    entity_weight_summary = "; ".join(
        f"{node_labels[node]}: {weight}"
        for node, weight in most_connected_entities
    ) or "нет"

    # Формирование фактов для модели.
    node_lines = [f"- {node_labels[node]}" for node in nodes]

    facts = "\n".join(
        [
            f"Идентификатор сообщества: {community.community_id}",
            f"Количество сущностей: {len(nodes)}",
            f"Количество сущностей по типам: {entity_type_summary}",
            f"Количество внутренних связей: {len(internal_edges)}",
            f"Сумма весов внутренних связей: {total_weight}",
            f"Сумма весов по типам отношений: {relation_summary}",
            f"Наиболее связанные сущности по сумме весов инцидентных рёбер: {entity_weight_summary}",
            "",
            "Сущности сообщества:",
            *node_lines,
            "",
            "Внутренние связи (вес — число чанков, в которых найдена связь):",
            *(edge_lines or ["- нет"]),
        ]
    )

    # Один вызов модели для генерации сводки.
    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты составляешь краткие сводки сообществ графа знаний. "
                    "Напиши связную сводку на русском языке объёмом не более 150 слов. "
                    "Опиши состав сообщества, основные связи и наиболее часто встречающиеся "
                    "сущности или типы отношений. Используй только предоставленные факты. "
                    "Все числа уже рассчитаны программно: не пересчитывай их, не изменяй "
                    "и не добавляй новые. Не придумывай отсутствующие связи и свойства."
                ),
            },
            {
                "role": "user",
                "content": f"Факты о сообществе:\n{facts}",
            },
        ],
        max_tokens=300,
    )

    # Ограничение длины ответа независимо от соблюдения инструкции моделью.
    words = response.content.split()
    return " ".join(words[:150])