"""Сообщества графа и их сводки. Требование FR-3."""

from collections import Counter, defaultdict

import networkx as nx

from assistant.infra.llm import LLMClient
from assistant.infra.schemas import Community

LOUVAIN_SEED = 7


def _node_type(graph: nx.DiGraph, node: str) -> str:
    node_type = graph.nodes[node].get("type")
    return str(node_type) if node_type else node.partition(":")[0]


def _node_name(graph: nx.DiGraph, node: str) -> str:
    name = graph.nodes[node].get("name")
    if name:
        return str(name)
    _, separator, fallback = node.partition(":")
    return fallback if separator else node


def _has_relation(graph: nx.DiGraph, source: str, target: str, relation_type: str) -> bool:
    return graph.has_edge(source, target) and graph.edges[source, target].get("type") == relation_type


def detect_communities(graph: nx.DiGraph) -> list[Community]:
    """Разбивает узлы на сообщества алгоритмом Лувена с фиксированным seed.

    Каждый узел входит ровно в одно сообщество. Поле summary здесь остаётся пустым.
    """
    if graph.number_of_nodes() == 0:
        return []

    # Создание неориентированной проекции с сохранением всех узлов.
    undirected_graph = nx.Graph()
    undirected_graph.add_nodes_from(sorted(graph.nodes))

    # Суммирование весов встречных направленных рёбер.
    edges = sorted(graph.edges(data=True), key=lambda edge: (edge[0], edge[1]))
    for source, target, attributes in edges:
        weight = attributes.get("weight", 1)
        if undirected_graph.has_edge(source, target):
            undirected_graph[source][target]["weight"] += weight
        else:
            undirected_graph.add_edge(source, target, weight=weight)

    detected = nx.community.louvain_communities(
        undirected_graph,
        weight="weight",
        seed=LOUVAIN_SEED,
    )

    # Стабильная нумерация сообществ.
    sorted_communities = sorted(
        (sorted(nodes) for nodes in detected if nodes),
        key=lambda nodes: nodes[0],
    )
    return [
        Community(community_id=index, nodes=nodes)
        for index, nodes in enumerate(sorted_communities)
    ]


def _replacement_paths(graph: nx.DiGraph, start: str) -> list[list[str]]:
    """Поиск всех конечных цепочек замен от исходной детали без повторения узлов."""
    paths: list[list[str]] = []

    def visit(current: str, path: list[str]) -> None:
        next_parts = sorted(
            target
            for target in graph.successors(current)
            if _has_relation(graph, current, target, "REPLACED_BY") and target not in path
        )
        if not next_parts:
            if len(path) > 1:
                paths.append(path)
            return
        for next_part in next_parts:
            visit(next_part, [*path, next_part])

    visit(start, [start])
    return paths


def _product_line_statistics(graph: nx.DiGraph, product_line: str) -> str:
    """Расчёт статистики отказов и цепочек замен по всей продуктовой линейке."""
    models = sorted(
        model
        for model in graph.predecessors(product_line)
        if _has_relation(graph, model, product_line, "BELONGS_TO")
    )

    failed_by_component: Counter[str] = Counter()
    failed_by_model: Counter[str] = Counter()
    installed_parts: dict[str, set[str]] = defaultdict(set)

    for model in models:
        model_failures = 0
        for component in graph.successors(model):
            if not _has_relation(graph, model, component, "FAILED"):
                continue
            if _node_type(graph, component) != "Component":
                continue
            weight = int(graph.edges[model, component].get("weight", 1))
            failed_by_component[_node_name(graph, component)] += weight
            model_failures += weight
        failed_by_model[_node_name(graph, model)] = model_failures

        for part in graph.successors(model):
            if _has_relation(graph, model, part, "HAS_PART") and _node_type(graph, part) == "Part":
                installed_parts[part].add(model)

    component_stats = "; ".join(
        f"{name}: {count}"
        for name, count in sorted(failed_by_component.items(), key=lambda item: (-item[1], item[0]))
    ) or "нет данных"
    model_stats = "; ".join(
        f"{name}: {count}"
        for name, count in sorted(failed_by_model.items(), key=lambda item: (-item[1], item[0]))
    ) or "нет данных"

    # Сбор цепочек замен только для деталей, действительно установленных в моделях линейки.
    replacement_lines: list[str] = []
    for initial_part, part_models in sorted(installed_parts.items()):
        for path in _replacement_paths(graph, initial_part):
            terminal = path[-1]
            suppliers = sorted(
                _node_name(graph, supplier)
                for supplier in graph.successors(terminal)
                if _has_relation(graph, terminal, supplier, "SUPPLIED_BY")
            )
            supplier_text = ", ".join(suppliers) if suppliers else "поставщик не указан"
            model_text = ", ".join(sorted(_node_name(graph, model) for model in part_models))
            chain_text = " -> ".join(_node_name(graph, part) for part in path)
            replacement_lines.append(
                f"{chain_text} (модели: {model_text}; конечный поставщик: {supplier_text})"
            )

    replacements = "; ".join(replacement_lines) if replacement_lines else "не обнаружены"
    return "\n".join(
        [
            f"Продуктовая линейка: {_node_name(graph, product_line)}",
            "Количество отказов по компонентам (только рёбра FAILED, сумма weight): " + component_stats,
            "Количество отказов по моделям (сумма weight рёбер FAILED): " + model_stats,
            "Цепочки замен среди установленных деталей: " + replacements,
        ]
    )


def summarize_community(community: Community, graph: nx.DiGraph, llm: LLMClient) -> str:
    """Сводка сообщества не длиннее 150 слов с программно рассчитанными числами."""
    nodes = sorted({node for node in community.nodes if node in graph})
    node_set = set(nodes)

    node_labels: dict[str, str] = {}
    entity_type_counts: Counter[str] = Counter()
    for node in nodes:
        entity_type = _node_type(graph, node)
        node_labels[node] = f"{entity_type}:{_node_name(graph, node)}"
        entity_type_counts[entity_type] += 1

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
        weight = int(attributes.get("weight", 1))
        relation_weight_counts[relation_type] += weight
        incident_weight_counts[source] += weight
        if source != target:
            incident_weight_counts[target] += weight
        total_weight += weight
        edge_lines.append(
            f"- {node_labels[source]} -{relation_type}-> {node_labels[target]} (вес: {weight})"
        )

    entity_type_summary = "; ".join(
        f"{entity_type}: {count}" for entity_type, count in sorted(entity_type_counts.items())
    ) or "нет"
    relation_summary = "; ".join(
        f"{relation_type}: {weight}"
        for relation_type, weight in sorted(
            relation_weight_counts.items(), key=lambda item: (-item[1], item[0])
        )
    ) or "нет"
    most_connected = sorted(
        incident_weight_counts.items(),
        key=lambda item: (-item[1], node_labels[item[0]]),
    )[:10]
    entity_weight_summary = "; ".join(
        f"{node_labels[node]}: {weight}" for node, weight in most_connected
    ) or "нет"

    # Добавление статистики всей линейки, включая рёбра за границами сообщества.
    product_line_stats = [
        _product_line_statistics(graph, node)
        for node in nodes
        if _node_type(graph, node) == "ProductLine"
    ]

    facts_parts = [
        f"Идентификатор сообщества: {community.community_id}",
        f"Количество сущностей: {len(nodes)}",
        f"Количество сущностей по типам: {entity_type_summary}",
        f"Количество внутренних связей: {len(internal_edges)}",
        f"Сумма весов внутренних связей: {total_weight}",
        f"Сумма весов по типам отношений: {relation_summary}",
        f"Наиболее связанные сущности по всем типам отношений (не считать частотой отказов): {entity_weight_summary}",
    ]
    if product_line_stats:
        facts_parts.extend(["", "Дополнительная статистика продуктовой линейки по всему графу:", *product_line_stats])

    facts_parts.extend(
        [
            "",
            "Сущности сообщества:",
            *(f"- {node_labels[node]}" for node in nodes),
            "",
            "Внутренние связи (вес — число чанков, в которых найдена связь):",
            *(edge_lines or ["- нет"]),
        ]
    )
    facts = "\n".join(facts_parts)

    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты составляешь краткие сводки сообществ графа знаний на русском языке, "
                    "не более 150 слов. Используй только переданные факты и не придумывай числа. "
                    "Для вопросов о частоте отказов учитывай исключительно статистику рёбер FAILED "
                    "и сравнивай только сущности Component или модели Model. Никогда не считай "
                    "ErrorCode узлом оборудования и не используй общую связанность узла как частоту "
                    "отказов. Для вопросов о деталях указывай полные цепочки REPLACED_BY и конечные "
                    "детали. Дополнительная статистика продуктовой линейки рассчитана по всему графу, "
                    "включая связи между разными сообществами. Не пересчитывай и не изменяй числа."
                ),
            },
            {"role": "user", "content": f"Факты о сообществе:\n{facts}"},
        ],
        max_tokens=512,
    )

    return " ".join(response.content.split()[:150])