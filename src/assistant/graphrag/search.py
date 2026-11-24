"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

MAX_FACTS_LENGTH = 8000


def _refused_answer() -> Answer:
    """Ответ при отсутствии подтверждающих данных."""
    return Answer(text="В базе знаний не найдено.", refused=True)


def _document_sources(sources: list[str] | None) -> set[str]:
    """Преобразование идентификаторов чанков в идентификаторы документов."""
    return {
        source.split("#", maxsplit=1)[0]
        for source in sources or []
        if isinstance(source, str) and source
    }


def _node_name(graph: nx.DiGraph, node: str) -> str:
    """Получение отображаемого имени сущности."""
    name = graph.nodes[node].get("name")
    if name:
        return str(name)

    _, separator, fallback_name = node.partition(":")
    return fallback_name if separator else node


def _collect_path_edges(
    graph: nx.DiGraph,
    anchors: list[str],
    undirected_graph: nx.Graph,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Сбор узлов и направленных рёбер всех путей между опорными сущностями."""
    path_nodes: set[str] = set()
    selected_edges: set[tuple[str, str]] = set()

    for source, target in combinations(sorted(anchors), 2):
        paths = nx.all_simple_paths(undirected_graph, source, target, cutoff=3)

        for path in paths:
            path_nodes.update(path)

            for first, second in zip(path, path[1:], strict=False):
                if graph.has_edge(first, second):
                    selected_edges.add((first, second))

                if graph.has_edge(second, first):
                    selected_edges.add((second, first))

    return path_nodes, selected_edges


def _collect_replacement_edges(
    graph: nx.DiGraph,
    starting_nodes: set[str],
    selected_edges: set[tuple[str, str]],
) -> None:
    """Добавление полных цепочек REPLACED_BY от деталей на найденных путях."""
    for start_node in sorted(starting_nodes):
        if graph.nodes[start_node].get("type") != "Part":
            continue

        stack: list[tuple[str, set[str]]] = [(start_node, {start_node})]

        while stack:
            current, visited = stack.pop()

            for next_node in sorted(graph.successors(current)):
                attributes = graph.edges[current, next_node]

                if attributes.get("type") != "REPLACED_BY":
                    continue

                selected_edges.add((current, next_node))

                if next_node not in visited:
                    stack.append((next_node, visited | {next_node}))


def _collect_local_edges(
    graph: nx.DiGraph,
    anchors: list[str],
) -> set[tuple[str, str]]:
    """Сбор фактов по путям между сущностями или по окрестности одной сущности."""
    undirected_graph = graph.to_undirected()

    if len(anchors) == 1:
        # Получение окрестности опорной сущности на глубину двух связей.
        distances = nx.single_source_shortest_path_length(
            undirected_graph,
            anchors[0],
            cutoff=2,
        )
        neighborhood = set(distances)

        return {
            (source, target)
            for source, target in graph.edges
            if source in neighborhood and target in neighborhood
        }

    # Получение всех путей длиной до трёх связей между парами опорных сущностей.
    path_nodes, selected_edges = _collect_path_edges(
        graph,
        anchors,
        undirected_graph,
    )

    # Добавление цепочек замен для деталей, входящих в найденные пути.
    _collect_replacement_edges(graph, path_nodes, selected_edges)

    return selected_edges


def _format_local_facts(
    graph: nx.DiGraph,
    selected_edges: set[tuple[str, str]],
) -> tuple[str, list[str], list[str]]:
    """Формирование нумерованного списка фактов с ограничением в 8000 символов."""
    lines: list[str] = []
    facts: list[str] = []
    sources: set[str] = set()
    current_length = 0

    for source, target in sorted(selected_edges):
        attributes = graph.edges[source, target]
        relation_type = attributes.get("type", "UNKNOWN")

        fact = (
            f"{_node_name(graph, source)} "
            f"-{relation_type}-> "
            f"{_node_name(graph, target)}"
        )
        line = f"{len(facts) + 1}. {fact}"
        added_length = len(line) + (1 if lines else 0)

        if current_length + added_length > MAX_FACTS_LENGTH:
            break

        lines.append(line)
        facts.append(fact)
        sources.update(_document_sources(attributes.get("sources")))

        current_length += added_length

    return "\n".join(lines), facts, sorted(sources)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа.

    Опорные сущности ищутся без модели: коды через `find_codes`, названия через `find_names`.
    При двух и более опорных сущностях берутся факты на путях длиной до трёх связей между ними
    и цепочки `REPLACED_BY` от деталей на этих путях. При одной берётся окрестность на глубину
    до двух связей. Модель получает пронумерованный список фактов и отвечает только по нему.
    Если в вопросе есть код, которого нет в графе, или фактов нет, возвращается
    `Answer(..., refused=True)` без вызова модели.
    """
    graph = kb.graph
    anchors: set[str] = set()

    # Поиск кодов и проверка их наличия в графе.
    for code in find_codes(question):
        node_id = f"{code_type(code)}:{code}"

        if node_id not in graph:
            return _refused_answer()

        anchors.add(node_id)

    # Поиск названий из словаря и проверка наличия соответствующих узлов.
    for entity_type, name in find_names(question, kb.vocabulary):
        node_id = f"{entity_type}:{name}"

        if node_id in graph:
            anchors.add(node_id)

    if not anchors:
        return _refused_answer()

    # Сбор фактов по найденным опорным сущностям.
    selected_edges = _collect_local_edges(graph, sorted(anchors))
    facts_text, facts, sources = _format_local_facts(graph, selected_edges)

    if not facts:
        return _refused_answer()

    # Один запрос модели с фактами, полученными исключительно из графа.
    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты отвечаешь на вопросы по базе знаний сервисных документов. "
                    "Используй исключительно приведённые факты и не добавляй сведения "
                    "из собственных знаний. Если среди фактов есть цепочка REPLACED_BY, "
                    "обязательно назови последнюю, актуальную деталь этой цепочки. "
                    "Не путай старую деталь с её заменой. Отвечай кратко на русском языке. "
                    "Если фактов недостаточно для ответа, прямо сообщи об этом."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    f"Факты из графа:\n{facts_text}\n\n"
                    "Ответь на вопрос, опираясь только на перечисленные факты."
                ),
            },
        ],
        max_tokens=1024,
    )

    answer_text = response.content.strip()

    if not answer_text:
        return _refused_answer()

    return Answer(
        text=answer_text,
        sources=sources,
        facts=facts,
    )


def _community_sources(kb: KnowledgeBase, community_nodes: list[str]) -> set[str]:
    """Получение документов-источников, относящихся к сообществу."""
    graph = kb.graph
    nodes = {node for node in community_nodes if node in graph}
    sources: set[str] = set()

    # Сбор источников узлов сообщества.
    for node in nodes:
        sources.update(_document_sources(graph.nodes[node].get("sources")))

    # Сбор источников внутренних связей сообщества.
    for _, _, attributes in graph.subgraph(nodes).edges(data=True):
        sources.update(_document_sources(attributes.get("sources")))

    return sources


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага.

    Шаг map: для каждой сводки модель даёт частичный ответ и оценку полезности от 0 до 10.
    Шаг reduce: частичные ответы с оценкой выше 0 объединяются в итоговый ответ.
    Если таких ответов нет, возвращается `Answer(..., refused=True)`.
    """
    useful_answers: list[str] = []
    sources: set[str] = set()

    # Шаг map: получение частичного ответа по каждой непустой сводке.
    for community in kb.communities:
        summary = community.summary.strip()

        if not summary:
            continue

        response = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Ты отвечаешь на вопрос только по одной сводке сообщества графа знаний. "
                        "Верни JSON с полями answer и score. В answer укажи краткий ответ "
                        "по сводке. В score поставь целое число от 0 до 10: 0, если сводка "
                        "не содержит полезной информации для ответа, и больше 0, если содержит."
                    ),
                },
                {
                    "role": "user",
                    "content": (
                        f"Вопрос: {question}\n\n"
                        f"Сводка сообщества:\n{summary}"
                    ),
                },
            ],
            json_schema=MAP_SCHEMA,
            max_tokens=512,
        )

        # Проверка JSON и полей частичного ответа.
        try:
            partial = json.loads(response.content)
        except (json.JSONDecodeError, TypeError):
            continue

        if not isinstance(partial, dict):
            continue

        partial_text = partial.get("answer")
        score = partial.get("score")

        if not isinstance(partial_text, str) or not partial_text.strip():
            continue

        if isinstance(score, bool) or not isinstance(score, int) or not 0 <= score <= 10:
            continue

        if score == 0:
            continue

        useful_answers.append(partial_text.strip())
        sources.update(_community_sources(kb, community.nodes))

    if not useful_answers:
        return _refused_answer()

    # Шаг reduce: объединение только полезных частичных ответов.
    partial_answers_text = "\n".join(
        f"{index}. {answer}"
        for index, answer in enumerate(useful_answers, start=1)
    )

    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты объединяешь частичные ответы по сообществам графа знаний. "
                    "Составь один связный и краткий ответ на русском языке. "
                    "Используй только приведённые частичные ответы, не добавляй "
                    "неподтверждённые сведения и не делай выводов, противоречащих им."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\n"
                    f"Полезные частичные ответы:\n{partial_answers_text}\n\n"
                    "Объедини их в итоговый ответ."
                ),
            },
        ],
        max_tokens=1024,
    )

    answer_text = response.content.strip()

    if not answer_text:
        return _refused_answer()

    return Answer(
        text=answer_text,
        sources=sorted(sources),
    )