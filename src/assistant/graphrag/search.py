"""Два режима поиска по графу. Требования FR-4, FR-5 и FR-10."""

import json
from itertools import combinations

import networkx as nx

from assistant.infra.grounding import code_type, find_codes, find_names
from assistant.infra.llm import LLMClient
from assistant.infra.schemas import MAP_SCHEMA, Answer, KnowledgeBase

MAX_FACTS_LENGTH = 8000
RELATION_PRIORITY = {
    "INDICATES": 0,
    "FAILED": 0,
    "HAS_PART": 0,
    "IS_A": 0,
    "REPLACED_BY": 1,
    "SUPPLIED_BY": 2,
    "FIXED_BY": 2,
    "BELONGS_TO": 3,
}


def _refused_answer() -> Answer:
    """Ответ при отсутствии подтверждающих данных."""
    return Answer(text="В базе знаний не найдено.", refused=True)


def _is_refusal(text: str) -> bool:
    """Проверка явного отказа модели в начале ответа."""
    normalized = " ".join(text.strip().lower().split())
    markers = (
        "в базе знаний не найдено",
        "в предоставленных фактах нет информации",
        "в предоставленных фактах недостаточно",
        "в перечисленных фактах нет информации",
        "факты не содержат информации",
        "фактов недостаточно",
        "недостаточно данных",
        "не могу определить",
        "невозможно определить",
        "не удаётся определить",
        "не удалось определить",
        "в сводке не указано",
        "в сводке нет информации",
        "в предоставленных данных нет информации",
        "не упоминается в сводке",
        "в сводке отсутствует информация",
        "в предоставленной сводке отсутствует",
        "в сводке не содержится",
        "данных недостаточно",
        "нет достаточных данных",
        "невозможно установить",
    )
    return any(normalized.startswith(marker) for marker in markers)


def _document_sources(sources: list[str] | None) -> set[str]:
    """Преобразование идентификаторов чанков в идентификаторы документов."""
    return {
        source.split("#", maxsplit=1)[0]
        for source in sources or []
        if isinstance(source, str) and source
    }


def _node_type(graph: nx.DiGraph, node: str) -> str:
    """Получение типа сущности из атрибутов узла или его идентификатора."""
    node_type = graph.nodes[node].get("type")
    if node_type:
        return str(node_type)
    return node.partition(":")[0]


def _node_name(graph: nx.DiGraph, node: str) -> str:
    """Получение отображаемого имени сущности."""
    name = graph.nodes[node].get("name")
    if name:
        return str(name)
    _, separator, fallback_name = node.partition(":")
    return fallback_name if separator else node


def _has_relation(graph: nx.DiGraph, source: str, target: str, relation_type: str) -> bool:
    return graph.has_edge(source, target) and graph.edges[source, target].get("type") == relation_type


def _collect_path_edges(
    graph: nx.DiGraph,
    anchors: list[str],
    undirected_graph: nx.Graph,
) -> tuple[set[str], set[tuple[str, str]]]:
    """Сбор всех направленных рёбер на простых путях длиной до трёх связей."""
    path_nodes: set[str] = set()
    selected_edges: set[tuple[str, str]] = set()

    for source, target in combinations(sorted(anchors), 2):
        for path in nx.all_simple_paths(undirected_graph, source, target, cutoff=3):
            path_nodes.update(path)
            for first, second in zip(path, path[1:], strict=False):
                if graph.has_edge(first, second):
                    selected_edges.add((first, second))
                if graph.has_edge(second, first):
                    selected_edges.add((second, first))

    return path_nodes, selected_edges

def _collect_replacement_and_supplier_edges(
    graph: nx.DiGraph,
    starting_nodes: set[str],
    selected_edges: set[tuple[str, str]],
) -> None:
    """Сбор цепочек замен, типов деталей и поставщиков конечных деталей."""
    for start_node in sorted(starting_nodes):
        if start_node not in graph or _node_type(graph, start_node) != "Part":
            continue

        reachable_parts: set[str] = {start_node}
        stack = [start_node]

        while stack:
            current = stack.pop()
            replacement_targets = [
                target
                for target in graph.successors(current)
                if _has_relation(graph, current, target, "REPLACED_BY")
            ]

            for target in sorted(replacement_targets):
                selected_edges.add((current, target))
                if target not in reachable_parts:
                    reachable_parts.add(target)
                    stack.append(target)

        # Добавление типов всех деталей в цепочке замен.
        for part in sorted(reachable_parts):
            for component in graph.successors(part):
                if _has_relation(graph, part, component, "IS_A"):
                    selected_edges.add((part, component))

        # Поставщик добавляется для каждой конечной ветви цепочки.
        terminal_parts = [
            part
            for part in reachable_parts
            if not any(
                _has_relation(graph, part, target, "REPLACED_BY")
                for target in graph.successors(part)
            )
        ]

        for terminal in sorted(terminal_parts):
            for supplier in graph.successors(terminal):
                if _has_relation(graph, terminal, supplier, "SUPPLIED_BY"):
                    selected_edges.add((terminal, supplier))


def _collect_local_edges(
    graph: nx.DiGraph,
    anchors: list[str],
) -> set[tuple[str, str]]:
    """Сбор фактов на путях между якорями или в двухшаговой окрестности."""
    undirected_graph = graph.to_undirected()

    if len(anchors) == 1:
        # Получение полной окрестности опорной сущности на глубину до двух связей.
        distances = nx.single_source_shortest_path_length(
            undirected_graph,
            anchors[0],
            cutoff=2,
        )
        neighborhood = set(distances)
        selected_edges = {
            (source, target)
            for source, target in graph.edges
            if source in neighborhood and target in neighborhood
        }
        context_nodes = neighborhood
    else:
        # Получение всех путей длиной до трёх связей между парами опорных сущностей.
        context_nodes, selected_edges = _collect_path_edges(
            graph,
            anchors,
            undirected_graph,
        )
        if not context_nodes:
            return set()

    # Расширение фактов цепочками замен и сведениями о поставщиках конечных деталей.
    _collect_replacement_and_supplier_edges(graph, context_nodes, selected_edges)
    return selected_edges

def _replacement_chain_context(
    graph: nx.DiGraph,
    selected_edges: set[tuple[str, str]],
) -> list[str]:
    """Формирование явного описания цепочек замен и их конечных деталей."""
    adjacency: dict[str, list[str]] = {}
    incoming: set[str] = set()

    for source, target in selected_edges:
        if _has_relation(graph, source, target, "REPLACED_BY"):
            adjacency.setdefault(source, []).append(target)
            incoming.add(target)

    if not adjacency:
        return []

    roots = sorted(set(adjacency) - incoming)
    if not roots:
        roots = sorted(adjacency)

    chains: set[tuple[str, ...]] = set()

    def walk(current: str, path: tuple[str, ...]) -> None:
        children = sorted(adjacency.get(current, []))
        valid_children = [child for child in children if child not in path]
        if not valid_children:
            if len(path) > 1:
                chains.add(path)
            return
        for child in valid_children:
            walk(child, (*path, child))

    for root in roots:
        walk(root, (root,))

    return [
        f"{' -> '.join(_node_name(graph, node) for node in chain)}; "
        f"конечная деталь: {_node_name(graph, chain[-1])}"
        for chain in sorted(chains)
    ]


def _format_local_facts(
    graph: nx.DiGraph,
    selected_edges: set[tuple[str, str]],
) -> tuple[str, list[str], list[str]]:
    """Формирование пронумерованных фактов с ограничением в 8000 символов."""
    lines: list[str] = []
    facts: list[str] = []
    sources: set[str] = set()
    current_length = 0

    def edge_order(edge: tuple[str, str]) -> tuple[int, str, str]:
        source, target = edge
        relation_type = str(graph.edges[source, target].get("type", "UNKNOWN"))
        return RELATION_PRIORITY.get(relation_type, 10), source, target

    for source, target in sorted(selected_edges, key=edge_order):
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
            continue

        lines.append(line)
        facts.append(fact)
        sources.update(_document_sources(attributes.get("sources")))
        current_length += added_length

    return "\n".join(lines), facts, sorted(sources)


def local_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на вопрос о конкретных сущностях по фактам из графа."""
    graph = kb.graph
    anchors: set[str] = set()

    # Поиск кодов и проверка их наличия в графе.
    for code in find_codes(question):
        node_id = f"{code_type(code)}:{code}"
        if node_id not in graph:
            return _refused_answer()
        anchors.add(node_id)

    # Поиск названий из закрытого словаря.
    for entity_type, name in find_names(question, kb.vocabulary):
        node_id = f"{entity_type}:{name}"
        if node_id in graph:
            anchors.add(node_id)

    if not anchors:
        return _refused_answer()

    selected_edges = _collect_local_edges(graph, sorted(anchors))
    facts_text, facts, sources = _format_local_facts(graph, selected_edges)
    if not facts:
        return _refused_answer()

    replacement_chains = _replacement_chain_context(graph, selected_edges)
    if replacement_chains:
        chain_text = "\n".join(f"- {chain}" for chain in replacement_chains)
        chain_section = f"\n\nЦепочки замен, рассчитанные по графу:\n{chain_text}"
        if len(facts_text) + len(chain_section) <= MAX_FACTS_LENGTH:
            facts_text += chain_section

    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты отвечаешь по фактам из графа знаний. Используй только переданные факты. "
                    "Стрелка в записи A -RELATION-> B всегда означает отношение от A к B; "
                    "никогда не меняй направление связи при пересказе. "
                    "Component — название узла оборудования, Part — конкретная деталь с кодом. "
                    "Если спрашивают, какую деталь установить или заказать, называй код Part, "
                    "а не только название Component. "
                    "Для REPLACED_BY следуй по всей цепочке от старой детали к новой. "
                    "Если цепочка A -> B -> C, актуальная деталь — C, если у неё нет дальнейшей "
                    "замены. Если есть несколько конечных ветвей, укажи неоднозначность, "
                    "не выбирай одну произвольно. Для поставщика используй SUPPLIED_BY "
                    "от конечной детали цепочки. Не путай ErrorCode с Component. "
                    "Если подтверждающих фактов недостаточно, прямо сообщи об этом. "
                    "Отвечай кратко на русском языке."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\nФакты из графа:\n{facts_text}\n\n"
                    "Сформулируй ответ только на основании этих фактов."
                ),
            },
        ],
        max_tokens=1024,
    )

    answer_text = response.content.strip()
    if not answer_text or _is_refusal(answer_text):
        return _refused_answer()

    return Answer(
        text=answer_text,
        sources=sources,
        facts=facts,
        refused=False,
    )


def _community_sources(kb: KnowledgeBase, community_nodes: list[str]) -> set[str]:
    """Получение документов-источников, относящихся к сообществу."""
    graph = kb.graph
    nodes = {node for node in community_nodes if node in graph}
    sources: set[str] = set()

    for node in nodes:
        sources.update(_document_sources(graph.nodes[node].get("sources")))

    for _, _, attributes in graph.subgraph(nodes).edges(data=True):
        sources.update(_document_sources(attributes.get("sources")))

    # Сбор источников для статистики продуктовой линейки, рассчитанной по всему графу.
    for product_line in sorted(node for node in nodes if _node_type(graph, node) == "ProductLine"):
        models = [
            model
            for model in graph.predecessors(product_line)
            if _has_relation(graph, model, product_line, "BELONGS_TO")
        ]
        for model in models:
            sources.update(_document_sources(graph.nodes[model].get("sources")))
            for target in graph.successors(model):
                attributes = graph.edges[model, target]
                relation_type = attributes.get("type")
                if relation_type in {"FAILED", "HAS_PART", "BELONGS_TO"}:
                    sources.update(_document_sources(attributes.get("sources")))
                if relation_type != "HAS_PART" or _node_type(graph, target) != "Part":
                    continue

                # Сбор источников компонентов, цепочек замен и поставщиков установленных деталей.
                parts = {target}
                stack = [target]
                while stack:
                    part = stack.pop()
                    for component in graph.successors(part):
                        edge = graph.edges[part, component]
                        if edge.get("type") == "IS_A":
                            sources.update(_document_sources(edge.get("sources")))
                    for replacement in graph.successors(part):
                        edge = graph.edges[part, replacement]
                        if edge.get("type") == "REPLACED_BY":
                            sources.update(_document_sources(edge.get("sources")))
                            if replacement not in parts:
                                parts.add(replacement)
                                stack.append(replacement)
                for part in parts:
                    for supplier in graph.successors(part):
                        edge = graph.edges[part, supplier]
                        if edge.get("type") == "SUPPLIED_BY":
                            sources.update(_document_sources(edge.get("sources")))

    return sources


def _parse_map_response(raw: str) -> dict | None:
    """Разбор JSON-ответа Map с поддержкой JSON в Markdown-блоке."""
    content = raw.strip()
    if content.startswith("```"):
        content = content.strip("`").strip()
        if content.lower().startswith("json"):
            content = content[4:].strip()
    try:
        parsed = json.loads(content)
    except (json.JSONDecodeError, TypeError):
        return None
    return parsed if isinstance(parsed, dict) else None


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос по сводкам сообществ в два шага."""
    useful_answers: list[str] = []
    sources: set[str] = set()

    # Шаг map: получение частичных ответов по всем непустым сводкам.
    for community in kb.communities:
        summary = community.summary.strip()
        if not summary:
            continue

        response = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Отвечай только по переданной сводке. Верни JSON с полями answer и score. "
                        "score — целое число от 0 до 10. Поставь 0, если сводка не содержит "
                        "конкретных фактов для ответа на вопрос. Положительную оценку ставь только "
                        "если в сводке есть фактическая опора для ответа. Не используй общие знания."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nСводка сообщества:\n{summary}",
                },
            ],
            json_schema=MAP_SCHEMA,
            max_tokens=512,
        )

        partial = _parse_map_response(response.content)
        if partial is None:
            continue

        partial_text = partial.get("answer")
        score = partial.get("score")
        if not isinstance(partial_text, str) or not partial_text.strip():
            continue
        if isinstance(score, bool) or not isinstance(score, int) or not 1 <= score <= 10:
            continue
        if _is_refusal(partial_text):
            continue

        useful_answers.append(partial_text.strip())
        sources.update(_community_sources(kb, community.nodes))

    if not useful_answers:
        return _refused_answer()

    partial_answers_text = "\n".join(
        f"{index}. {answer}"
        for index, answer in enumerate(useful_answers, start=1)
    )

    # Шаг reduce: объединение только полезных частичных ответов.
    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Объедини частичные ответы в один точный ответ на русском языке. "
                    "Используй только факты из частичных ответов. Не добавляй неподтверждённые "
                    "коды, количества, частоты или причинно-следственные связи. "
                    "Если ответы противоречат друг другу или не позволяют однозначно ответить, "
                    "укажи, что данных недостаточно, вместо догадки."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Вопрос: {question}\n\nПолезные частичные ответы:\n"
                    f"{partial_answers_text}\n\nОбъедини их в итоговый ответ."
                ),
            },
        ],
        max_tokens=1024,
    )

    answer_text = response.content.strip()
    if not answer_text or _is_refusal(answer_text):
        return _refused_answer()

    return Answer(
        text=answer_text,
        sources=sorted(sources),
        refused=False,
    )