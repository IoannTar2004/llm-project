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
    "FIXED_BY": 0,
    "REPLACED_BY": 1,
    "SUPPLIED_BY": 2,
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


def _needs_part_details(question: str) -> bool:
    """Проверка необходимости раскрытия состава деталей и цепочек замен."""
    lowered = question.lower()
    return any(
        marker in lowered
        for marker in (
            "детал",
            "заказ",
            "поставщик",
            "поставляет",
            "замен",
            "снят",
            "производств",
            "артикул",
        )
    )


def _needs_replacement_details(question: str) -> bool:
    """Проверка необходимости поиска снятых и заменённых деталей."""
    lowered = question.lower()
    return any(
        marker in lowered
        for marker in ("замен", "снят", "производств", "устарев", "выведен")
    )


def _add_edge(
    graph: nx.DiGraph,
    selected_edges: set[tuple[str, str]],
    source: str,
    target: str,
    relation_types: set[str] | None = None,
) -> None:
    """Добавление существующего ребра с необязательной проверкой типа."""
    if not graph.has_edge(source, target):
        return
    relation_type = str(graph.edges[source, target].get("type", "UNKNOWN"))
    if relation_types is None or relation_type in relation_types:
        selected_edges.add((source, target))


def _collect_component_context(
    graph: nx.DiGraph,
    component: str,
    selected_edges: set[tuple[str, str]],
    starting_parts: set[str],
    question: str,
) -> None:
    """Сбор ошибок, регламентов и при необходимости деталей узла."""
    need_parts = _needs_part_details(question)
    need_models = "модел" in question.lower() or any(
        marker in question.lower() for marker in ("какой аппарат", "какие аппараты")
    )

    for source in sorted(graph.predecessors(component)):
        relation_type = graph.edges[source, component].get("type")
        if relation_type == "INDICATES":
            selected_edges.add((source, component))
            for procedure in sorted(graph.successors(source)):
                _add_edge(graph, selected_edges, source, procedure, {"FIXED_BY"})
        elif relation_type == "FAILED" and need_models:
            selected_edges.add((source, component))
        elif relation_type == "IS_A" and need_parts:
            selected_edges.add((source, component))
            starting_parts.add(source)

    if not need_parts:
        return

    # Поиск установленных деталей только в моделях, связанных с этим узлом.
    for model in sorted(graph.predecessors(component)):
        if graph.edges[model, component].get("type") != "FAILED":
            continue
        for part in sorted(graph.successors(model)):
            if graph.edges[model, part].get("type") != "HAS_PART":
                continue
            if _has_relation(graph, part, component, "IS_A"):
                selected_edges.add((model, part))
                selected_edges.add((part, component))
                starting_parts.add(part)


def _collect_single_anchor_edges(
    graph: nx.DiGraph,
    anchor: str,
    question: str,
) -> tuple[set[tuple[str, str]], set[str]]:
    """Сбор релевантных связей по типу единственного опорного узла."""
    selected_edges: set[tuple[str, str]] = set()
    starting_parts: set[str] = set()
    node_type = _node_type(graph, anchor)
    need_parts = _needs_part_details(question)

    if node_type == "Part":
        # Прямые модели установки, тип детали, замены и поставщик.
        for model in sorted(graph.predecessors(anchor)):
            _add_edge(graph, selected_edges, model, anchor, {"HAS_PART"})
        for target in sorted(graph.successors(anchor)):
            _add_edge(graph, selected_edges, anchor, target, {"IS_A", "REPLACED_BY", "SUPPLIED_BY"})
        for previous in sorted(graph.predecessors(anchor)):
            _add_edge(graph, selected_edges, previous, anchor, {"REPLACED_BY"})
            if _has_relation(graph, previous, anchor, "REPLACED_BY"):
                starting_parts.add(previous)
        starting_parts.add(anchor)

    elif node_type == "Component":
        _collect_component_context(graph, anchor, selected_edges, starting_parts, question)

    elif node_type == "ErrorCode":
        for target in sorted(graph.successors(anchor)):
            relation_type = graph.edges[anchor, target].get("type")
            if relation_type == "INDICATES":
                selected_edges.add((anchor, target))
                _collect_component_context(graph, target, selected_edges, starting_parts, question)
            elif relation_type == "FIXED_BY":
                selected_edges.add((anchor, target))

    elif node_type == "Model":
        for target in sorted(graph.successors(anchor)):
            relation_type = graph.edges[anchor, target].get("type")
            if relation_type in {"BELONGS_TO", "FAILED", "HAS_PART"}:
                selected_edges.add((anchor, target))
                if relation_type == "HAS_PART":
                    starting_parts.add(target)
                elif relation_type == "FAILED":
                    _collect_component_context(
                        graph,
                        target,
                        selected_edges,
                        starting_parts,
                        question,
                    )

    elif node_type == "ProductLine":
        # Сбор моделей линейки и их фактов об отказах.
        for model in sorted(graph.predecessors(anchor)):
            if not _has_relation(graph, model, anchor, "BELONGS_TO"):
                continue
            selected_edges.add((model, anchor))
            for target in sorted(graph.successors(model)):
                relation_type = graph.edges[model, target].get("type")
                if relation_type == "FAILED":
                    selected_edges.add((model, target))
                elif relation_type == "HAS_PART" and need_parts:
                    if _needs_replacement_details(question) and not any(
                        _has_relation(graph, target, replacement, "REPLACED_BY")
                        for replacement in graph.successors(target)
                    ):
                        continue
                    selected_edges.add((model, target))
                    starting_parts.add(target)

    elif node_type == "Procedure":
        for error_code in sorted(graph.predecessors(anchor)):
            if not _has_relation(graph, error_code, anchor, "FIXED_BY"):
                continue
            selected_edges.add((error_code, anchor))
            for target in sorted(graph.successors(error_code)):
                _add_edge(graph, selected_edges, error_code, target)

    elif node_type == "Supplier":
        for part in sorted(graph.predecessors(anchor)):
            if not _has_relation(graph, part, anchor, "SUPPLIED_BY"):
                continue
            selected_edges.add((part, anchor))
            for source in sorted(graph.predecessors(part)):
                _add_edge(graph, selected_edges, source, part)
            for target in sorted(graph.successors(part)):
                _add_edge(graph, selected_edges, part, target)

    else:
        # Резервный сбор непосредственных связей для остальных типов сущностей.
        for source, target in graph.in_edges(anchor):
            selected_edges.add((source, target))
        for source, target in graph.out_edges(anchor):
            selected_edges.add((source, target))

    return selected_edges, starting_parts


def _collect_local_edges(
    graph: nx.DiGraph,
    anchors: list[str],
    question: str = "",
) -> set[tuple[str, str]]:
    """Сбор фактов по путям между якорями или по типизированным связям одного якоря."""
    starting_parts: set[str] = set()

    # Для пары Model + Component требуется точное совпадение обеих связей:
    # Model -HAS_PART-> Part -IS_A-> Component. Общие пути между якорями
    # могут включать детали других моделей с тем же типом компонента.
    model_anchors = [node for node in anchors if _node_type(graph, node) == "Model"]
    component_anchors = [node for node in anchors if _node_type(graph, node) == "Component"]
    if len(anchors) == 2 and len(model_anchors) == 1 and len(component_anchors) == 1:
        model = model_anchors[0]
        component = component_anchors[0]
        selected_edges: set[tuple[str, str]] = set()

        if _has_relation(graph, model, component, "FAILED"):
            selected_edges.add((model, component))

        for part in sorted(graph.successors(model)):
            if (
                _has_relation(graph, model, part, "HAS_PART")
                and _has_relation(graph, part, component, "IS_A")
            ):
                selected_edges.add((model, part))
                selected_edges.add((part, component))
                starting_parts.add(part)

    elif len(anchors) == 1:
        selected_edges, starting_parts = _collect_single_anchor_edges(
            graph,
            anchors[0],
            question,
        )
    else:
        undirected_graph = graph.to_undirected()
        context_nodes, selected_edges = _collect_path_edges(
            graph,
            anchors,
            undirected_graph,
        )
        if not context_nodes:
            return set()

        # Дополнение связей между моделью и деталью через общий компонент.
        components = {node for node in context_nodes if _node_type(graph, node) == "Component"}
        for error_code in [node for node in context_nodes if _node_type(graph, node) == "ErrorCode"]:
            for component in graph.successors(error_code):
                if _has_relation(graph, error_code, component, "INDICATES"):
                    components.add(component)
                    for procedure in graph.successors(error_code):
                        _add_edge(graph, selected_edges, error_code, procedure, {"FIXED_BY"})

        if _needs_part_details(question):
            for model in [node for node in context_nodes if _node_type(graph, node) == "Model"]:
                for part in graph.successors(model):
                    if not _has_relation(graph, model, part, "HAS_PART"):
                        continue
                    matching_components = [
                        component
                        for component in components
                        if _has_relation(graph, part, component, "IS_A")
                    ]
                    for component in matching_components:
                        selected_edges.add((model, part))
                        selected_edges.add((part, component))
                        starting_parts.add(part)

        starting_parts.update(
            node for node in context_nodes if _node_type(graph, node) == "Part"
        )

    # Расширение только найденных деталей их цепочками замен и конечными поставщиками.
    _collect_replacement_and_supplier_edges(graph, starting_parts, selected_edges)
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



def _part_models_context(
    graph: nx.DiGraph,
    anchors: set[str],
    question: str,
) -> list[str]:
    """Формирование точного списка моделей, в которых установлена деталь."""
    if "модел" not in question.lower():
        return []

    lines: list[str] = []
    for part in sorted(node for node in anchors if _node_type(graph, node) == "Part"):
        models = sorted(
            _node_name(graph, model)
            for model in graph.predecessors(part)
            if _has_relation(graph, model, part, "HAS_PART") and _node_type(graph, model) == "Model"
        )
        if models:
            lines.append(
                f"Деталь {_node_name(graph, part)} установлена в моделях: {', '.join(models)}."
            )

    return lines


def _replacement_paths(graph: nx.DiGraph, start: str) -> list[list[str]]:
    """Получение всех конечных цепочек замен без повторного обхода узлов."""
    paths: list[list[str]] = []

    def walk(current: str, path: list[str]) -> None:
        children = sorted(
            target
            for target in graph.successors(current)
            if _has_relation(graph, current, target, "REPLACED_BY") and target not in path
        )
        if not children:
            if len(path) > 1:
                paths.append(path)
            return
        for child in children:
            walk(child, [*path, child])

    walk(start, [start])
    return paths


def _product_line_statistics(graph: nx.DiGraph, product_line: str) -> list[str]:
    """Подсчёт отказов и замен деталей линейки непосредственно по графу."""
    models = sorted(
        model
        for model in graph.predecessors(product_line)
        if _node_type(graph, model) == "Model" and _has_relation(graph, model, product_line, "BELONGS_TO")
    )
    if not models:
        return []

    model_counts: dict[str, int] = {}
    component_counts: dict[str, int] = {}
    model_parts: dict[str, set[str]] = {}

    for model in models:
        count = 0
        model_parts[model] = set()
        for target in graph.successors(model):
            attributes = graph.edges[model, target]
            relation_type = attributes.get("type")
            if relation_type == "FAILED" and _node_type(graph, target) == "Component":
                weight = int(attributes.get("weight", 1))
                count += weight
                component_counts[target] = component_counts.get(target, 0) + weight
            elif relation_type == "HAS_PART" and _node_type(graph, target) == "Part":
                model_parts[model].add(target)
        model_counts[model] = count

    lines = [
        "Проверяемая статистика по линейке "
        f"«{_node_name(graph, product_line)}» (сумма весов рёбер FAILED; вес — число чанков):"
    ]
    lines.append(
        "Отказы по моделям: "
        + "; ".join(
            f"{_node_name(graph, model)} — {model_counts[model]}"
            for model in models
        )
        + "."
    )

    if component_counts:
        ordered_components = sorted(
            component_counts,
            key=lambda component: (-component_counts[component], _node_name(graph, component)),
        )
        lines.append(
            "Отказы по узлам: "
            + "; ".join(
                f"{_node_name(graph, component)} — {component_counts[component]}"
                for component in ordered_components
            )
            + "."
        )
        highest_component = component_counts[ordered_components[0]]
        top_components = [
            _node_name(graph, component)
            for component in ordered_components
            if component_counts[component] == highest_component
        ]
        lines.append(
            f"Максимальный вес отказов у узла: {', '.join(top_components)} ({highest_component})."
        )

    if model_counts:
        highest_model_count = max(model_counts.values())
        top_models = [
            _node_name(graph, model)
            for model in models
            if model_counts[model] == highest_model_count
        ]
        lines.append(
            f"Максимальный вес отказов у модели: {', '.join(top_models)} ({highest_model_count})."
        )

    discontinued_lines: list[str] = []
    discontinued_parts: set[str] = set()
    for part in sorted({part for parts in model_parts.values() for part in parts}):
        paths = _replacement_paths(graph, part)
        if not paths:
            continue
        installed_models = [
            _node_name(graph, model)
            for model in models
            if part in model_parts[model]
        ]
        discontinued_parts.add(part)
        for chain in paths:
            discontinued_parts.update(chain[:-1])
            discontinued_lines.append(
                f"{_node_name(graph, part)} (установлена в {', '.join(installed_models)}): "
                f"{' -> '.join(_node_name(graph, node) for node in chain)}; "
                f"последняя замена: {_node_name(graph, chain[-1])}."
            )
    if discontinued_lines:
        discontinued_names = sorted(_node_name(graph, part) for part in discontinued_parts)
        lines.append(
            "Снятые/заменённые позиции по цепочкам REPLACED_BY: "
            + ", ".join(discontinued_names)
            + ". Это исходные детали и промежуточные замены; конечные элементы цепочек "
            "являются последними версиями, если у них нет дальнейшей замены."
        )
        lines.append("Цепочки замен из руководств: " + " ".join(discontinued_lines))

    return lines


def _append_context(
    facts_text: str,
    facts: list[str],
    context_lines: list[str],
) -> tuple[str, list[str]]:
    """Добавление вычисленных по графу фактов в контекст LLM."""
    if not context_lines:
        return facts_text, facts

    section = "\n\nПроверяемые дополнительные факты:\n" + "\n".join(
        f"- {line}" for line in context_lines
    )
    if len(facts_text) + len(section) > MAX_FACTS_LENGTH:
        return facts_text, facts

    return facts_text + section, [*facts, *context_lines]


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

    selected_edges = _collect_local_edges(graph, sorted(anchors), question)
    facts_text, facts, sources = _format_local_facts(graph, selected_edges)
    if not facts:
        return _refused_answer()

    replacement_chains = _replacement_chain_context(graph, selected_edges)
    context_lines = _part_models_context(graph, anchors, question)

    for product_line in sorted(
        node for node in anchors if _node_type(graph, node) == "ProductLine"
    ):
        context_lines.extend(_product_line_statistics(graph, product_line))

    lowered_question = question.lower()
    if any(marker in lowered_question for marker in ("снят", "производств", "устарев", "выведен")):
        for part in sorted(node for node in anchors if _node_type(graph, node) == "Part"):
            paths = _replacement_paths(graph, part)
            if paths:
                chains_text = "; ".join(
                    " -> ".join(_node_name(graph, node) for node in path)
                    for path in sorted(paths)
                )
                context_lines.append(
                    f"Деталь {_node_name(graph, part)} снята с производства; "
                    f"цепочки замен: {chains_text}."
                )

    if replacement_chains:
        context_lines.append(
            "Цепочки замен, рассчитанные по графу: " + "; ".join(replacement_chains)
        )
    facts_text, facts = _append_context(facts_text, facts, context_lines)

    response = llm.chat(
        messages=[
            {
                "role": "system",
                "content": (
                    "Ты отвечаешь по фактам из графа знаний. Используй только переданные факты. "
                    "Стрелка A -RELATION-> B всегда означает отношение от A к B; не меняй направление. "
                    "Семантика: Model -HAS_PART-> Part означает, что деталь установлена в модели; "
                    "Part -IS_A-> Component задаёт тип детали; Model -FAILED-> Component означает отказ узла; "
                    "ErrorCode -INDICATES-> Component связывает код ошибки с узлом; "
                    "ErrorCode -FIXED_BY-> Procedure указывает регламент устранения ошибки; "
                    "Part -REPLACED_BY-> Part означает, что исходная деталь заменена и больше не является "
                    "актуальной. При цепочке A -> B -> C называй A и B снятыми/заменёнными позициями, "
                    "а C — последней актуальной деталью, если у неё нет дальнейшей замены. "
                    "Если спрашивают, в каких моделях установлена деталь, перечисляй точные коды Model "
                    "из фактов о HAS_PART; не подменяй их названием ProductLine. "
                    "Если спрашивают про регламент для узла, используй путь ErrorCode -> FIXED_BY -> Procedure. "
                    "Для поставщика используй SUPPLIED_BY от конечной детали цепочки. "
                    "Не путай FAILED с REPLACED_BY: неисправность узла сама по себе не означает снятие детали. "
                    "Если есть проверяемые дополнительные факты, рассчитанные по графу, используй их "
                    "для точных списков и статистики. При недостатке подтверждений прямо сообщи об этом. "
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


def _global_anchor_nodes(question: str, kb: KnowledgeBase) -> tuple[set[str], set[str]]:
    """Поиск кодовых и словарных якорей для выбора локального или глобального режима."""
    code_nodes = {
        f"{code_type(code)}:{code}"
        for code in find_codes(question)
    }
    name_nodes = {
        f"{entity_type}:{name}"
        for entity_type, name in find_names(question, kb.vocabulary)
        if f"{entity_type}:{name}" in kb.graph
    }
    return code_nodes, name_nodes


def global_search(question: str, kb: KnowledgeBase, llm: LLMClient) -> Answer:
    """Отвечает на обзорный вопрос через релевантные сводки Map-Reduce."""
    code_nodes, name_nodes = _global_anchor_nodes(question, kb)
    non_global_names = {
        node for node in name_nodes if _node_type(kb.graph, node) != "ProductLine"
    }

    # Запросы об известных кодах, узлах и поставщиках точнее обрабатываются локальным поиском.
    if code_nodes or non_global_names:
        return local_search(question, kb, llm)

    product_lines = sorted(
        node for node in name_nodes if _node_type(kb.graph, node) == "ProductLine"
    )

    graph_context: list[str] = []
    for product_line in product_lines:
        graph_context.extend(_product_line_statistics(kb.graph, product_line))

    useful_answers: list[str] = []
    sources: set[str] = set()

    # Шаг map: обработка каждой сводки сообщества согласно контракту FR-5.
    for community in kb.communities:
        summary = community.summary.strip()
        community_context = []
        if any(line in community.nodes for line in product_lines):
            community_context = graph_context
            sources.update(_community_sources(kb, community.nodes))
        if not summary and not community_context:
            continue

        context_text = summary
        if community_context:
            context_text += (
                "\n\nПроверяемые дополнительные данные, рассчитанные непосредственно по графу. "
                "При противоречии со сводкой используй эти данные:\n"
                + "\n".join(f"- {line}" for line in community_context)
            )

        response = llm.chat(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Отвечай только по переданной сводке и проверяемым дополнительным данным. "
                        "Верни JSON с полями answer и score. score — целое число от 0 до 10. "
                        "Поставь 0, если контекст не содержит конкретных фактов для ответа на вопрос. "
                        "Положительную оценку ставь только при наличии фактической опоры. "
                        "Проверяемые данные, рассчитанные по графу, имеют приоритет над текстом сводки. "
                        "Не используй общие знания и не додумывай отсутствующие числа."
                    ),
                },
                {
                    "role": "user",
                    "content": f"Вопрос: {question}\n\nСводка и данные сообщества:\n{context_text}",
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
    # Проверяемая статистика графа остаётся доступной на этапе Reduce,
    # но не считается успешным частичным ответом Map.
    if graph_context:
        partial_answers_text += (
            "\n\nДЕТЕРМИНИРОВАННЫЕ ДАННЫЕ ГРАФА (приоритет над ответами Map):\n"
            + "\n".join(graph_context)
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
                    "Блок 'ДЕТЕРМИНИРОВАННЫЕ ДАННЫЕ ГРАФА' вычислен непосредственно из рёбер "
                    "графа и имеет приоритет над любым ответом Map при расхождении. "
                    "Веса FAILED — это число подтверждающих чанков, а не гарантированное число "
                    "реальных поломок. Если вопрос про максимум, сравнивай именно приведённые веса. "
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
        facts=graph_context,
        refused=False,
    )