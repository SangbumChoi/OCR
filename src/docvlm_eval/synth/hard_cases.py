"""Hard chart, table, finance, and scientific-paper document programs.

Each factory authors an executable latent graph first, renders the same values through
``DocBuilder``, and derives every reasoning target from the graph. The factories are deterministic
for a caller-provided RNG and expose a five-level curriculum coordinate.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from functools import partial
from typing import Callable

from .hard_layout import hard_layout_spec
from .hard_locale import hard_text
from .latent import (
    DifficultySpec,
    GraphEdge,
    GraphNode,
    GraphQuery,
    LatentDocumentGraph,
)
from .patterns import DocBuilder


@dataclass(frozen=True)
class HardCase:
    key: str
    builder: DocBuilder
    domain: str
    acquisition: str
    degradation_preset: str
    layout_family: str


def _difficulty(
    level: int,
    *,
    skills: tuple[str, ...],
    base_hops: int,
    cross_region: bool,
) -> DifficultySpec:
    return DifficultySpec(
        level=level,
        reasoning_hops=min(base_hops, level),
        distractor_count=max(0, level - 2) * 2,
        visual_density=min(1.0, 0.35 + level * 0.12),
        cross_region=cross_region and level >= 4,
        skills=skills,
    )


def _attach(builder: DocBuilder, graph: LatentDocumentGraph, difficulty: DifficultySpec) -> None:
    graph.add_questions(builder)
    builder.probe(
        "abstain",
        hard_text(graph.language, "absence_q"),
        hard_text(graph.language, "absence_expected"),
    )
    builder.difficulty = difficulty.to_dict()
    builder.semantic_graph["difficulty"] = difficulty.to_dict()
    builder.semantic_graph.setdefault("metadata", {})["layout_family"] = (
        builder.layout_family
    )


def hard_table_case(
    rng: random.Random,
    level: int = 4,
    language: str = "en",
    layout_family: str = "classic-v1",
) -> HardCase:
    """Dense regional operating table with multi-cell arithmetic and cross-region comparison."""

    text = partial(hard_text, language)
    difficulty = _difficulty(
        level,
        skills=("table-structure", "aggregation", "margin", "cross-region"),
        base_hops=2,
        cross_region=True,
    )
    regions = [
        text("region_north"),
        text("region_south"),
        text("region_east"),
        text("region_west"),
    ] + [
        text("region_aux", index=index + 1)
        for index in range(difficulty.distractor_count)
    ]
    rows: list[list[str]] = []
    nodes: list[GraphNode] = []
    spot_cells: list[tuple[int, int]] = []
    for row_index, region in enumerate(regions):
        revenue = rng.randrange(420, 980) * 1000
        cost = rng.randrange(230, min(700, revenue // 1000 - 40)) * 1000
        units = rng.randrange(80, 260)
        rows.append([region, f"${revenue:,}", f"${cost:,}", str(units)])
        for col_index, (name, value, unit) in enumerate(
            (
                ("revenue", revenue, "USD"),
                ("cost", cost, "USD"),
                ("units", units, "count"),
            ),
            start=1,
        ):
            node_id = f"{name}_{row_index}"
            field_key = f"operations_r{row_index}c{col_index}"
            nodes.append(
                GraphNode(
                    node_id,
                    "table-cell",
                    value,
                    f"{region} {text(name)}",
                    unit,
                    {"field_key": field_key, "row": row_index, "column": col_index},
                )
            )
            spot_cells.append((row_index, col_index))
    budget = sum(node.value for node in nodes if node.node_id.startswith("cost_")) + 125_000
    nodes.append(
        GraphNode(
            "budget",
            "summary-field",
            budget,
            text("budget_label"),
            "USD",
            {"field_key": "budget"},
        )
    )
    core_revenues = tuple(f"revenue_{index}" for index in range(4))
    core_costs = tuple(f"cost_{index}" for index in range(4))
    queries = [
        GraphQuery(
            "north_revenue",
            text("table_q_north"),
            "value",
            ("revenue_0",),
            "T-table-lookup",
            answer_format="money",
        ),
        GraphQuery(
            "core_revenue",
            text("table_q_core_revenue"),
            "sum",
            core_revenues,
            "H-table-multicell",
            answer_format="money",
        ),
        GraphQuery(
            "core_profit",
            text("table_q_profit"),
            "difference",
            ("core_revenue_total", "core_cost_total"),
            "H-table-multistep",
            evidence=core_revenues + core_costs,
            answer_format="money",
        ),
        GraphQuery(
            "largest_revenue",
            text("table_q_largest"),
            "argmax",
            core_revenues,
            "H-table-argmax",
            metric="anls",
            answer_format="text",
            parameters={"outputs": regions[:4]},
        ),
        GraphQuery(
            "budget_headroom",
            text("table_q_budget"),
            "difference",
            ("budget", "all_cost_total"),
            "H-table-cross-region",
            evidence=("budget",) + tuple(
                f"cost_{index}" for index in range(len(regions))
            ),
            answer_format="money",
        ),
    ]
    nodes.extend(
        [
            GraphNode(
                "core_revenue_total",
                "latent-aggregate",
                sum(node.value for node in nodes if node.node_id in core_revenues),
                text("four_region_revenue"),
                "USD",
            ),
            GraphNode(
                "core_cost_total",
                "latent-aggregate",
                sum(node.value for node in nodes if node.node_id in core_costs),
                text("four_region_cost"),
                "USD",
            ),
            GraphNode(
                "all_cost_total",
                "latent-aggregate",
                sum(node.value for node in nodes if node.node_id.startswith("cost_")),
                text("all_costs"),
                "USD",
            ),
        ]
    )
    graph = LatentDocumentGraph(
        graph_id=f"hard-table-{rng.randrange(1_000_000_000)}",
        template_family="hard-operating-table-v1",
        nodes=nodes,
        queries=queries[:level],
        metadata={
            "primary_rows": 4,
            "distractor_rows": difficulty.distractor_count,
        },
        language=language,
    )
    layout = hard_layout_spec("hard_table", layout_family)
    b = DocBuilder(
        "dense operating table",
        ["dense-table", "cross-region", "distractors", "multi-step arithmetic"],
        "TEDS + relaxed accuracy",
        page=layout.page,
        css=layout.css,
        language=language,
        layout_family=layout.family,
    )
    headers = [
        text("table_h_region"),
        text("table_h_revenue"),
        text("table_h_cost"),
        text("table_h_units"),
    ]

    def add_budget() -> None:
        b.field(
            text("table_budget"),
            f"${budget:,}",
            key="budget",
            spot=True,
            cls="summary",
        )

    def add_table() -> None:
        b.table(
            headers,
            rows,
            key="operations",
            spot_cells=spot_cells,
            region=text("table_region"),
        )

    if layout.family == "classic-v1":
        b.title(text("table_title"))
        add_budget()
        add_table()
    elif layout.family == "compact-v1":
        b.raw("<section class=hard-header>")
        b.title(text("table_title"))
        add_budget()
        b.raw("</section>")
        add_table()
    else:
        b.raw("<header class=hard-header>")
        b.title(text("table_title"))
        b.raw("</header><section class=data-section>")
        add_table()
        b.raw("</section>")
        add_budget()
    b.task(text("table_task"))
    _attach(b, graph, difficulty)
    b.want_fulltext(text("fulltext"))
    return HardCase(
        "hard_table",
        b,
        "operations",
        "pdf-native",
        "scan",
        layout.family,
    )


def hard_chart_case(
    rng: random.Random,
    level: int = 4,
    language: str = "en",
    layout_family: str = "classic-v1",
) -> HardCase:
    """Programmatic bar chart with exact numeric labels and temporal reasoning."""

    text = partial(hard_text, language)
    difficulty = _difficulty(
        level,
        skills=("chart-reading", "temporal-comparison", "percent-change", "argmax"),
        base_hops=2,
        cross_region=False,
    )
    years = list(range(2020, 2026))
    values = [rng.randrange(42, 75)]
    for _ in years[1:]:
        values.append(max(20, values[-1] + rng.randrange(-8, 16)))
    nodes = [
        GraphNode(
            f"value_{index}",
            "chart-mark",
            value,
            str(year),
            "index points",
            {"field_key": f"bar_{index}"},
        )
        for index, (year, value) in enumerate(zip(years, values))
    ]
    queries = [
        GraphQuery(
            "latest_value",
            text("chart_q_latest", latest=years[-1]),
            "value",
            (f"value_{len(values)-1}",),
            "T-chart-read",
            answer_format="integer",
        ),
        GraphQuery(
            "end_change",
            text("chart_q_change", first=years[0], latest=years[-1]),
            "percent_change",
            (f"value_{len(values)-1}", "value_0"),
            "H-chart-percent-change",
            answer_format="percent",
        ),
        GraphQuery(
            "peak_year",
            text("chart_q_peak"),
            "argmax",
            tuple(f"value_{index}" for index in range(len(values))),
            "H-chart-argmax",
            metric="exact",
            answer_format="text",
            parameters={"outputs": [str(year) for year in years]},
        ),
        GraphQuery(
            "recent_mean",
            text("chart_q_mean"),
            "mean",
            tuple(f"value_{index}" for index in range(len(values) - 3, len(values))),
            "H-chart-aggregate",
            answer_format="decimal:2",
        ),
    ]
    graph = LatentDocumentGraph(
        graph_id=f"hard-chart-{rng.randrange(1_000_000_000)}",
        template_family="hard-labelled-bar-chart-v1",
        nodes=nodes,
        queries=queries[:level],
        language=language,
    )
    layout = hard_layout_spec("hard_chart", layout_family)
    b = DocBuilder(
        "labelled temporal bar chart",
        ["chart", "small-labels", "temporal-reasoning", "visual-comparison"],
        "relaxed accuracy",
        page=layout.page,
        css=layout.css,
        language=language,
        layout_family=layout.family,
    )
    chart_height = {
        "classic-v1": 220,
        "compact-v1": 150,
        "report-v1": 260,
    }[layout.family]
    max_value = max(values)

    def add_chart() -> None:
        b.raw("<div class=chart>")
        for index, (year, value) in enumerate(zip(years, values)):
            bar_height = max(18, round(value / max_value * chart_height))
            b.raw(
                f"<div class=col><div class=bar style='height:{bar_height}px'>{value}</div>"
                f"<div class=year>{year}</div></div>"
            )
            b.spot(f"bar_{index}", str(value))
        b.raw("</div>")

    if layout.family == "classic-v1":
        b.title(text("chart_title"))
        b.line(text("chart_note"), cls="note")
        add_chart()
    elif layout.family == "compact-v1":
        b.raw("<section class=chart-shell><header class=chart-copy>")
        b.title(text("chart_title"))
        b.line(text("chart_note"), cls="note")
        b.raw("</header>")
        add_chart()
        b.raw("</section>")
    else:
        b.title(text("chart_title"))
        b.raw("<section class=chart-card>")
        add_chart()
        b.line(text("chart_note"), cls="note")
        b.raw("</section>")
    b.task(text("chart_task"))
    _attach(b, graph, difficulty)
    b.want_fulltext(text("fulltext"))
    return HardCase(
        "hard_chart",
        b,
        "analytics",
        "pdf-native",
        "photo",
        layout.family,
    )


def hard_investment_case(
    rng: random.Random,
    level: int = 5,
    language: str = "en",
    layout_family: str = "classic-v1",
) -> HardCase:
    """Multi-path beneficial-ownership document with exact look-through reasoning."""

    text = partial(hard_text, language)
    difficulty = _difficulty(
        level,
        skills=("entity-resolution", "ownership-paths", "multi-hop-relation", "percentage"),
        base_hops=3,
        cross_region=True,
    )
    companies = ["Aurora Fund", "Birch Holdings", "Cobalt SPV", "Delta Labs"]
    direct_ab = rng.choice([0.30, 0.35, 0.40])
    direct_ac = rng.choice([0.15, 0.20, 0.25])
    bd = rng.choice([0.40, 0.45, 0.50])
    cd = rng.choice([0.20, 0.25, 0.30])
    nodes = [
        GraphNode(company.lower().replace(" ", "_"), "entity", company, company)
        for company in companies
    ]
    ownership_values = [direct_ab, direct_ac, bd, cd]
    ownership_keys = ["holding_r0c2", "holding_r1c2", "holding_r2c2", "holding_r3c2"]
    edges = [
        GraphEdge("ab", "aurora_fund", "owns", "birch_holdings", direct_ab),
        GraphEdge("ac", "aurora_fund", "owns", "cobalt_spv", direct_ac),
        GraphEdge("bd", "birch_holdings", "owns", "delta_labs", bd),
        GraphEdge("cd", "cobalt_spv", "owns", "delta_labs", cd),
    ]
    for index, (edge, value, key) in enumerate(zip(edges, ownership_values, ownership_keys)):
        nodes.append(
            GraphNode(
                f"stake_{index}",
                "relation-label",
                value * 100,
                text(
                    "ownership_label",
                    source=edge.source,
                    target=edge.target,
                ),
                "percent",
                {"field_key": key},
            )
        )
    queries = [
        GraphQuery(
            "direct_birch",
            text("investment_q_direct"),
            "value",
            ("stake_0",),
            "T-finance-relation",
            answer_format="percent",
        ),
        GraphQuery(
            "via_birch",
            text("investment_q_path"),
            "path_product",
            ("ab", "bd"),
            "H-finance-path",
            evidence=("stake_0", "stake_2"),
            answer_format="fraction_percent",
        ),
        GraphQuery(
            "effective_delta",
            text("investment_q_total"),
            "sum_products",
            (),
            "H-finance-multipath",
            evidence=("stake_0", "stake_1", "stake_2", "stake_3"),
            parameters={"paths": [["ab", "bd"], ["ac", "cd"]]},
            answer_format="fraction_percent",
        ),
    ]
    graph = LatentDocumentGraph(
        graph_id=f"hard-investment-{rng.randrange(1_000_000_000)}",
        template_family="hard-beneficial-ownership-v1",
        nodes=nodes,
        edges=edges,
        queries=[
            queries[0],
            *([queries[1]] if level >= 3 else []),
            *([queries[2]] if level >= 5 else []),
        ],
        metadata={"path_count": 2, "relation_depth": 2},
        language=language,
    )
    rows = [
        [companies[0], companies[1], f"{direct_ab * 100:.0f}%"],
        [companies[0], companies[2], f"{direct_ac * 100:.0f}%"],
        [companies[1], companies[3], f"{bd * 100:.0f}%"],
        [companies[2], companies[3], f"{cd * 100:.0f}%"],
    ]
    layout = hard_layout_spec("hard_investment", layout_family)
    b = DocBuilder(
        "beneficial ownership disclosure",
        ["investment-relations", "multi-hop", "entity-resolution", "multiple-paths"],
        "relaxed accuracy",
        page=layout.page,
        css=layout.css,
        language=language,
        layout_family=layout.family,
    )
    headers = [
        text("investment_h_investor"),
        text("investment_h_holding"),
        text("investment_h_ownership"),
    ]

    def add_table() -> None:
        b.table(
            headers,
            rows,
            key="holding",
            spot_cells=[(index, 2) for index in range(4)],
            region=text("investment_region"),
        )

    if layout.family == "classic-v1":
        b.title(text("investment_title"))
        add_table()
        b.line(text("investment_legal"), cls="legal")
    elif layout.family == "compact-v1":
        b.title(text("investment_title"))
        b.raw("<section class=disclosure-grid><div>")
        add_table()
        b.raw("</div>")
        b.line(text("investment_legal"), cls="legal")
        b.raw("</section>")
    else:
        b.line(text("investment_legal"), cls="legal")
        b.title(text("investment_title"))
        b.raw("<section class=ownership-card>")
        add_table()
        b.raw("</section>")
    b.task(text("investment_task"))
    _attach(b, graph, difficulty)
    b.want_fulltext(text("fulltext"))
    return HardCase(
        "hard_investment",
        b,
        "finance",
        "pdf-native",
        "scan",
        layout.family,
    )


def hard_science_case(
    rng: random.Random,
    level: int = 5,
    language: str = "en",
    layout_family: str = "classic-v1",
) -> HardCase:
    """Research result with exact effect, uncertainty, and inference programs."""

    text = partial(hard_text, language)
    difficulty = _difficulty(
        level,
        skills=(
            "scientific-table",
            "control-comparison",
            "effect-size",
            "confidence-interval",
            "statistical-inference",
            "claim-verification",
        ),
        base_hops=5,
        cross_region=True,
    )
    conditions = [
        text("condition_control"),
        text("condition_a"),
        text("condition_b"),
        text("condition_ab"),
    ]
    control = rng.randrange(92, 112)
    stderr = [
        value / 10
        for value in rng.sample(range(12, 49), k=len(conditions))
    ]
    pooled_b = math.sqrt(stderr[0] ** 2 + stderr[2] ** 2)
    b_supported = rng.random() < 0.5
    b_effect = (
        math.ceil(2.3 * pooled_b)
        if b_supported
        else max(1, math.floor(1.2 * pooled_b))
    )
    means = [
        control,
        control - rng.randrange(10, 22),
        control - b_effect,
        control - rng.randrange(22, 36),
    ]
    nodes: list[GraphNode] = []
    rows: list[list[str]] = []
    spots: list[tuple[int, int]] = []
    for index, (condition, mean, se) in enumerate(zip(conditions, means, stderr)):
        rows.append([condition, f"{mean:.1f}", f"{se:.1f}", str(rng.randrange(6, 13))])
        nodes.append(
            GraphNode(
                f"mean_{index}",
                "experimental-result",
                mean,
                text("response_label", condition=condition),
                "relative fluorescence units",
                {"field_key": f"results_r{index}c1"},
            )
        )
        nodes.append(
            GraphNode(
                f"se_{index}",
                "experimental-uncertainty",
                se,
                text("uncertainty_label", condition=condition),
                "standard error",
                {"field_key": f"results_r{index}c2"},
            )
        )
        spots.extend(((index, 1), (index, 2)))
    queries = [
        GraphQuery(
            "control_value",
            text("science_q_control"),
            "value",
            ("mean_0",),
            "T-science-read",
            answer_format="decimal:1",
        ),
        GraphQuery(
            "combination_reduction",
            text("science_q_reduction"),
            "relative_reduction",
            ("mean_3", "mean_0"),
            "H-science-effect",
            answer_format="percent",
        ),
        GraphQuery(
            "lowest_response",
            text("science_q_lowest"),
            "argmin",
            tuple(f"mean_{index}" for index in range(4)),
            "H-science-claim",
            metric="anls",
            answer_format="text",
            parameters={"outputs": conditions},
        ),
        GraphQuery(
            "a_vs_b",
            text("science_q_difference"),
            "difference",
            ("mean_1", "mean_2"),
            "H-science-comparison",
            answer_format="decimal:1",
        ),
        GraphQuery(
            "combination_interval",
            text("science_q_ci"),
            "confidence_interval",
            ("mean_3", "se_3"),
            "H-science-confidence-interval",
            metric="anls",
            answer_format="text",
            parameters={
                "critical_value": 1.96,
                "decimal_places": 1,
                "separator": text("science_interval_separator"),
            },
        ),
        GraphQuery(
            "most_precise_condition",
            text("science_q_precision"),
            "argmin",
            tuple(f"se_{index}" for index in range(4)),
            "H-science-uncertainty",
            metric="anls",
            answer_format="text",
            parameters={"outputs": conditions},
        ),
        GraphQuery(
            "compound_b_significance",
            text("science_q_significance", condition=conditions[2]),
            "significance_decision",
            ("mean_2", "mean_0", "se_2", "se_0"),
            "H-science-inference",
            metric="anls",
            answer_format="text",
            parameters={
                "threshold": 1.96,
                "outputs": [
                    text("science_not_supported"),
                    text("science_supported"),
                ],
            },
        ),
    ]
    graph = LatentDocumentGraph(
        graph_id=f"hard-science-{rng.randrange(1_000_000_000)}",
        template_family="hard-scientific-results-v1",
        nodes=nodes,
        queries=[
            queries[0],
            *([queries[2]] if level >= 2 else []),
            *([queries[1]] if level >= 3 else []),
            *([queries[4], queries[5]] if level >= 4 else []),
            *([queries[3], queries[6]] if level >= 5 else []),
        ],
        metadata={
            "control_node": "mean_0",
            "reported_uncertainty": "standard error",
            "confidence_level": 0.95,
            "critical_value": 1.96,
            "significance_rule": "two-sided absolute z threshold",
            "authored_compound_b_supported": b_supported,
        },
        language=language,
    )
    layout = hard_layout_spec("hard_science", layout_family)
    b = DocBuilder(
        "scientific research paper",
        [
            "two-column-paper",
            "scientific-table",
            "effect-size",
            "uncertainty",
            "statistical-inference",
            "claim-verification",
        ],
        "relaxed accuracy",
        page=layout.page,
        css=layout.css,
        language=language,
        layout_family=layout.family,
    )
    headers = [
        text("science_h_condition"),
        text("science_h_mean"),
        text("science_h_se"),
        text("science_h_n"),
    ]

    def add_table() -> None:
        b.table(
            headers,
            rows,
            key="results",
            spot_cells=spots,
            region=text("science_region"),
        )

    if layout.family == "classic-v1":
        b.title(text("science_title"), level=2)
        b.line("M. Rivera, J. Chen, and S. Okafor", cls="authors")
        b.raw(f"<div class=abstract>{text('science_abstract')}</div>")
        b.line(text("science_equation"), cls="equation")
        b.line(text("science_inference"), cls="equation")
        add_table()
        b.line(text("science_caption"), cls="caption")
    elif layout.family == "compact-v1":
        b.raw("<section class=paper-grid><div class=paper-intro>")
        b.title(text("science_title"), level=2)
        b.line("M. Rivera, J. Chen, and S. Okafor", cls="authors")
        b.raw(f"<div class=abstract>{text('science_abstract')}</div></div>")
        b.raw("<div class=paper-results>")
        b.line(text("science_equation"), cls="equation")
        b.line(text("science_inference"), cls="equation")
        add_table()
        b.line(text("science_caption"), cls="caption")
        b.raw("</div></section>")
    else:
        b.title(text("science_title"), level=2)
        b.line("M. Rivera, J. Chen, and S. Okafor", cls="authors")
        b.raw("<section class=results-card>")
        add_table()
        b.line(text("science_caption"), cls="caption")
        b.raw("</section>")
        b.raw(f"<div class=abstract>{text('science_abstract')}</div>")
        b.line(text("science_equation"), cls="equation")
        b.line(text("science_inference"), cls="equation")
    b.task(text("science_task"))
    _attach(b, graph, difficulty)
    b.want_fulltext(text("fulltext"))
    return HardCase(
        "hard_science",
        b,
        "science",
        "pdf-native",
        "scan",
        layout.family,
    )


HARD_CASE_FACTORIES: dict[str, Callable[..., HardCase]] = {
    "hard_table": hard_table_case,
    "hard_chart": hard_chart_case,
    "hard_investment": hard_investment_case,
    "hard_science": hard_science_case,
}
