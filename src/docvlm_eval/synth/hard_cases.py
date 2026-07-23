"""Hard chart, table, finance, and scientific-paper document programs.

Each factory authors an executable latent graph first, renders the same values through
``DocBuilder``, and derives every reasoning target from the graph. The factories are deterministic
for a caller-provided RNG and expose a five-level curriculum coordinate.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Callable

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
    builder.difficulty = difficulty.to_dict()
    builder.semantic_graph["difficulty"] = difficulty.to_dict()


def hard_table_case(rng: random.Random, level: int = 4) -> HardCase:
    """Dense regional operating table with multi-cell arithmetic and cross-region comparison."""

    difficulty = _difficulty(
        level,
        skills=("table-structure", "aggregation", "margin", "cross-region"),
        base_hops=2,
        cross_region=True,
    )
    regions = ["North", "South", "East", "West"] + [
        f"Aux-{index + 1}" for index in range(difficulty.distractor_count)
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
            (("revenue", revenue, "USD"), ("cost", cost, "USD"), ("units", units, "count")),
            start=1,
        ):
            node_id = f"{name}_{row_index}"
            field_key = f"operations_r{row_index}c{col_index}"
            nodes.append(
                GraphNode(
                    node_id,
                    "table-cell",
                    value,
                    f"{region} {name}",
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
            "approved operating budget",
            "USD",
            {"field_key": "budget"},
        )
    )
    core_revenues = tuple(f"revenue_{index}" for index in range(4))
    core_costs = tuple(f"cost_{index}" for index in range(4))
    queries = [
        GraphQuery(
            "north_revenue",
            "What revenue is reported for North?",
            "value",
            ("revenue_0",),
            "T-table-lookup",
            answer_format="money",
        ),
        GraphQuery(
            "core_revenue",
            "What is the combined revenue of North, South, East, and West?",
            "sum",
            core_revenues,
            "H-table-multicell",
            answer_format="money",
        ),
        GraphQuery(
            "core_profit",
            "What is total operating profit across the four primary regions?",
            "difference",
            ("core_revenue_total", "core_cost_total"),
            "H-table-multistep",
            evidence=core_revenues + core_costs,
            answer_format="money",
        ),
        GraphQuery(
            "largest_revenue",
            "Which primary region has the highest revenue?",
            "argmax",
            core_revenues,
            "H-table-argmax",
            metric="anls",
            answer_format="text",
            parameters={"outputs": ["North", "South", "East", "West"]},
        ),
        GraphQuery(
            "budget_headroom",
            "How much approved budget remains after all listed operating costs?",
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
                "four-region revenue",
                "USD",
            ),
            GraphNode(
                "core_cost_total",
                "latent-aggregate",
                sum(node.value for node in nodes if node.node_id in core_costs),
                "four-region cost",
                "USD",
            ),
            GraphNode(
                "all_cost_total",
                "latent-aggregate",
                sum(node.value for node in nodes if node.node_id.startswith("cost_")),
                "all listed costs",
                "USD",
            ),
        ]
    )
    graph = LatentDocumentGraph(
        graph_id=f"hard-table-{rng.randrange(1_000_000_000)}",
        template_family="hard-operating-table-v1",
        nodes=nodes,
        queries=queries[:level],
        metadata={"primary_rows": 4, "distractor_rows": difficulty.distractor_count},
    )
    b = DocBuilder(
        "dense operating table",
        ["dense-table", "cross-region", "distractors", "multi-step arithmetic"],
        "TEDS + relaxed accuracy",
        page="A4",
        css=".summary{border:1px solid #666;padding:8px;margin-top:12px}.num{text-align:right}",
    )
    b.title("REGIONAL OPERATING REVIEW")
    b.field("Approved operating budget", f"${budget:,}", key="budget", spot=True, cls="summary")
    b.table(
        ["Region", "Revenue", "Operating cost", "Units"],
        rows,
        key="operations",
        spot_cells=spot_cells,
        region="the regional operating table",
    )
    b.task("Reconstruct the table and answer multi-cell operating questions.")
    _attach(b, graph, difficulty)
    b.want_fulltext()
    return HardCase("hard_table", b, "operations", "pdf-native", "scan")


def hard_chart_case(rng: random.Random, level: int = 4) -> HardCase:
    """Programmatic bar chart with exact numeric labels and temporal reasoning."""

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
            f"What index value is shown for {years[-1]}?",
            "value",
            (f"value_{len(values)-1}",),
            "T-chart-read",
            answer_format="integer",
        ),
        GraphQuery(
            "end_change",
            f"What is the percentage change in the index from {years[0]} to {years[-1]}?",
            "percent_change",
            (f"value_{len(values)-1}", "value_0"),
            "H-chart-percent-change",
            answer_format="percent",
        ),
        GraphQuery(
            "peak_year",
            "In which year does the chart reach its highest value?",
            "argmax",
            tuple(f"value_{index}" for index in range(len(values))),
            "H-chart-argmax",
            metric="exact",
            answer_format="text",
            parameters={"outputs": [str(year) for year in years]},
        ),
        GraphQuery(
            "recent_mean",
            "What is the mean index value across the final three years?",
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
    )
    b = DocBuilder(
        "labelled temporal bar chart",
        ["chart", "small-labels", "temporal-reasoning", "visual-comparison"],
        "relaxed accuracy",
        page="A5",
        css=(
            ".chart{height:250px;display:flex;align-items:flex-end;gap:12px;border-left:2px solid #333;"
            "border-bottom:2px solid #333;padding:14px 10px 0}.col{flex:1;text-align:center}.bar{"
            "background:#2f6c9e;color:white;display:flex;align-items:flex-start;justify-content:center;"
            "padding-top:4px;font-weight:bold}.year{font-size:9px;margin-top:5px}.note{font-size:9px}"
        ),
    )
    b.title("SUPPLY RESILIENCE INDEX")
    b.line("Annual composite score (higher is better)", cls="note")
    b.raw("<div class=chart>")
    for index, (year, value) in enumerate(zip(years, values)):
        b.raw(
            f"<div class=col><div class=bar style='height:{value * 2.4}px'>{value}</div>"
            f"<div class=year>{year}</div></div>"
        )
        b.spot(f"bar_{index}", str(value))
    b.raw("</div>")
    b.task("Read exact labels before performing temporal chart calculations.")
    _attach(b, graph, difficulty)
    return HardCase("hard_chart", b, "analytics", "pdf-native", "photo")


def hard_investment_case(rng: random.Random, level: int = 5) -> HardCase:
    """Multi-path beneficial-ownership document with exact look-through reasoning."""

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
                f"{edge.source} to {edge.target} ownership",
                "percent",
                {"field_key": key},
            )
        )
    queries = [
        GraphQuery(
            "direct_birch",
            "What direct ownership does Aurora Fund report in Birch Holdings?",
            "value",
            ("stake_0",),
            "T-finance-relation",
            answer_format="percent",
        ),
        GraphQuery(
            "via_birch",
            "What indirect ownership does Aurora Fund have in Delta Labs through Birch Holdings?",
            "path_product",
            ("ab", "bd"),
            "H-finance-path",
            evidence=("stake_0", "stake_2"),
            answer_format="fraction_percent",
        ),
        GraphQuery(
            "effective_delta",
            "What is Aurora Fund's total effective ownership of Delta Labs across both disclosed paths?",
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
    )
    rows = [
        [companies[0], companies[1], f"{direct_ab * 100:.0f}%"],
        [companies[0], companies[2], f"{direct_ac * 100:.0f}%"],
        [companies[1], companies[3], f"{bd * 100:.0f}%"],
        [companies[2], companies[3], f"{cd * 100:.0f}%"],
    ]
    b = DocBuilder(
        "beneficial ownership disclosure",
        ["investment-relations", "multi-hop", "entity-resolution", "multiple-paths"],
        "relaxed accuracy",
        page="A5",
        css=".legal{font-size:9px;color:#444;border-top:1px solid #999;margin-top:14px;padding-top:8px}",
    )
    b.title("BENEFICIAL OWNERSHIP DISCLOSURE")
    b.table(
        ["Investor", "Direct holding", "Ownership"],
        rows,
        key="holding",
        spot_cells=[(index, 2) for index in range(4)],
        region="the direct ownership schedule",
    )
    b.line(
        "Effective ownership must include every disclosed indirect path. Direct percentages are "
        "multiplicative along a path and additive across independent paths.",
        cls="legal",
    )
    b.task("Resolve entities and calculate indirect beneficial ownership without double-counting.")
    _attach(b, graph, difficulty)
    return HardCase("hard_investment", b, "finance", "pdf-native", "scan")


def hard_science_case(rng: random.Random, level: int = 5) -> HardCase:
    """Two-column research result with control-relative effect and best-condition selection."""

    difficulty = _difficulty(
        level,
        skills=("scientific-table", "control-comparison", "effect-size", "claim-verification"),
        base_hops=3,
        cross_region=True,
    )
    conditions = ["Control", "Compound A", "Compound B", "A+B"]
    control = rng.randrange(92, 112)
    means = [
        control,
        control - rng.randrange(10, 22),
        control - rng.randrange(5, 16),
        control - rng.randrange(22, 36),
    ]
    stderr = [round(rng.uniform(1.2, 4.8), 1) for _ in conditions]
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
                f"{condition} response",
                "relative fluorescence units",
                {"field_key": f"results_r{index}c1"},
            )
        )
        spots.append((index, 1))
    queries = [
        GraphQuery(
            "control_value",
            "What mean response is reported for Control?",
            "value",
            ("mean_0",),
            "T-science-read",
            answer_format="decimal:1",
        ),
        GraphQuery(
            "combination_reduction",
            "By what percentage did A+B reduce the mean response relative to Control?",
            "relative_reduction",
            ("mean_3", "mean_0"),
            "H-science-effect",
            answer_format="percent",
        ),
        GraphQuery(
            "lowest_response",
            "Which condition produced the lowest mean response?",
            "argmin",
            tuple(f"mean_{index}" for index in range(4)),
            "H-science-claim",
            metric="anls",
            answer_format="text",
            parameters={"outputs": conditions},
        ),
        GraphQuery(
            "a_vs_b",
            "What is the difference between the mean responses for Compound A and Compound B?",
            "difference",
            ("mean_1", "mean_2"),
            "H-science-comparison",
            answer_format="decimal:1",
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
            *([queries[3]] if level >= 5 else []),
        ],
        metadata={"control_node": "mean_0", "reported_uncertainty": "standard error"},
    )
    b = DocBuilder(
        "scientific research paper",
        ["two-column-paper", "scientific-table", "effect-size", "claim-verification"],
        "relaxed accuracy",
        page="A4",
        css=(
            ".authors{font-size:9px;text-align:center}.abstract{columns:2;column-gap:18px;"
            "font-size:9px;text-align:justify}.equation{text-align:center;font-family:serif;"
            "margin:12px}.caption{font-size:8px;color:#444}"
        ),
    )
    b.title("Combinatorial Modulation of Cellular Stress Response", level=2)
    b.line("M. Rivera, J. Chen, and S. Okafor", cls="authors")
    b.raw(
        "<div class=abstract><b>Abstract.</b> We compared two compounds and their combination "
        "against an untreated control. Lower fluorescence indicates reduced stress signalling. "
        "All values are independently generated means with standard errors.</div>"
    )
    b.line("Relative effect = (treatment - control) / |control| x 100", cls="equation")
    b.table(
        ["Condition", "Mean response", "SE", "n"],
        rows,
        key="results",
        spot_cells=spots,
        region="Table 1 experimental results",
    )
    b.line(
        "Table 1. Mean response, standard error, and replicate count by intervention.",
        cls="caption",
    )
    b.task("Verify quantitative claims against Table 1 and the stated control-relative equation.")
    _attach(b, graph, difficulty)
    b.want_fulltext()
    return HardCase("hard_science", b, "science", "pdf-native", "scan")


HARD_CASE_FACTORIES: dict[str, Callable[[random.Random, int], HardCase]] = {
    "hard_table": hard_table_case,
    "hard_chart": hard_chart_case,
    "hard_investment": hard_investment_case,
    "hard_science": hard_science_case,
}
