"""Locale projections for executable hard-document programs."""

from __future__ import annotations

import re
from typing import Any


HARD_DOCUMENT_LANGUAGES = ("en", "es", "ko", "ja", "zh")


_CATALOG: dict[str, dict[str, str]] = {
    "en": {
        "concise": "Answer concisely, no explanation.",
        "fulltext": "Transcribe all text in this document in reading order.",
        "region_north": "North",
        "region_south": "South",
        "region_east": "East",
        "region_west": "West",
        "region_aux": "Aux-{index}",
        "revenue": "revenue",
        "cost": "operating cost",
        "units": "units",
        "budget_label": "approved operating budget",
        "table_q_north": "What revenue is reported for North?",
        "table_q_core_revenue": "What is the combined revenue of North, South, East, and West?",
        "table_q_profit": "What is total operating profit across the four primary regions?",
        "table_q_largest": "Which primary region has the highest revenue?",
        "table_q_budget": "How much approved budget remains after all listed operating costs?",
        "four_region_revenue": "four-region revenue",
        "four_region_cost": "four-region cost",
        "all_costs": "all listed costs",
        "table_title": "REGIONAL OPERATING REVIEW",
        "table_budget": "Approved operating budget",
        "table_h_region": "Region",
        "table_h_revenue": "Revenue",
        "table_h_cost": "Operating cost",
        "table_h_units": "Units",
        "table_region": "the regional operating table",
        "table_task": "Reconstruct the table and answer multi-cell operating questions.",
        "chart_unit": "index points",
        "chart_q_latest": "What index value is shown for {latest}?",
        "chart_q_change": "What is the percentage change in the index from {first} to {latest}?",
        "chart_q_peak": "In which year does the chart reach its highest value?",
        "chart_q_mean": "What is the mean index value across the final three years?",
        "chart_title": "SUPPLY RESILIENCE INDEX",
        "chart_note": "Annual composite score (higher is better)",
        "chart_task": "Read exact labels before performing temporal chart calculations.",
        "ownership_label": "{source} to {target} ownership",
        "investment_q_direct": "What direct ownership does Aurora Fund report in Birch Holdings?",
        "investment_q_path": "What indirect ownership does Aurora Fund have in Delta Labs through Birch Holdings?",
        "investment_q_total": "What is Aurora Fund's total effective ownership of Delta Labs across both disclosed paths?",
        "investment_title": "BENEFICIAL OWNERSHIP DISCLOSURE",
        "investment_h_investor": "Investor",
        "investment_h_holding": "Direct holding",
        "investment_h_ownership": "Ownership",
        "investment_region": "the direct ownership schedule",
        "investment_legal": "Effective ownership must include every disclosed indirect path. Direct percentages are multiplicative along a path and additive across independent paths.",
        "investment_task": "Resolve entities and calculate indirect beneficial ownership without double-counting.",
        "condition_control": "Control",
        "condition_a": "Compound A",
        "condition_b": "Compound B",
        "condition_ab": "A+B",
        "response_label": "{condition} response",
        "science_unit": "relative fluorescence units",
        "science_q_control": "What mean response is reported for Control?",
        "science_q_reduction": "By what percentage did A+B reduce the mean response relative to Control?",
        "science_q_lowest": "Which condition produced the lowest mean response?",
        "science_q_difference": "What is the difference between the mean responses for Compound A and Compound B?",
        "science_title": "Combinatorial Modulation of Cellular Stress Response",
        "science_abstract": "<b>Abstract.</b> We compared two compounds and their combination against an untreated control. Lower fluorescence indicates reduced stress signalling. All values are independently generated means with standard errors.",
        "science_equation": "Relative effect = (treatment - control) / |control| x 100",
        "science_h_condition": "Condition",
        "science_h_mean": "Mean response",
        "science_h_se": "SE",
        "science_h_n": "n",
        "science_region": "Table 1 experimental results",
        "science_caption": "Table 1. Mean response, standard error, and replicate count by intervention.",
        "science_task": "Verify quantitative claims against Table 1 and the stated control-relative equation.",
        "r_read": "Read {label} directly: {value}.",
        "r_add": "Add {values} = {result}.",
        "r_average": "Average ({values}) / {count} = {result}.",
        "r_subtract": "Subtract {right_label} ({right}) from {left_label} ({left}) = {result}.",
        "r_divide": "Divide {left} by {right} = {result}.",
        "r_reduction": "Relative reduction = ({right} - {left}) / {denominator} x 100 = {result}%.",
        "r_change": "Percent change = ({left} - {right}) / {denominator} x 100 = {result}%.",
        "r_extreme": "{label} has the {direction} value, {value}.",
        "largest": "largest",
        "smallest": "smallest",
        "r_weighted": "Weighted sum = {terms} = {result}.",
        "r_path": "Multiply path weights {weights} = {result}.",
        "r_paths": "Sum path products ({paths}) = {result}.",
    },
    "es": {
        "concise": "Responde brevemente, sin explicación.",
        "fulltext": "Transcribe todo el texto de este documento en orden de lectura.",
        "region_north": "Norte",
        "region_south": "Sur",
        "region_east": "Este",
        "region_west": "Oeste",
        "region_aux": "Aux-{index}",
        "revenue": "ingresos",
        "cost": "coste operativo",
        "units": "unidades",
        "budget_label": "presupuesto operativo aprobado",
        "table_q_north": "¿Qué ingresos se indican para Norte?",
        "table_q_core_revenue": "¿Cuál es el ingreso combinado de Norte, Sur, Este y Oeste?",
        "table_q_profit": "¿Cuál es el beneficio operativo total de las cuatro regiones principales?",
        "table_q_largest": "¿Qué región principal tiene los ingresos más altos?",
        "table_q_budget": "¿Cuánto presupuesto aprobado queda después de todos los costes operativos indicados?",
        "four_region_revenue": "ingresos de cuatro regiones",
        "four_region_cost": "coste de cuatro regiones",
        "all_costs": "todos los costes indicados",
        "table_title": "REVISIÓN OPERATIVA REGIONAL",
        "table_budget": "Presupuesto operativo aprobado",
        "table_h_region": "Región",
        "table_h_revenue": "Ingresos",
        "table_h_cost": "Coste operativo",
        "table_h_units": "Unidades",
        "table_region": "la tabla operativa regional",
        "table_task": "Reconstruye la tabla y responde preguntas operativas de varias celdas.",
        "chart_unit": "puntos de índice",
        "chart_q_latest": "¿Qué valor del índice se muestra para {latest}?",
        "chart_q_change": "¿Cuál es el cambio porcentual del índice de {first} a {latest}?",
        "chart_q_peak": "¿En qué año alcanza el gráfico su valor máximo?",
        "chart_q_mean": "¿Cuál es el valor medio del índice en los últimos tres años?",
        "chart_title": "ÍNDICE DE RESILIENCIA DEL SUMINISTRO",
        "chart_note": "Puntuación compuesta anual (un valor mayor es mejor)",
        "chart_task": "Lee las etiquetas exactas antes de realizar cálculos temporales.",
        "ownership_label": "participación de {source} en {target}",
        "investment_q_direct": "¿Qué participación directa declara Aurora Fund en Birch Holdings?",
        "investment_q_path": "¿Qué participación indirecta tiene Aurora Fund en Delta Labs a través de Birch Holdings?",
        "investment_q_total": "¿Cuál es la participación efectiva total de Aurora Fund en Delta Labs a través de las dos rutas declaradas?",
        "investment_title": "DECLARACIÓN DE TITULARIDAD EFECTIVA",
        "investment_h_investor": "Inversor",
        "investment_h_holding": "Participación directa",
        "investment_h_ownership": "Titularidad",
        "investment_region": "la relación de titularidad directa",
        "investment_legal": "La titularidad efectiva debe incluir todas las rutas indirectas declaradas. Los porcentajes se multiplican a lo largo de una ruta y se suman entre rutas independientes.",
        "investment_task": "Resuelve las entidades y calcula la titularidad efectiva indirecta sin doble conteo.",
        "condition_control": "Control",
        "condition_a": "Compuesto A",
        "condition_b": "Compuesto B",
        "condition_ab": "A+B",
        "response_label": "respuesta de {condition}",
        "science_unit": "unidades relativas de fluorescencia",
        "science_q_control": "¿Qué respuesta media se indica para Control?",
        "science_q_reduction": "¿En qué porcentaje redujo A+B la respuesta media respecto a Control?",
        "science_q_lowest": "¿Qué condición produjo la respuesta media más baja?",
        "science_q_difference": "¿Cuál es la diferencia entre las respuestas medias del Compuesto A y el Compuesto B?",
        "science_title": "MODULACIÓN COMBINATORIA DE LA RESPUESTA AL ESTRÉS CELULAR",
        "science_abstract": "<b>Resumen.</b> Comparamos dos compuestos y su combinación con un control sin tratar. Una fluorescencia menor indica una señal de estrés reducida. Todos los valores son medias generadas independientemente con errores estándar.",
        "science_equation": "Efecto relativo = (tratamiento - control) / |control| x 100",
        "science_h_condition": "Condición",
        "science_h_mean": "Respuesta media",
        "science_h_se": "EE",
        "science_h_n": "n",
        "science_region": "resultados experimentales de la Tabla 1",
        "science_caption": "Tabla 1. Respuesta media, error estándar y número de réplicas por intervención.",
        "science_task": "Verifica las afirmaciones cuantitativas con la Tabla 1 y la ecuación relativa al control.",
        "r_read": "Lee directamente {label}: {value}.",
        "r_add": "Suma {values} = {result}.",
        "r_average": "Promedio ({values}) / {count} = {result}.",
        "r_subtract": "Resta {right_label} ({right}) de {left_label} ({left}) = {result}.",
        "r_divide": "Divide {left} entre {right} = {result}.",
        "r_reduction": "Reducción relativa = ({right} - {left}) / {denominator} x 100 = {result}%.",
        "r_change": "Cambio porcentual = ({left} - {right}) / {denominator} x 100 = {result}%.",
        "r_extreme": "{label} tiene el valor {direction}, {value}.",
        "largest": "máximo",
        "smallest": "mínimo",
        "r_weighted": "Suma ponderada = {terms} = {result}.",
        "r_path": "Multiplica los pesos de la ruta {weights} = {result}.",
        "r_paths": "Suma los productos de las rutas ({paths}) = {result}.",
    },
    "ko": {
        "concise": "설명 없이 간단히 답하세요.",
        "fulltext": "이 문서의 모든 텍스트를 읽기 순서대로 전사하세요.",
        "region_north": "북부",
        "region_south": "남부",
        "region_east": "동부",
        "region_west": "서부",
        "region_aux": "보조-{index}",
        "revenue": "매출",
        "cost": "운영비",
        "units": "수량",
        "budget_label": "승인된 운영 예산",
        "table_q_north": "북부의 보고 매출은 얼마입니까?",
        "table_q_core_revenue": "북부, 남부, 동부, 서부의 합산 매출은 얼마입니까?",
        "table_q_profit": "4개 주요 지역의 총 영업이익은 얼마입니까?",
        "table_q_largest": "매출이 가장 높은 주요 지역은 어디입니까?",
        "table_q_budget": "표시된 모든 운영비를 제외한 승인 예산 잔액은 얼마입니까?",
        "four_region_revenue": "4개 지역 매출",
        "four_region_cost": "4개 지역 운영비",
        "all_costs": "표시된 전체 운영비",
        "table_title": "지역별 운영 검토",
        "table_budget": "승인된 운영 예산",
        "table_h_region": "지역",
        "table_h_revenue": "매출",
        "table_h_cost": "운영비",
        "table_h_units": "수량",
        "table_region": "지역별 운영 표",
        "table_task": "표를 재구성하고 여러 셀을 사용하는 운영 질문에 답하세요.",
        "chart_unit": "지수 점수",
        "chart_q_latest": "{latest}년의 지수 값은 얼마입니까?",
        "chart_q_change": "{first}년부터 {latest}년까지 지수의 변화율은 얼마입니까?",
        "chart_q_peak": "그래프의 값이 가장 높은 연도는 언제입니까?",
        "chart_q_mean": "마지막 3개 연도의 평균 지수 값은 얼마입니까?",
        "chart_title": "공급 회복탄력성 지수",
        "chart_note": "연간 종합 점수(높을수록 좋음)",
        "chart_task": "시계열 계산을 하기 전에 정확한 레이블을 읽으세요.",
        "ownership_label": "{source}에서 {target}으로의 지분",
        "investment_q_direct": "Aurora Fund가 Birch Holdings에 대해 공시한 직접 지분은 얼마입니까?",
        "investment_q_path": "Aurora Fund가 Birch Holdings를 통해 Delta Labs에 보유한 간접 지분은 얼마입니까?",
        "investment_q_total": "공시된 두 경로를 통한 Aurora Fund의 Delta Labs 총 유효 지분은 얼마입니까?",
        "investment_title": "실질 소유 지분 공시",
        "investment_h_investor": "투자자",
        "investment_h_holding": "직접 보유 대상",
        "investment_h_ownership": "지분율",
        "investment_region": "직접 소유 지분 표",
        "investment_legal": "유효 지분은 공시된 모든 간접 경로를 포함해야 합니다. 직접 지분율은 경로를 따라 곱하고 독립 경로 사이에서는 더합니다.",
        "investment_task": "법인을 식별하고 중복 계산 없이 간접 실질 지분을 계산하세요.",
        "condition_control": "대조군",
        "condition_a": "화합물 A",
        "condition_b": "화합물 B",
        "condition_ab": "A+B",
        "response_label": "{condition} 반응",
        "science_unit": "상대 형광 단위",
        "science_q_control": "대조군의 평균 반응값은 얼마입니까?",
        "science_q_reduction": "A+B는 대조군 대비 평균 반응을 몇 퍼센트 감소시켰습니까?",
        "science_q_lowest": "평균 반응이 가장 낮은 조건은 무엇입니까?",
        "science_q_difference": "화합물 A와 화합물 B의 평균 반응 차이는 얼마입니까?",
        "science_title": "세포 스트레스 반응의 조합 조절",
        "science_abstract": "<b>초록.</b> 두 화합물과 그 조합을 무처리 대조군과 비교했습니다. 형광이 낮을수록 스트레스 신호가 감소했음을 뜻합니다. 모든 값은 독립적으로 생성된 평균과 표준오차입니다.",
        "science_equation": "상대 효과 = (처리군 - 대조군) / |대조군| x 100",
        "science_h_condition": "조건",
        "science_h_mean": "평균 반응",
        "science_h_se": "표준오차",
        "science_h_n": "n",
        "science_region": "표 1 실험 결과",
        "science_caption": "표 1. 중재별 평균 반응, 표준오차 및 반복 수.",
        "science_task": "표 1과 대조군 대비 식을 사용해 정량적 주장을 검증하세요.",
        "r_read": "{label} 값을 직접 읽으면 {value}입니다.",
        "r_add": "{values}을 더하면 {result}입니다.",
        "r_average": "평균은 ({values}) / {count} = {result}입니다.",
        "r_subtract": "{left_label}({left})에서 {right_label}({right})을 빼면 {result}입니다.",
        "r_divide": "{left}을 {right}로 나누면 {result}입니다.",
        "r_reduction": "상대 감소율 = ({right} - {left}) / {denominator} x 100 = {result}%입니다.",
        "r_change": "변화율 = ({left} - {right}) / {denominator} x 100 = {result}%입니다.",
        "r_extreme": "{label}의 값 {value}이 {direction}입니다.",
        "largest": "최댓값",
        "smallest": "최솟값",
        "r_weighted": "가중합 = {terms} = {result}입니다.",
        "r_path": "경로 지분율을 곱하면 {weights} = {result}입니다.",
        "r_paths": "경로별 곱을 더하면 ({paths}) = {result}입니다.",
    },
    "ja": {
        "concise": "説明せず簡潔に答えてください。",
        "fulltext": "この文書のすべての文字を読み順に転記してください。",
        "region_north": "北部",
        "region_south": "南部",
        "region_east": "東部",
        "region_west": "西部",
        "region_aux": "補助-{index}",
        "revenue": "売上",
        "cost": "運営費",
        "units": "数量",
        "budget_label": "承認済み運営予算",
        "table_q_north": "北部の報告売上はいくらですか？",
        "table_q_core_revenue": "北部、南部、東部、西部の合計売上はいくらですか？",
        "table_q_profit": "4つの主要地域の営業利益合計はいくらですか？",
        "table_q_largest": "売上が最も高い主要地域はどこですか？",
        "table_q_budget": "記載された全運営費を差し引いた承認予算の残額はいくらですか？",
        "four_region_revenue": "4地域の売上",
        "four_region_cost": "4地域の運営費",
        "all_costs": "記載された全運営費",
        "table_title": "地域別運営レビュー",
        "table_budget": "承認済み運営予算",
        "table_h_region": "地域",
        "table_h_revenue": "売上",
        "table_h_cost": "運営費",
        "table_h_units": "数量",
        "table_region": "地域別運営表",
        "table_task": "表を再構成し、複数セルを使う運営上の質問に答えてください。",
        "chart_unit": "指数ポイント",
        "chart_q_latest": "{latest}年の指数値はいくつですか？",
        "chart_q_change": "{first}年から{latest}年までの指数の変化率は何パーセントですか？",
        "chart_q_peak": "グラフが最高値に達する年はいつですか？",
        "chart_q_mean": "最後の3年間の指数平均はいくつですか？",
        "chart_title": "供給レジリエンス指数",
        "chart_note": "年間総合スコア（高いほど良い）",
        "chart_task": "時系列計算の前に正確なラベルを読んでください。",
        "ownership_label": "{source}から{target}への持分",
        "investment_q_direct": "Aurora FundがBirch Holdingsに対して開示した直接持分は何パーセントですか？",
        "investment_q_path": "Aurora FundがBirch Holdingsを通じてDelta Labsに持つ間接持分は何パーセントですか？",
        "investment_q_total": "開示された2経路を通じたAurora FundのDelta Labsに対する総有効持分は何パーセントですか？",
        "investment_title": "実質的所有持分の開示",
        "investment_h_investor": "投資家",
        "investment_h_holding": "直接保有先",
        "investment_h_ownership": "持分率",
        "investment_region": "直接所有持分表",
        "investment_legal": "有効持分には開示されたすべての間接経路を含めます。直接持分率は経路上で乗算し、独立した経路間で加算します。",
        "investment_task": "法人を解決し、二重計上せずに間接的な実質持分を計算してください。",
        "condition_control": "対照群",
        "condition_a": "化合物A",
        "condition_b": "化合物B",
        "condition_ab": "A+B",
        "response_label": "{condition}の反応",
        "science_unit": "相対蛍光単位",
        "science_q_control": "対照群の平均反応値はいくつですか？",
        "science_q_reduction": "A+Bは対照群に比べて平均反応を何パーセント低下させましたか？",
        "science_q_lowest": "平均反応が最も低かった条件はどれですか？",
        "science_q_difference": "化合物Aと化合物Bの平均反応の差はいくつですか？",
        "science_title": "細胞ストレス反応の組合せ調節",
        "science_abstract": "<b>要旨。</b> 2つの化合物とその組合せを未処理対照群と比較しました。蛍光が低いほどストレスシグナルが低下したことを示します。すべての値は独立に生成した平均値と標準誤差です。",
        "science_equation": "相対効果 = (処理群 - 対照群) / |対照群| x 100",
        "science_h_condition": "条件",
        "science_h_mean": "平均反応",
        "science_h_se": "標準誤差",
        "science_h_n": "n",
        "science_region": "表1の実験結果",
        "science_caption": "表1．介入別の平均反応、標準誤差、反復数。",
        "science_task": "表1と対照群相対式に照らして定量的主張を検証してください。",
        "r_read": "{label}を直接読むと{value}です。",
        "r_add": "{values}を加えると{result}です。",
        "r_average": "平均は（{values}）/ {count} = {result}です。",
        "r_subtract": "{left_label}（{left}）から{right_label}（{right}）を引くと{result}です。",
        "r_divide": "{left}を{right}で割ると{result}です。",
        "r_reduction": "相対減少率 = ({right} - {left}) / {denominator} x 100 = {result}%です。",
        "r_change": "変化率 = ({left} - {right}) / {denominator} x 100 = {result}%です。",
        "r_extreme": "{label}の値{value}が{direction}です。",
        "largest": "最大",
        "smallest": "最小",
        "r_weighted": "加重和 = {terms} = {result}です。",
        "r_path": "経路の持分率を掛けると{weights} = {result}です。",
        "r_paths": "経路ごとの積を足すと（{paths}）= {result}です。",
    },
    "zh": {
        "concise": "请简洁作答，无需解释。",
        "fulltext": "请按阅读顺序转录此文档中的所有文字。",
        "region_north": "北部",
        "region_south": "南部",
        "region_east": "东部",
        "region_west": "西部",
        "region_aux": "辅助-{index}",
        "revenue": "收入",
        "cost": "运营成本",
        "units": "数量",
        "budget_label": "已批准运营预算",
        "table_q_north": "北部报告的收入是多少？",
        "table_q_core_revenue": "北部、南部、东部和西部的合计收入是多少？",
        "table_q_profit": "四个主要区域的总营业利润是多少？",
        "table_q_largest": "哪个主要区域的收入最高？",
        "table_q_budget": "扣除所有列出的运营成本后，已批准预算还剩多少？",
        "four_region_revenue": "四个区域的收入",
        "four_region_cost": "四个区域的运营成本",
        "all_costs": "列出的全部运营成本",
        "table_title": "区域运营审查",
        "table_budget": "已批准运营预算",
        "table_h_region": "区域",
        "table_h_revenue": "收入",
        "table_h_cost": "运营成本",
        "table_h_units": "数量",
        "table_region": "区域运营表",
        "table_task": "重建表格并回答需要多个单元格的运营问题。",
        "chart_unit": "指数点",
        "chart_q_latest": "{latest}年的指数值是多少？",
        "chart_q_change": "从{first}年到{latest}年，指数的百分比变化是多少？",
        "chart_q_peak": "图表在哪一年达到最高值？",
        "chart_q_mean": "最后三年的平均指数值是多少？",
        "chart_title": "供应韧性指数",
        "chart_note": "年度综合得分（越高越好）",
        "chart_task": "进行时间序列计算前，请先读取准确标签。",
        "ownership_label": "{source}对{target}的持股",
        "investment_q_direct": "Aurora Fund披露其在Birch Holdings的直接持股是多少？",
        "investment_q_path": "Aurora Fund通过Birch Holdings间接持有Delta Labs多少股权？",
        "investment_q_total": "通过披露的两条路径，Aurora Fund对Delta Labs的总有效持股是多少？",
        "investment_title": "实益所有权披露",
        "investment_h_investor": "投资者",
        "investment_h_holding": "直接持有对象",
        "investment_h_ownership": "持股比例",
        "investment_region": "直接持股明细表",
        "investment_legal": "有效持股必须包含所有已披露的间接路径。直接持股比例沿路径相乘，并在相互独立的路径之间相加。",
        "investment_task": "解析实体并计算间接实益所有权，避免重复计算。",
        "condition_control": "对照组",
        "condition_a": "化合物A",
        "condition_b": "化合物B",
        "condition_ab": "A+B",
        "response_label": "{condition}反应",
        "science_unit": "相对荧光单位",
        "science_q_control": "对照组报告的平均反应是多少？",
        "science_q_reduction": "与对照组相比，A+B使平均反应降低了百分之多少？",
        "science_q_lowest": "哪个条件产生了最低的平均反应？",
        "science_q_difference": "化合物A和化合物B的平均反应相差多少？",
        "science_title": "细胞应激反应的组合调节",
        "science_abstract": "<b>摘要。</b> 我们比较了两种化合物及其组合与未处理对照组。荧光越低表示应激信号越弱。所有数值均为独立生成的平均值和标准误。",
        "science_equation": "相对效应 = (处理组 - 对照组) / |对照组| x 100",
        "science_h_condition": "条件",
        "science_h_mean": "平均反应",
        "science_h_se": "标准误",
        "science_h_n": "n",
        "science_region": "表1实验结果",
        "science_caption": "表1。各干预条件的平均反应、标准误和重复次数。",
        "science_task": "根据表1和相对于对照组的公式验证定量结论。",
        "r_read": "直接读取{label}：{value}。",
        "r_add": "相加{values} = {result}。",
        "r_average": "平均值为（{values}）/ {count} = {result}。",
        "r_subtract": "用{left_label}（{left}）减去{right_label}（{right}）= {result}。",
        "r_divide": "{left}除以{right} = {result}。",
        "r_reduction": "相对降幅 = ({right} - {left}) / {denominator} x 100 = {result}%。",
        "r_change": "百分比变化 = ({left} - {right}) / {denominator} x 100 = {result}%。",
        "r_extreme": "{label}的值{value}为{direction}。",
        "largest": "最大值",
        "smallest": "最小值",
        "r_weighted": "加权和 = {terms} = {result}。",
        "r_path": "路径权重相乘{weights} = {result}。",
        "r_paths": "路径乘积求和（{paths}）= {result}。",
    },
}


def hard_text(language: str, key: str, **values: Any) -> str:
    """Return one validated hard-document locale string."""

    if language not in _CATALOG:
        raise ValueError(
            f"hard-document language must be one of {HARD_DOCUMENT_LANGUAGES}, "
            f"got {language!r}"
        )
    try:
        template = _CATALOG[language][key]
    except KeyError as exc:
        raise ValueError(f"missing hard-document locale key {key!r} for {language!r}") from exc
    return template.format(**values)


def validate_hard_locale_catalog() -> None:
    """Fail when a locale is missing a source key or introduces an unknown key."""

    source = set(_CATALOG["en"])
    for language in HARD_DOCUMENT_LANGUAGES:
        keys = set(_CATALOG[language])
        if keys != source:
            missing = sorted(source - keys)
            extra = sorted(keys - source)
            raise ValueError(
                f"hard-document locale {language!r} mismatch: "
                f"missing={missing}, extra={extra}"
            )


validate_hard_locale_catalog()


_SCRIPT_PATTERNS = {
    "ko": re.compile(r"[\uac00-\ud7a3]"),
    "ja": re.compile(r"[\u3040-\u30ff]"),
    "zh": re.compile(r"[\u4e00-\u9fff]"),
}


def validate_hard_document_language(record: dict[str, Any], language: str) -> None:
    """Fail closed when rendered hard-document content and language labels disagree."""

    if language not in HARD_DOCUMENT_LANGUAGES:
        raise ValueError(f"unsupported hard-document language {language!r}")
    if record.get("languages") != [language]:
        raise ValueError(
            f"hard-document languages must be [{language!r}], "
            f"got {record.get('languages')!r}"
        )
    field_languages = {
        str(field.get("language"))
        for field in record.get("fields_detailed", [])
    }
    if field_languages != {language}:
        raise ValueError(
            f"hard-document field language mismatch: {sorted(field_languages)}"
        )
    qa_languages = {
        tuple(str(value) for value in qa.get("languages", []))
        for qa in record.get("qa_detailed", [])
    }
    if qa_languages != {(language,)}:
        raise ValueError(
            f"hard-document QA language mismatch: {sorted(qa_languages)}"
        )
    graph = record.get("semantic_graph")
    graph_language = (graph or {}).get("language")
    if graph is not None and graph_language != language:
        raise ValueError(
            f"semantic graph language must be {language!r}, got {graph_language!r}"
        )
    full_text = str((record.get("fields") or {}).get("full_text") or "")
    questions = " ".join(
        str(qa.get("question") or "")
        for qa in record.get("qa_detailed", [])
    )
    if not full_text:
        raise ValueError("hard-document full-text render target is missing")
    pattern = _SCRIPT_PATTERNS.get(language)
    if pattern and (not pattern.search(full_text) or not pattern.search(questions)):
        raise ValueError(
            f"hard-document {language!r} content does not contain its expected script"
        )
    if language == "es" and "¿" not in questions:
        raise ValueError("Spanish hard-document questions are not localized")
