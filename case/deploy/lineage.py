"""Render an offline lineage graph from the campaign's exported public journal."""

import argparse
from collections import defaultdict
from html import escape
import json
from pathlib import Path


def graph_layout(candidates: list[dict]) -> tuple[dict, dict, list]:
    """Размещает DAG по поколениям, проверяя родителей и отсутствие циклов."""
    nodes = {row["candidate_id"]: row for row in candidates}
    if len(nodes) != len(candidates):
        raise ValueError("Duplicate candidate IDs in lineage")
    ranks = {}
    remaining = list(nodes)
    edges = []
    for row in candidates:
        parents = row.get("parents", [])
        if len(set(parents)) != len(parents):
            raise ValueError("Duplicate parents in lineage")
        for parent in parents:
            if parent not in nodes:
                raise ValueError(f"Unknown parent: {parent}")
            edges.append((parent, row["candidate_id"]))
    while remaining:
        ready = [key for key in remaining if all(p in ranks for p in nodes[key].get("parents", []))]
        if not ready:
            raise ValueError("Cycle in lineage")
        for key in ready:
            ranks[key] = max((ranks[p] + 1 for p in nodes[key].get("parents", [])), default=0)
            remaining.remove(key)
    return nodes, ranks, edges


def render_lineage(journal: dict, finalist_ids: set[str]) -> str:
    """Рисует SVG всех предложений; выделяет финалистов без private-оценок."""
    nodes, ranks, edges = graph_layout(journal["candidates"])
    if not finalist_ids <= nodes.keys():
        raise ValueError("Unknown finalist in lineage")
    columns = defaultdict(list)
    for key in nodes:
        columns[ranks[key]].append(key)
    width = max(270, max((len(key) for key in nodes), default=0) * 8 + 32)
    step, gap = 108, 115
    long_edges = [(a, b) for a, b in edges if ranks[b] > ranks[a] + 1]
    top = 145 + len(long_edges) * 12
    height = top + max((len(c) for c in columns.values()), default=1) * step + 30
    canvas_width = 64 + (max(ranks.values(), default=0) + 1) * (width + gap) - gap
    positions = {}
    for rank in sorted(columns):
        # Близкие родители группируют потомков, уменьшая пересечения соседних рёбер.
        columns[rank].sort(key=lambda key: (
            sum(positions[p][1] for p in nodes[key].get("parents", []))
            / max(1, len(nodes[key].get("parents", []))),
            nodes[key].get("proposal", 0), key,
        ))
        offset = (height - top - 30 - len(columns[rank]) * step) / 2
        for index, key in enumerate(columns[rank]):
            positions[key] = (32 + rank * (width + gap), top + offset + index * step)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {canvas_width} {height}" '
        f'width="{canvas_width}" height="{height}" role="img" aria-labelledby="title desc">',
        '<title id="title">Родословная решений кампании</title>',
        '<desc id="desc">Стрелки направлены от родителей к потомкам. '
        'Синим отмечены финалисты. Поколение вычислено по связям родителей.</desc>',
        '<defs><marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" '
        'markerWidth="6" markerHeight="6" orient="auto-start-reverse">'
        '<path d="M 0 0 L 10 5 L 0 10 z" fill="#718096"/></marker></defs>',
        '<rect width="100%" height="100%" fill="#fff"/>',
        '<g font-family="system-ui, sans-serif" fill="#17212b">',
        '<text x="32" y="37" font-size="24" font-weight="600">Родословная решений</text>',
        f'<text x="32" y="64" font-size="14" fill="#52606d">'
        f'{len(nodes)} предложений · {len(edges)} связей · {len(finalist_ids)} финалиста</text>',
        '<rect x="32" y="79" width="12" height="12" fill="#e8f0ff" stroke="#315fc4"/>',
        '<text x="52" y="90" font-size="13">Финалист · стрелка: родитель → потомок</text>',
    ]
    for rank in sorted(columns):
        x = 32 + rank * (width + gap)
        parts.append(f'<text x="{x}" y="{top-22}" font-size="13" fill="#52606d">'
                     f'Поколение {rank}</text>')
    lane = 0
    for parent, child in edges:
        x1, y1 = positions[parent]
        x2, y2 = positions[child]
        x1, y1, y2 = x1 + width, y1 + 40, y2 + 40
        if ranks[child] == ranks[parent] + 1:
            middle = (x1 + x2) / 2
            path = f'M {x1} {y1} C {middle} {y1}, {middle} {y2}, {x2} {y2}'
        else:
            # Длинные связи обходим сверху, чтобы они не проходили сквозь другие узлы.
            rail = 111 + lane * 12
            lane += 1
            start, end = x1 + 22 + lane, x2 - 22 - lane
            path = f'M {x1} {y1} H {start} V {rail} H {end} V {y2} H {x2}'
        parts.append(f'<path class="edge" data-parent="{escape(parent, quote=True)}" '
                     f'data-child="{escape(child, quote=True)}" d="{path}" fill="none" '
                     'stroke="#8996a5" stroke-width="1.4" marker-end="url(#arrow)">'
                     f'<title>{escape(parent)} → {escape(child)}</title></path>')
    for key, row in nodes.items():
        x, y = positions[key]
        finalist = key in finalist_ids
        fill, stroke = ('#e8f0ff', '#315fc4') if finalist else ('#f7f9fb', '#b6c0cb')
        parents = row.get("parents", [])
        note = f"Предложение {row.get('proposal', '—')} · "
        note += 'финалист' if finalist else ('без родителей' if not parents else f'родителей: {len(parents)}')
        detail = (f"{key}\nРодители: {', '.join(parents) or 'нет'}\n"
                  f"Гипотеза: {row.get('hypothesis', '')}\nИзменения: {row.get('change', '')}\n"
                  f"Снимок: {row.get('digest', '')}")
        parts.extend([
            f'<g class="candidate" data-id="{escape(key, quote=True)}">',
            f'<title>{escape(detail)}</title>',
            f'<rect x="{x}" y="{y}" width="{width}" height="80" rx="8" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="{2 if finalist else 1}"/>',
            f'<text x="{x+14}" y="{y+31}" font-family="monospace" font-size="13">{escape(key)}</text>',
            f'<text x="{x+14}" y="{y+57}" font-size="12" fill="#52606d">{escape(note)}</text>',
            '</g>',
        ])
    parts.append('</g></svg>')
    return '\n'.join(parts)


def write_lineage(journal: dict, finalists: dict, output: Path) -> None:
    """Публикует SVG-граф рядом с остальными артефактами кампании."""
    svg = render_lineage(journal, {row['id'] for row in finalists['candidates']})
    (output / 'lineage.svg').write_text(svg + '\n', encoding='utf-8')


def main() -> None:
    """Обновляет только визуализацию по уже экспортированным артефактам кампании."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--artifacts', type=Path, required=True)
    args = parser.parse_args()
    journal = json.loads((args.artifacts / 'results/journal-export.json').read_text())
    finalists = json.loads((args.artifacts / 'finalists.json').read_text())
    write_lineage(journal, finalists, args.artifacts)
    print(args.artifacts / 'lineage.svg')


if __name__ == '__main__':
    main()
