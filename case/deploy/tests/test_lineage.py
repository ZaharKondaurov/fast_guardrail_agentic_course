"""Validate lineage relationships, safe SVG output and standalone export files."""

import xml.etree.ElementTree as ET

import pytest

from case.deploy.lineage import graph_layout, render_lineage, write_lineage


def candidate(name, parents=()):
    """Создаёт предложение с родителями и текстом, требующим HTML-экранирования."""
    return {"candidate_id": name, "parents": list(parents), "proposal": 1,
            "hypothesis": '<script>alert("test")</script>', "change": 'a & b', "digest": 'abc'}


def test_lineage_preserves_multiple_parents_and_independent_roots(tmp_path):
    """Проверяет обе связи ансамбля, корневые узлы и маркировку финалистов."""
    rows = [candidate('root-a'), candidate('root-b'), candidate('child', ['root-a']),
            candidate('ensemble', ['root-b', 'child'])]
    _, ranks, edges = graph_layout(rows)
    assert ranks == {'root-a': 0, 'root-b': 0, 'child': 1, 'ensemble': 2}
    assert set(edges) == {('root-a', 'child'), ('root-b', 'ensemble'), ('child', 'ensemble')}
    svg = render_lineage({'candidates': rows}, {'ensemble'})
    tree = ET.fromstring(svg)
    nodes = tree.findall('.//{*}g[@class="candidate"]')
    paths = tree.findall('.//{*}path[@class="edge"]')
    assert {node.attrib['data-id'] for node in nodes} == {row['candidate_id'] for row in rows}
    assert {(p.attrib['data-parent'], p.attrib['data-child']) for p in paths} == set(edges)
    assert len(tree.findall('.//{*}script')) == 0
    assert '&lt;script&gt;' in svg
    write_lineage({'candidates': rows}, {'candidates': [{'id': 'ensemble'}]}, tmp_path)
    assert {path.name for path in tmp_path.iterdir()} == {'lineage.svg'}
    assert (tmp_path / 'lineage.svg').read_text().strip() == svg


@pytest.mark.parametrize('rows,error', [
    ([candidate('a', ['missing'])], 'Unknown parent'),
    ([candidate('a', ['b']), candidate('b', ['a'])], 'Cycle'),
    ([candidate('a'), candidate('a')], 'Duplicate candidate'),
    ([candidate('a'), candidate('b', ['a', 'a'])], 'Duplicate parents'),
])
def test_invalid_lineage_is_not_silently_drawn(rows, error):
    """Отклоняет неполный или противоречивый граф вместо искажения родословной."""
    with pytest.raises(ValueError, match=error):
        graph_layout(rows)


def test_unknown_finalist_is_rejected():
    """Не рисует выдуманную отметку финалиста, отсутствующего в журнале."""
    with pytest.raises(ValueError, match='Unknown finalist'):
        render_lineage({'candidates': [candidate('a')]}, {'missing'})
