"""Check cumulative best-score curves against proposal indices, not completion times."""

import json
import xml.etree.ElementTree as ET

import pytest

from evaluator.search import Journal, Resources
from case.deploy.score_curves import collect_curves, write_score_curves

ZERO = Resources(total_tokens=0, seconds=0)
METRIC = {'version': 'ar12-f1-v1', 'direction': 'maximize'}


@pytest.fixture
def search(tmp_path):
    """Создаёт журнал и маленький снимок для проверки реальной кривой Arena."""
    source = tmp_path / 'candidate'
    source.mkdir()
    (source / 'solve.py').write_text('print(0)\n')
    journal = Journal.create(tmp_path / 'search.sqlite3', max_solutions=3)
    return journal, source


def propose(journal, source, name):
    """Регистрирует предложение без искусственного расхода ресурсов."""
    journal.propose(name, source, hypothesis='Test score curve', change='Recorded variant', resources=ZERO)


def measure(journal, candidate, score, split='train', dataset='dataset'):
    """Завершает замер при текущем числе предложений, сохраняя условия сравнения."""
    measurement = f'{candidate}-{split}-{dataset}'
    journal.begin(measurement, candidate, split=split, dataset_id=dataset,
                  subset_id='fixed', conditions_id='fixed')
    resources = {} if split == 'public' else {'solver': ZERO, 'judge': ZERO}
    journal.finish(measurement, score=score, **resources)


def test_step_auc_uses_cumulative_max_by_proposal_index(search):
    """Считает накопительный максимум и его ступенчатую площадь по индексам."""
    journal, source = search
    propose(journal, source, 'a')
    measure(journal, 'a', 0.4)
    propose(journal, source, 'b')
    measure(journal, 'b', 0.8)
    propose(journal, source, 'c')
    curve, = collect_curves(journal, METRIC)
    assert curve['integration'] == 'left_step'
    assert curve['auc'] == pytest.approx(1.2)
    assert curve['mean_quality'] == pytest.approx(0.4)
    assert curve['final_quality'] == 0.8


def test_late_public_is_placed_at_candidate_index_and_missing_scores_are_carried(search):
    """Поздний public предложения 1 относится к x=1; отсутствие других оценок не снижает max."""
    journal, source = search
    for name in ['a', 'b', 'c']:
        propose(journal, source, name)
    measure(journal, 'a', 0.95, split='public')
    curve, = collect_curves(journal, METRIC)
    assert curve['auc'] == pytest.approx(1.9)
    assert [p['quality'] for p in curve['points']] == [0, 0.95, 0.95, 0.95]
    assert [p['score'] for p in curve['points'][1:]] == [0.95, None, None]
    assert curve['axis'] == 'proposal_index'
    assert curve['retrospective'] is True
    assert curve['final_quality'] == 0.95
    assert curve['baseline_assumption']
    assert curve['svg'] == 'score-public.svg'


def test_completion_order_does_not_change_proposal_order_or_max(search):
    """Возвращает результат к индексу кандидата даже при обратном порядке готовности."""
    journal, source = search
    for name in ['a', 'b', 'c']:
        propose(journal, source, name)
    measure(journal, 'c', 0.7)
    measure(journal, 'b', 0.3)
    measure(journal, 'a', 0.8)
    curve, = collect_curves(journal, METRIC)
    assert [p['candidate_id'] for p in curve['points'][1:]] == ['a', 'b', 'c']
    assert [p['quality'] for p in curve['points'][1:]] == [0.8, 0.8, 0.8]
    assert curve['auc'] == pytest.approx(1.6)


def test_different_datasets_and_splits_are_not_combined(search):
    """Разделяет кривые при разных данных, даже если split совпадает."""
    journal, source = search
    propose(journal, source, 'a')
    measure(journal, 'a', 0.5, dataset='one')
    measure(journal, 'a', 0.6, dataset='two')
    measure(journal, 'a', 0.7, split='public', dataset='sealed')
    curves = collect_curves(journal, METRIC)
    assert len(curves) == 3
    assert len({c['svg'] for c in curves}) == 3
    assert all(c['baseline'] == 0 for c in curves)


def test_non_f1_metric_does_not_get_invented_zero_baseline(search):
    """Не навязывает другим метрикам F1-допущение о начальном нуле."""
    journal, source = search
    propose(journal, source, 'a')
    measure(journal, 'a', 10)
    curve, = collect_curves(journal, {'version': 'cost-v1', 'direction': 'minimize'})
    assert curve['baseline'] is None
    assert curve['auc'] is None


def test_writes_svg_and_machine_readable_auc(search, tmp_path):
    """Проверяет итоговые SVG и JSON в окружениях, которыми пользуется экспортёр."""
    journal, source = search
    propose(journal, source, 'a')
    measure(journal, 'a', 0.7)
    output = tmp_path / 'artifacts'
    curves = write_score_curves(journal, METRIC, output)
    assert curves[0]['final_quality'] == 0.7
    data = json.loads((output / 'results/score-curves.json').read_text())
    assert data['curves'][0]['auc'] == 0
    tree = ET.parse(output / 'score-train.svg')
    text = ''.join(tree.getroot().itertext())
    assert 'AUC = 0.000000' in text
    assert 'train' in text
    assert not list(output.glob('*.html'))
