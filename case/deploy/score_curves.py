"""Export retrospective cumulative best scores by proposal index as SVG and JSON."""

import argparse
from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[2]


def proposal_curve(exported: dict, comparison: dict, direction: str, baseline: float | None) -> dict:
    """Считает max/min по индексам предложений, используя последний успешный замер снимка."""
    measurements = sorted(
        (row for row in exported['measurements'] if row['comparison'] == comparison
         and row['state'] == 'succeeded' and row['score'] is not None),
        key=lambda row: row['completion_order'],
    )
    latest = {row['candidate_id']: row for row in measurements}
    points = [] if baseline is None else [{'proposals': 0, 'quality': baseline, 'source': 'baseline'}]
    quality = baseline
    for candidate in sorted(exported['candidates'], key=lambda row: row['proposal']):
        measurement = latest.get(candidate['candidate_id'])
        score = measurement['score'] if measurement else None
        if score is not None:
            quality = score if quality is None else (
                max(quality, score) if direction == 'maximize' else min(quality, score)
            )
        points.append({
            'proposals': candidate['proposal'], 'candidate_id': candidate['candidate_id'],
            'score': score, 'quality': quality,
            'measurement_id': measurement['measurement_id'] if measurement else None,
            'source': 'measurement' if measurement else 'unmeasured',
        })
    # Оставляем ступенчатый интеграл по [0, N]: значение в i действует на [i, i+1).
    area = math.fsum((b['proposals'] - a['proposals']) * a['quality']
                     for a, b in zip(points, points[1:]) if a['quality'] is not None)
    observed = next((p['proposals'] for p in points if p['quality'] is not None), None)
    horizon = exported['campaign']['proposed']
    auc = area if observed == 0 else None
    return {
        'comparison': comparison, 'direction': direction, 'baseline': baseline,
        'axis': 'proposal_index', 'retrospective': True, 'integration': 'left_step',
        'measurement_selection': 'latest_successful_per_candidate',
        'horizon': horizon, 'actual_proposals': horizon, 'points': points,
        'final_quality': quality, 'auc': auc,
        'mean_quality': auc / horizon if auc is not None and horizon else None,
        'observed_auc': area if observed is not None else None,
        'observed_from': observed, 'covered_proposals': horizon - observed if observed is not None else 0,
    }


def collect_curves(journal, metric: dict) -> list[dict]:
    """Разделяет ретроспективные кривые по сопоставимым train/public-условиям."""
    exported = journal.export()
    comparisons = {
        tuple(sorted(row['comparison'].items())) for row in exported['measurements']
        if row['comparison']['split'] in {'train', 'public'}
    }
    baseline = 0.0 if metric['version'] == 'ar12-f1-v1' and metric['direction'] == 'maximize' else None
    counts = Counter(dict(key)['split'] for key in comparisons)
    curves = []
    for key in sorted(comparisons):
        comparison = dict(key)
        curve = proposal_curve(exported, comparison, metric['direction'], baseline)
        suffix = '' if counts[comparison['split']] == 1 else '-' + hashlib.sha256(
            json.dumps(comparison, sort_keys=True).encode()
        ).hexdigest()[:10]
        curve['svg'] = f"score-{comparison['split']}{suffix}.svg"
        curve['metric_version'] = metric['version']
        curve['baseline_assumption'] = (
            'До первого оценённого предложения принято F1=0; это допущение, а не измеренный score.'
            if baseline is not None else None
        )
        curves.append(curve)
    return curves


def render_curves(curves: list[dict], output: Path) -> None:
    """Рисует ступенчатый лучший известный score и его интеграл без смешивания условий."""
    os.environ.setdefault('MPLCONFIGDIR', str(ROOT / '.venv/.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    for curve in curves:
        split = curve['comparison']['split']
        horizon = curve['horizon']
        points = curve['points']
        fig, ax = plt.subplots(figsize=(10.5, 6.4))
        fig.subplots_adjust(left=0.09, right=0.97, bottom=0.29, top=0.84)
        observed_points = [p for p in points if p['quality'] is not None]
        xs = [p['proposals'] for p in observed_points]
        ys = [p['quality'] for p in observed_points]
        if observed_points:
            ax.step(xs, ys, where='post', color='#2459a6', linewidth=2,
                    label='max(prev_score, ind_score)' if curve['direction'] == 'maximize'
                    else 'min(prev_score, ind_score)')
            ax.fill_between(xs, ys, 0, step='post', color='#2459a6', alpha=0.12)
            measured = [p for p in points if p.get('score') is not None]
            ax.scatter([p['proposals'] for p in measured], [p['score'] for p in measured],
                       color='#bf6c24', s=32, zorder=3, label='Score предложения')
        else:
            ax.text(0.5, 0.5, 'Завершённых результатов нет', transform=ax.transAxes, ha='center')
        ax.set_xlim(0, max(1, horizon))
        if curve['metric_version'] == 'ar12-f1-v1':
            ax.set_ylim(0, 1.04)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10))
        ax.set_xlabel('Порядковый номер предложения')
        ax.set_ylabel('Score (F1)' if curve['metric_version'] == 'ar12-f1-v1' else 'Score')
        ax.grid(alpha=0.22)
        ax.spines[['top', 'right']].set_visible(False)
        if observed_points:
            ax.legend(loc='lower right', frameon=False)
        auc = curve['auc']
        mean = curve['mean_quality']
        statistic = (
            f'AUC = {auc:.6f}   ·   AUC / {horizon} = {mean:.6f}'
            if auc is not None and mean is not None else
            f"AUC на наблюдаемом интервале = {curve['observed_auc']}"
        )
        if curve['final_quality'] is not None:
            statistic += f"   ·   Финальный score = {curve['final_quality']:.6f}"
        fig.suptitle(f'Накопительный лучший score · {split}', x=0.09, ha='left', fontsize=16)
        fig.text(0.09, 0.88, statistic, fontsize=12)
        comparison = curve['comparison']
        notes = [
            f"Данные: {comparison['dataset_id']}",
            f"Условия: {comparison['conditions_id']} · метрика: {curve['metric_version']}",
            'Ретроспективно по индексам: время получения оценок не учитывается.',
            'AUC: ступенчатый интеграл по [0, N]; значение в i действует на [i, i+1).',
        ]
        if curve['baseline_assumption']:
            notes.append(curve['baseline_assumption'])
        fig.text(0.09, 0.035, '\n'.join(notes), fontsize=9, linespacing=1.6, color='#465361')
        # Текст остаётся текстом в векторном файле; SVG открывается без Python и сети.
        with matplotlib.rc_context({'svg.fonttype': 'none'}):
            fig.savefig(output / curve['svg'], format='svg')
        plt.close(fig)


def write_score_curves(journal, metric: dict, output: Path) -> list[dict]:
    """Экспортирует точки/AUC и строит SVG через окружение обучения с matplotlib."""
    curves = collect_curves(journal, metric)
    data = output / 'results/score-curves.json'
    data.parent.mkdir(parents=True, exist_ok=True)
    data.write_text(json.dumps({'curves': curves}, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    if curves:
        # Arena-окружение минимально; matplotlib уже входит в зависимости обучения CatBoost.
        subprocess.run([str(ROOT / '.venv/bin/python'), '-m', 'case.deploy.score_curves',
                        '--render-data', str(data.resolve()), '--output', str(output.resolve())],
                       cwd=ROOT, check=True)
    return curves


def main() -> None:
    """Обновляет графики артефактов либо выполняет отрисовку готовых точек без Arena."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path)
    parser.add_argument('--render-data', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.render_data:
        if args.output is None:
            parser.error('--render-data requires --output')
        render_curves(json.loads(args.render_data.read_text())['curves'], args.output)
    elif args.campaign:
        from evaluator.search import Journal
        campaign = args.campaign.resolve()
        output = args.output.resolve() if args.output else campaign / 'artifacts'
        metric = json.loads((campaign / 'metric/metric.json').read_text())
        curves = write_score_curves(Journal(campaign / '.run/search.sqlite3'), metric, output)
        for curve in curves:
            print(curve['svg'], 'AUC:', curve['auc'], 'AUC/horizon:', curve['mean_quality'])
    else:
        parser.error('Specify --campaign or --render-data')


if __name__ == '__main__':
    main()
