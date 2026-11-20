"""Check campaign routing in shell launchers without Docker, secrets or model calls."""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.fixture
def installation(tmp_path):
    """Создаёт минимальную установку с заглушкой оценщика вместо реального Docker."""
    deploy = tmp_path / 'case/deploy'
    deploy.mkdir(parents=True)
    shutil.copyfile(Path(__file__).parents[1] / 'e2e.sh', deploy / 'e2e.sh')
    interpreter = tmp_path / '.venv/bin/python'
    interpreter.parent.mkdir(parents=True)
    interpreter.symlink_to(sys.executable)
    stub = '''import json, os, sys
from pathlib import Path
Path(os.environ['CAPTURE']).write_text(json.dumps(sys.argv[1:]))
'''
    (deploy / 'arena.py').write_text(stub)
    evaluator = tmp_path / 'arena/.venv/bin/python'
    evaluator.parent.mkdir(parents=True)
    evaluator.write_text(f'#!{sys.executable}\n' + stub)
    evaluator.chmod(0o755)
    campaign = tmp_path / 'case/campaign/ar-12-first'
    run = campaign / '.run'
    run.mkdir(parents=True)
    (run / 'search.sqlite3').touch()
    (run / 'runner-image.txt').write_text('test-image')
    closed = tmp_path / 'closed'
    closed.mkdir()
    (run / 'closed-dir').write_text(str(closed))
    (campaign / 'runtime.json').write_text(json.dumps({
        'data_dir': 'first-data', 'release_id': 'first-release',
        'max_solutions': 18, 'finalists': 3, 'output_count': 1,
    }))
    env = dict(os.environ, AR12_CAMPAIGN='case/campaign/ar-12-first',
               AR12_BIND_HOST='127.0.0.1', CAPTURE=str(tmp_path / 'capture.json'))
    return tmp_path, env


def test_prepare_uses_selected_campaign_limit(installation):
    """Не подменяет first-кампанию и её лимит значениями старого e2e."""
    root, env = installation
    subprocess.run(['bash', 'case/deploy/e2e.sh', 'prepare'], cwd=root, env=env, check=True)
    args = json.loads((root / 'capture.json').read_text())
    assert args == ['--campaign', str(root / 'case/campaign/ar-12-first'),
                    'prepare-pilot', '--max-solutions', '18']


@pytest.mark.parametrize('split,port', [('public', '8767'), ('private', '8766')])
def test_closed_evaluators_use_same_campaign_conditions(installation, split, port):
    """Передаёт обоим закрытым оценщикам лимит, финалистов, release и метрику first."""
    root, env = installation
    subprocess.run(['bash', 'case/deploy/e2e.sh', split], cwd=root, env=env, check=True)
    args = json.loads((root / 'capture.json').read_text())
    for option, expected in [('--max-solutions', '18'), ('--finalists', '3'),
                             ('--winners', '1'), ('--release-id', 'first-release'),
                             ('--port', port), ('--dataset', str(root / f'closed/{split}.jsonl')),
                             ('--metric', str(root / 'case/campaign/ar-12-first/metric'))]:
        assert args[args.index(option) + 1] == expected
