"""Stage a current-campaign workspace and launch a constrained Hermes container."""

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]


def copy_tree(source, destination):
    for path in source.rglob('*'):
        if any(part in {'.git', '.run', '__pycache__', '.venv'} for part in path.relative_to(source).parts):
            continue
        if path.is_symlink():
            raise ValueError(f'Symlink is not allowed in agent inputs: {path}')
        if path.is_file():
            target = destination / path.relative_to(source)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


def stage(campaign, train, destination, evaluation=None):
    campaign = campaign.resolve()
    if destination.exists():
        raise ValueError('Workspace already exists; retain it for resume')
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=destination.parent) as temporary:
        workspace = Path(temporary) / 'workspace'
        workspace.mkdir()
        for name in ('common', 'prompts', 'evaluator'):
            copy_tree(ROOT / 'arena' / name, workspace / 'arena' / name)
        for name in ('README.md', 'AGENTS.md'):
            shutil.copyfile(ROOT / 'arena' / name, workspace / 'arena' / name)
        for name in ('agents', 'ways/baseline', 'metric'):
            copy_tree(ROOT / 'case' / name, workspace / 'case' / name)
        shutil.copyfile(ROOT / 'case/README.md', workspace / 'case/README.md')
        target = workspace / 'campaign'
        target.mkdir()
        for name in ('campaign.md', 'preparation.md', 'runtime.json'):
            shutil.copyfile(campaign / name, target / name)
        copy_tree(campaign / 'metric', target / 'metric')
        readiness = campaign / '.run/preparation.json'
        if readiness.is_file():
            (target / '.run').mkdir()
            shutil.copyfile(readiness, target / '.run/preparation.json')
        (workspace / 'data').mkdir()
        shutil.copyfile(train, workspace / 'data/train.jsonl')
        if evaluation is not None:
            shutil.copyfile(evaluation, workspace / 'data/evaluation.arena.jsonl')
        baselines = campaign / '.run/candidates'
        if baselines.is_dir():
            copy_tree(baselines, workspace / 'baselines')
        workspace.rename(destination)
    return destination


def docker_command(args):
    workspace = args.workspace.resolve(strict=True)
    credentials = args.credentials.resolve(strict=True)
    journal = args.journal_dir.resolve(strict=True)
    config = args.config.resolve(strict=True) if args.config else None
    if args.private_config and config is None:
        raise ValueError('--private-config requires --config')
    if args.network in {'host', 'bridge', 'none'}:
        raise ValueError('Use a dedicated operator-configured network with permitted API routes')
    if any(',' in str(path) for path in (workspace, credentials, journal, config) if path is not None):
        raise ValueError('Docker mount paths must not contain commas')
    if bool(args.public_url) != bool(args.public_token):
        raise ValueError('Set --public-url and --public-token together')
    command = [
        'docker', 'run', '--rm', '--init', '--name', args.name,
        '--user', f'{os.getuid()}:{os.getgid()}', '--read-only', '--cap-drop', 'ALL',
        '--security-opt', 'no-new-privileges', '--pids-limit', '512',
        '--memory', args.memory, '--cpus', args.cpus, '--network', args.network,
        '--tmpfs', '/tmp:rw,nosuid,size=1g,mode=1777',
        '--mount', f'type=bind,src={workspace},dst=/workspace',
        '--mount', f'type=bind,src={credentials},dst=/run/arena/model-access.json,readonly',
        '--mount', f'type=bind,src={journal},dst=/search',
        '--env', 'PYTHONPATH=/workspace:/workspace/arena:/opt/arena:/opt/hermes',
        '--env', 'HOME=/tmp/agent-home', '--env', 'HERMES_HOME=/tmp/hermes-home',
        '--workdir', '/workspace', '--entrypoint', '/opt/hermes/.venv/bin/python', args.image,
        '-m', 'case.agents.run', args.role, '--campaign', '/workspace/campaign',
        '--journal', '/search/search.sqlite3',
        '--credentials-file', '/run/arena/model-access.json',
    ]
    if config is not None:
        position = command.index('--workdir')
        command[position:position] = ['--mount', f'type=bind,src={config},dst=/run/arena/campaign.json,readonly']
    for role in ('train', 'public', 'private'):
        token = getattr(args, f'{role}_token')
        if token is not None:
            token = token.resolve(strict=True)
            if ',' in str(token):
                raise ValueError('Docker mount paths must not contain commas')
            position = command.index('--workdir')
            command[position:position] = ['--mount', f'type=bind,src={token},dst=/run/arena/{role}-token,readonly']
    if args.train_url:
        command.extend(['--train-url', args.train_url])
    if args.public_url:
        command.extend(['--public-url', args.public_url])
    if args.assignment:
        command.extend(['--assignment', args.assignment])
    if args.private_config:
        command.extend(['--private-config', '/run/arena/campaign.json'])
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('stage')
    prepare.add_argument('--campaign', type=Path, required=True)
    prepare.add_argument('--train', type=Path, required=True)
    prepare.add_argument('--evaluation', type=Path)
    prepare.add_argument('--workspace', type=Path, required=True)
    launch = commands.add_parser('run')
    for name in ('workspace', 'credentials', 'journal-dir'):
        launch.add_argument(f'--{name}', type=Path, required=True)
    launch.add_argument('--config', type=Path)
    launch.add_argument('--role', choices=['prepare', 'orchestrator', 'researcher', 'reviewer'], default='orchestrator')
    launch.add_argument('--private-config', action='store_true')
    launch.add_argument('--train-url')
    launch.add_argument('--assignment')
    launch.add_argument('--train-token', type=Path)
    launch.add_argument('--public-url')
    launch.add_argument('--public-token', type=Path)
    launch.add_argument('--private-token', type=Path)
    launch.add_argument('--image', required=True)
    launch.add_argument('--network', required=True)
    launch.add_argument('--name', default='ar12-agents')
    launch.add_argument('--memory', default='8g')
    launch.add_argument('--cpus', default='4')
    launch.add_argument('--print-command', action='store_true')
    args = parser.parse_args()
    try:
        if args.command == 'stage':
            print(stage(args.campaign, args.train, args.workspace, args.evaluation))
        elif args.print_command:
            print(json.dumps(docker_command(args)))
        else:
            subprocess.run(docker_command(args), check=True)
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        parser.exit(1, f'{error}\n')


if __name__ == '__main__':
    main()
