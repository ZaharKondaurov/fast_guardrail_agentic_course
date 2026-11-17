"""Export shareable AR-12 campaign artifacts without copying operator state."""

import argparse
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZIP_DEFLATED, ZipFile

from evaluator.search import Journal


def copy_artifact(source: Path, destination: Path) -> None:
    """Copy a regular artifact while rejecting symlinks and external paths."""
    if source.resolve() != source.absolute() or not source.is_file():
        raise ValueError(f"Expected a regular artifact: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, destination)


def copy_directory(source: Path, destination: Path) -> None:
    """Preserve the contents and relative layout of a shareable directory."""
    for path in sorted(source.rglob("*")):
        relative = path.relative_to(source)
        if "arena" in relative.parts or "__pycache__" in relative.parts:
            continue
        if path.is_symlink():
            raise ValueError(f"Unexpected symlink in artifacts: {path}")
        if path.is_file():
            copy_artifact(path, destination / relative)


def write_snapshots(journal: Journal, archive_path: Path, scratch: Path) -> None:
    """Restore journal snapshots and package all candidates with their file modes."""
    candidates = journal.export()["candidates"]
    with ZipFile(archive_path, "x", compression=ZIP_DEFLATED) as archive:
        for index, candidate in enumerate(candidates):
            identifier = candidate["candidate_id"]
            if Path(identifier).name != identifier or identifier in {"", ".", ".."}:
                raise ValueError(f"Invalid snapshot directory name: {identifier!r}")
            snapshot = scratch / str(index)
            journal.restore(identifier, snapshot)
            for file in sorted(snapshot.rglob("*")):
                archive.write(file, Path(identifier) / file.relative_to(snapshot))


def export_campaign(campaign: Path, output: Path) -> None:
    """Publish Markdown, board, results, finalists and snapshots as one directory."""
    run = campaign / ".run"
    workspace = run / "agent-workspace"
    agent_campaign = workspace / "campaign"
    journal = Journal(run / "search.sqlite3")
    status = journal.status()
    if status["running"] or not status["finalization"]:
        raise ValueError("Expected a finalized campaign with no running measurements")
    if output.exists():
        raise FileExistsError(output)

    with TemporaryDirectory(dir=output.parent) as temporary:
        scratch = Path(temporary)
        staged = scratch / "artifacts"
        staged.mkdir()
        for directory in ("board", "results"):
            if not (agent_campaign / directory).is_dir():
                raise ValueError(f"Missing campaign artifact directory: {directory}")
            copy_directory(agent_campaign / directory, staged / directory)
        copy_artifact(agent_campaign / "finalists.json", staged / "finalists.json")

        # Skip the copied Arena and case sources in the agent workspace.
        for file in sorted(run.rglob("*.md")):
            relative = file.relative_to(run)
            if any(part in {"arena", ".venv", ".git"} for part in relative.parts):
                continue
            if relative.parts[:2] == ("agent-workspace", "case"):
                continue
            if relative.parts[:2] == ("agent-workspace", "campaign"):
                target = staged.joinpath(*relative.parts[2:])
            else:
                target = staged / "notes" / relative
            if not target.exists():
                copy_artifact(file, target)

        for file in sorted((agent_campaign / "work/public").glob("*/status.json")):
            copy_artifact(file, staged / "results/public" / f"{file.parent.name}.json")
        write_snapshots(journal, staged / "snapshots.zip", scratch)
        staged.rename(output)


def main() -> None:
    """Export a finalized campaign to a new directory under its root by default."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    campaign = args.campaign.resolve()
    output = args.output.resolve() if args.output else campaign / "artifacts"
    try:
        export_campaign(campaign, output)
    except (OSError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    print(output)


if __name__ == "__main__":
    main()
