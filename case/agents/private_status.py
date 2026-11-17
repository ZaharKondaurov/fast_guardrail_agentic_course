"""Consume the private evaluator's status-only API."""

from pathlib import Path

from evaluator.client import ClientError, request


def private_status(url: str, token_file: Path) -> str | None:
    try:
        result = request("GET", url, token_file, timeout=60)
    except ClientError as error:
        if error.status_code == 404 and error.error_code == "no_evaluation":
            return None
        raise ValueError("Private status unavailable") from error
    status = result.get("status")
    if set(result) != {"status"} or status not in ("running", "complete", "failed"):
        raise ValueError("Unexpected private evaluator response")
    return status


def wait_private(status, search, read_status, sleep):
    if status is None:
        return False
    if search["running"] or not search["finalization"] or (
        search["remaining_proposals"] and not search["stopped"]
    ):
        raise ValueError("Private evaluation exists before valid search finalization")
    while status == "running":
        sleep(60)
        status = read_status()
    if status != "complete":
        raise ValueError(f"Private evaluation is {status or 'unavailable'}; search incomplete")
    return True
