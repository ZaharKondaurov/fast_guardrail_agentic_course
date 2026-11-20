#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
if [[ ! -f .env ]]; then
  echo "No .env file found in the repository root." >&2
  exit 1
fi

set -a
source .env
set +a
if [[ -z "${OPENROUTER_API_KEY:-}" ]]; then
  echo "Set OPENROUTER_API_KEY in .env." >&2
  exit 1
fi

# Файл с ключом должен быть доступен только владельцу до завершения контейнера.
umask 077
access=$(mktemp "${TMPDIR:-/tmp}/ar12-model-access.XXXXXX.json")
trap 'rm -f -- "$access"' EXIT

.venv/bin/python - "$access" "${AR12_CAMPAIGN:-case/campaign/ar-12-e2e}/runtime.json" <<'PY'
import json
import os
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[2]).read_text())
Path(sys.argv[1]).write_text(json.dumps({
    "model": config["research_model"],
    "model_base_url": "https://openrouter.ai/api/v1",
    "api_key": os.environ["OPENROUTER_API_KEY"],
}) + "\n", encoding="utf-8")
PY

unset OPENROUTER_API_KEY
bash case/deploy/e2e.sh run "$access"
