#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
campaign=$(realpath -e -- "${AR12_CAMPAIGN:-case/campaign/ar-12-e2e}")
run="$campaign/.run"
arena_python="$PWD/arena/.venv/bin/python"
export PYTHONPATH="$PWD/arena"
network=ar12-research
closed_pointer="$run/closed-dir"

# Условия берём из выбранной кампании; значения e2e остаются совместимыми по умолчанию.
runtime_values=$(.venv/bin/python - "$campaign/runtime.json" <<'PYCONFIG'
import json
import sys
from pathlib import Path

config = json.loads(Path(sys.argv[1]).read_text())
for key, default in (
    ("data_dir", None), ("closed_source_dir", "case/dataset/ar12-test"),
    ("release_id", None), ("max_solutions", 6), ("finalists", 2), ("output_count", 1),
):
    print(config[key] if default is None else config.get(key, default))
PYCONFIG
)
mapfile -t runtime <<< "$runtime_values"
data="$PWD/${runtime[0]}"
closed_source="$PWD/${runtime[1]}"
release_id=${runtime[2]}
max_solutions=${runtime[3]}
finalists=${runtime[4]}
winners=${runtime[5]}

case "${1:-}" in
  prepare|setup|train|public|private|run) action=$1 ;;
  *) echo "Usage: bash case/deploy/e2e.sh {prepare|setup|train|public|private|run MODEL_ACCESS_JSON}" >&2; exit 2 ;;
esac

if [[ "$action" == prepare ]]; then
  exec .venv/bin/python case/deploy/arena.py --campaign "$campaign" \
    prepare-pilot --max-solutions "$max_solutions"
fi

if [[ ! -f "$run/search.sqlite3" ]]; then
  echo "Prepare the selected campaign with e2e.sh prepare before running this command." >&2
  exit 1
fi

if [[ "$action" == setup ]]; then
  umask 077
  if ! docker network inspect "$network" >/dev/null 2>&1; then
    docker network create "$network" >/dev/null
  fi
  # Сохраняем каталог серии, чтобы при повторном запуске использовать прежнее состояние private.
  if [[ ! -f "$closed_pointer" ]]; then
    closed=${AR12_CLOSED_DIR:-}
    if [[ -z "$closed" ]]; then
      closed=$(mktemp -d "${XDG_STATE_HOME:-$HOME/.local/state}/${release_id}.XXXXXX" 2>/dev/null) ||
        closed=$(mktemp -d /tmp/${release_id}.XXXXXX)
    fi
    closed=$(realpath -e -- "$closed")
    if [[ ! -d "$closed" || "$closed" == "$PWD" || "$closed" == "$PWD/"* ]]; then
      echo "The closed directory must exist outside the repository." >&2
      exit 1
    fi
    (set -o noclobber; printf '%s\n' "$closed" > "$closed_pointer")
  fi
  closed=$(<"$closed_pointer")
  for split in public private; do
    if [[ ! -f "$closed/$split.jsonl" ]]; then
      .venv/bin/python case/dataset/export_arena.py \
        "$closed_source/ar12.$split.jsonl" "$closed/$split.jsonl"
    fi
  done
fi

if [[ ! -f "$closed_pointer" ]]; then
  echo "Run bash case/deploy/e2e.sh setup first." >&2
  exit 1
fi
closed=$(<"$closed_pointer")
if [[ -f "$campaign/closed-arena.lock.json" ]]; then
  .venv/bin/python - "$campaign/closed-arena.lock.json" "$closed" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

lock, directory = map(Path, sys.argv[1:])
for name, expected in json.loads(lock.read_text()).items():
    path = directory / name
    if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
        raise SystemExit(f"Frozen closed export is missing or changed: {path}")
PY
fi
bind_host=${AR12_BIND_HOST:-}
if [[ -z "$bind_host" ]]; then
  bind_host=$(docker network inspect "$network" --format '{{(index .IPAM.Config 0).Gateway}}')
fi
if [[ -z "$bind_host" ]]; then
  echo "No network gateway found. Set AR12_BIND_HOST to an address on the evaluator host." >&2
  exit 1
fi

case "$action" in
  setup) printf 'Closed state: %s\nEvaluator bind address: %s\n' "$closed" "$bind_host" ;;
  train) exec .venv/bin/python case/deploy/arena.py --campaign "$campaign" serve --host "$bind_host" ;;
  run)
    umask 077
    if [[ $# != 2 || ! -f "$2" ]]; then
      echo "Usage: bash case/deploy/e2e.sh run /path/to/model-access.json" >&2
      exit 2
    fi
    credentials=$(realpath -e -- "$2")
    eval_host=${AR12_EVAL_HOST:-$bind_host}
    train_url="http://$eval_host:8765"
    public_url="http://$eval_host:8767"
    private_url="http://$eval_host:8766"
    for token in "$run/submit-token" "$closed/public/submit-token" "$closed/private/submit-token"; do
      if [[ ! -s "$token" ]]; then
        echo "Missing submit token: $token. Start all three evaluators first." >&2
        exit 1
      fi
    done
    docker run --rm --network "$network" --entrypoint /bin/sh ar12-hermes:configured -ec '
      for url do curl -fsS --max-time 5 "$url/healthz"; printf "\n"; done
    ' sh "$train_url" "$public_url" "$private_url"
    # Проверяем модель из той же сети и под тем же UID, что и Hermes.
    docker run --rm -i --network "$network" --user "$(id -u):$(id -g)" \
      --mount "type=bind,src=$credentials,dst=/run/arena/model-access.json,readonly" \
      --entrypoint /opt/hermes/.venv/bin/python ar12-hermes:configured - <<'PY'
import json
from pathlib import Path

import openai

access = json.loads(Path("/run/arena/model-access.json").read_text())
try:
    with openai.OpenAI(base_url=access["model_base_url"], api_key=access.get("api_key", "arena"),
                       max_retries=0, timeout=60) as client:
        response = client.chat.completions.create(
            model=access["model"], messages=[{"role": "user", "content": "Reply READY."}],
        )
except openai.APIError as error:
    raise SystemExit(f"Model gateway smoke failed: {type(error).__name__}") from None
if response.usage is None or response.usage.total_tokens <= 0:
    raise SystemExit("Model gateway smoke did not return token usage.")
print(f"Model gateway smoke passed: {response.usage.total_tokens} tokens.")
PY
    workspace="$run/agent-workspace"
    if [[ ! -d "$workspace" ]]; then
      .venv/bin/python case/deploy/agents.py stage --campaign "$campaign" \
        --train "$data/train.jsonl" --evaluation "$data/evaluation.arena.jsonl" \
        --workspace "$workspace"
    fi
    private_config="$closed/agent-access.json"
    .venv/bin/python - "$private_config" "$private_url" "$finalists" "$winners" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "url": sys.argv[2], "token_file": "/run/arena/private-token",
    "finalists": int(sys.argv[3]), "output_count": int(sys.argv[4]),
}) + "\n", encoding="utf-8")
PY
    exec .venv/bin/python case/deploy/agents.py run --workspace "$workspace" \
      --credentials "$credentials" --journal-dir "$run" \
      --image ar12-hermes:configured --network "$network" \
      --train-url "$train_url" --train-token "$run/submit-token" \
      --public-url "$public_url" --public-token "$closed/public/submit-token" \
      --private-config --config "$private_config" --private-token "$closed/private/submit-token"
    ;;
  public|private)
    port=8766
    if [[ "$action" == public ]]; then port=8767; fi
    exec "$arena_python" -m case.deploy.private serve \
      --split "$action" --port "$port" --host "$bind_host" \
      --dataset "$closed/$action.jsonl" --state "$closed/$action" \
      --metric "$campaign/metric" --runner-image "$(<"$run/runner-image.txt")" \
      --journal "$run/search.sqlite3" --release-id "$release_id" \
      --max-solutions "$max_solutions" --finalists "$finalists" --winners "$winners"
    ;;
esac
