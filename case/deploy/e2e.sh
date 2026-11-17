#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.."
campaign="$PWD/case/campaign/ar-12-e2e"
run="$campaign/.run"
arena_python="$PWD/arena/.venv/bin/python"
export PYTHONPATH="$PWD/arena"
network=ar12-research
closed_pointer="$run/closed-dir"

case "${1:-}" in
  setup|train|public|private|run) action=$1 ;;
  *) echo "Usage: bash case/deploy/e2e.sh {setup|train|public|private|run MODEL_ACCESS_JSON}" >&2; exit 2 ;;
esac

if [[ ! -f "$run/search.sqlite3" ]]; then
  echo "Prepare ar-12-e2e with prepare-pilot before running this command." >&2
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
      closed=$(mktemp -d "${XDG_STATE_HOME:-$HOME/.local/state}/ar12-e2e.XXXXXX" 2>/dev/null) ||
        closed=$(mktemp -d /tmp/ar12-e2e.XXXXXX)
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
        "case/dataset/ar12-test/ar12.$split.jsonl" "$closed/$split.jsonl"
    fi
  done
fi

if [[ ! -f "$closed_pointer" ]]; then
  echo "Run bash case/deploy/e2e.sh setup first." >&2
  exit 1
fi
closed=$(<"$closed_pointer")
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
    data="$PWD/case/dataset/ar12-pilot-v1/data"
    if [[ ! -d "$workspace" ]]; then
      .venv/bin/python case/deploy/agents.py stage --campaign "$campaign" \
        --train "$data/train.jsonl" --evaluation "$data/evaluation.arena.jsonl" \
        --workspace "$workspace"
    fi
    private_config="$closed/agent-access.json"
    .venv/bin/python - "$private_config" "$private_url" <<'PY'
import json
from pathlib import Path
import sys

Path(sys.argv[1]).write_text(json.dumps({
    "url": sys.argv[2], "token_file": "/run/arena/private-token",
    "finalists": 2, "output_count": 1,
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
      --journal "$run/search.sqlite3" --release-id ar12-e2e-v1 \
      --max-solutions 6 --finalists 2 --winners 1
    ;;
esac
