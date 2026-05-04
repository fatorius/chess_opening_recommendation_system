#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PGN_PATH="${1:-${ROOT}/lichess.pgn}"

if [[ ! -f "$PGN_PATH" ]]; then
  echo "PGN file not found: $PGN_PATH" >&2
  exit 1
fi

if [[ -f "$ROOT/.venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/.venv/bin/activate"
elif [[ -f "$ROOT/venv/bin/activate" ]]; then
  # shellcheck disable=SC1091
  source "$ROOT/venv/bin/activate"
else
  echo "No virtual environment found. Create .venv or venv first." >&2
  exit 1
fi

python -m pip install -r "$ROOT/requirements.txt"

echo "Starting extract_lichess_openings.py..."
python "$ROOT/extract_lichess_openings.py" --input "$PGN_PATH" --output "$ROOT/games.csv"

echo "Starting preprocess_chess_data.py..."
python "$ROOT/preprocess_chess_data.py" --input "$ROOT/games.csv" --output "$ROOT/games_encoded.csv" --encoders "$ROOT/encoders.pkl"

echo "Starting tf_style_model.py..."
python "$ROOT/tf_style_model.py"
