#!/usr/bin/env bash
# Rebuild every committed result from the recorded model calls, with no API key and no live calls.
# Usage: scripts/reproduce.sh            (about 5 minutes on a laptop CPU, most of it embedding)
# The pgvector index tuning needs Postgres and is not part of this run (see scripts/index_tuning.py).
set -euo pipefail
cd "$(dirname "$0")/.."

python3 -m venv .venv
.venv/bin/pip install -q -c constraints.txt -e ".[dev,retrieval]"
py() { env -u GROQ_API_KEY .venv/bin/python "$@"; }

py scripts/download_data.py
.venv/bin/python -m pytest -q

for approach in alone code temp0; do
  py scripts/run_approaches.py --approach "$approach" --max-live-calls 0
done
py scripts/run_approaches.py --approach vote --k 3 --max-live-calls 0
py scripts/compare.py

py scripts/embed_chunks.py
py scripts/eval_retrieval.py
py scripts/hybrid_by_number.py
py scripts/eval_rerank.py
py scripts/run_retrieved.py
py scripts/retrieval_taxonomy.py
py scripts/measure_incremental.py

echo
echo "Results that differ from the committed ones (empty means everything reproduced):"
git --no-pager diff --stat -- results/*.md
