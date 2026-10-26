#!/bin/sh
# Runs the offline tests of one example in a clean container.
#   scripts/test-example.sh examples/<name>
# Python examples: requirements.txt + pytest + ruff. JavaScript examples:
# package.json with an "npm test" script.
set -eu
dir=$(cd "$1" && pwd)
if [ -f "$dir/requirements.txt" ]; then
  docker run --rm -v "$dir:/w" -w /w python:3.12-slim sh -c \
    "pip install -q --root-user-action=ignore -r requirements.txt pytest ruff && ruff check . && pytest -q -p no:cacheprovider"
elif [ -f "$dir/package.json" ]; then
  docker run --rm -v "$dir:/w" -w /w node:24-slim sh -c "npm ci --no-audit --no-fund && npm test"
else
  echo "no requirements.txt or package.json in $1" >&2
  exit 1
fi
