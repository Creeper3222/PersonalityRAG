#!/usr/bin/env sh
set -eu

STATE_ROOT="${PERSONALITYRAG_STATE_ROOT:-/app/state}"
mkdir -p "$STATE_ROOT/config" "$STATE_ROOT/data"

if [ "$#" -eq 0 ]; then
    set -- python /app/run.py
fi

exec "$@"
