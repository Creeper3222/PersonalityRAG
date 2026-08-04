#!/usr/bin/env sh
set -eu

STATE_ROOT="${PERSONALITYRAG_STATE_ROOT:-/app/state}"
mkdir -p "$STATE_ROOT/config" "$STATE_ROOT/data"
PERSONALITYRAG_STATE_ROOT="$STATE_ROOT"
: "${PERSONALITYRAG_DEPLOYMENT:=docker}"
: "${PERSONALITYRAG_SUPPRESS_BROWSER:=1}"
export PERSONALITYRAG_STATE_ROOT PERSONALITYRAG_DEPLOYMENT PERSONALITYRAG_SUPPRESS_BROWSER

detect_cpu_limit() {
    detected=""
    if [ -r /sys/fs/cgroup/cpu.max ]; then
        set -- $(cat /sys/fs/cgroup/cpu.max)
        if [ "${1:-max}" != "max" ] && [ "${2:-0}" -gt 0 ] 2>/dev/null; then
            detected=$(( ($1 + $2 - 1) / $2 ))
        fi
    elif [ -r /sys/fs/cgroup/cpu/cpu.cfs_quota_us ] \
        && [ -r /sys/fs/cgroup/cpu/cpu.cfs_period_us ]; then
        quota=$(cat /sys/fs/cgroup/cpu/cpu.cfs_quota_us)
        period=$(cat /sys/fs/cgroup/cpu/cpu.cfs_period_us)
        if [ "$quota" -gt 0 ] 2>/dev/null && [ "$period" -gt 0 ] 2>/dev/null; then
            detected=$(( (quota + period - 1) / period ))
        fi
    fi
    if [ -z "$detected" ]; then
        detected=$(getconf _NPROCESSORS_ONLN 2>/dev/null || printf '1')
    fi
    if [ "$detected" -lt 1 ] 2>/dev/null; then
        detected=1
    elif [ "$detected" -gt 4 ] 2>/dev/null; then
        detected=4
    fi
    printf '%s' "$detected"
}

THREAD_LIMIT=$(detect_cpu_limit)
: "${OMP_NUM_THREADS:=$THREAD_LIMIT}"
: "${OPENBLAS_NUM_THREADS:=$THREAD_LIMIT}"
: "${MKL_NUM_THREADS:=$THREAD_LIMIT}"
: "${NUMEXPR_NUM_THREADS:=$THREAD_LIMIT}"
export OMP_NUM_THREADS OPENBLAS_NUM_THREADS MKL_NUM_THREADS NUMEXPR_NUM_THREADS

if [ "$#" -eq 0 ]; then
    set -- python /app/run.py
fi

exec "$@"
