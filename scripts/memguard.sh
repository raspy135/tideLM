#!/usr/bin/env bash
# memguard.sh MIN_GB CMD... -- run CMD in its own process group and kill the group once
# MemAvailable drops below MIN_GB (polled every 50 ms): the Rust server's KEEP_FREE_GB. On
# this unified-memory box the GPU's allocations are the system's, and a server that runs it out
# of memory takes the desktop with it; a margin above zero leaves room for the GPU teardown,
# which outlives SIGKILL.
MIN=$(( $1 * 1048576 )); shift
setsid "$@" & PID=$!
LOW=999999999
while kill -0 $PID 2>/dev/null; do
    A=$(awk '/MemAvailable/{print $2}' /proc/meminfo)
    [ "$A" -lt "$LOW" ] && LOW=$A
    if [ "$A" -lt "$MIN" ]; then
        kill -9 -$PID 2>/dev/null
        echo "GUARD: killed at MemAvailable $((A / 1048576)) GiB (KEEP_FREE_GB $((MIN / 1048576)))" >&2
        sleep 1
    fi
    sleep 0.05
done
wait $PID; RC=$?
echo "GUARD: lowest MemAvailable $((LOW / 1048576)) GiB" >&2
exit $RC
