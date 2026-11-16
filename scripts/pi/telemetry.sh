#!/usr/bin/env bash
# Sample the pot's thermals, clock, load and free memory once a second.
#
# Prints CSV to stdout until killed; run it in the background alongside a
# measurement. Columns: seconds since start, CPU temperature in °C, ARM clock
# in Hz, throttle flags from vcgencmd, 1-minute load average, and
# MemAvailable in KiB.
#
# Usage: telemetry.sh > telemetry.csv &

set -euo pipefail

echo "time_s,temp_c,arm_clock_hz,throttled,load_1m,mem_available_kib"
start=$(date +%s.%N)
while true; do
    now=$(date +%s.%N)
    temp=$(< /sys/class/thermal/thermal_zone0/temp)
    clock=$(vcgencmd measure_clock arm | cut -d= -f2)
    throttled=$(vcgencmd get_throttled | cut -d= -f2)
    load=$(cut -d' ' -f1 /proc/loadavg)
    mem=$(awk '/^MemAvailable:/ {print $2}' /proc/meminfo)
    awk -v t="$now" -v s="$start" -v c="$temp" \
        'BEGIN { printf "%.1f,%.1f,", t - s, c / 1000 }'
    echo "$clock,$throttled,$load,$mem"
    sleep 1
done
