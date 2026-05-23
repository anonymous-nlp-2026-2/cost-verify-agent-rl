#!/bin/bash
# Safety net: watch PID 920380, trigger auto_next when it exits
TARGET_PID=920380
H_VALUE=2
SEED=1
CHECK_INTERVAL=300

while kill -0 $TARGET_PID 2>/dev/null; do
    sleep $CHECK_INTERVAL
done

echo "[$(date)] PID $TARGET_PID exited, triggering auto_next_hsweep.sh $H_VALUE $SEED"
bash ./scripts/auto_next_hsweep.sh $H_VALUE $SEED
