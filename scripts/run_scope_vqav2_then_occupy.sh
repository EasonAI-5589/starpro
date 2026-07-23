#!/bin/bash
# Wait for VQAv2 to finish, then start occupy.py
set -e

cd /mnt/eason/LLaVA-STAR-Pro

# Wait for tidy-shoal (VQAv2 run) to finish
VQAV2_PID=$(pgrep -f "run_scope_vqav2.sh" | grep -v $$ | head -1)
if [ -n "$VQAV2_PID" ]; then
    echo "[$(date)] Waiting for VQAv2 (PID: $VQAV2_PID) to finish..."
    while kill -0 "$VQAV2_PID" 2>/dev/null; do
        sleep 30
    done
    echo "[$(date)] VQAv2 finished!"
fi

echo "[$(date)] Starting occupy.py --gpus all"
cd /mnt/eason
python occupy.py --gpus all
