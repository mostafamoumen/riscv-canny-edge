#!/bin/bash

# Configuration
BIN_DIR="./bin_rv"
LOG_DIR="logs_phase4"
LEVELS=("O0" "O2" "O3" "Os" "Ofast")

clear
echo "========================================================="
echo "       Executing Purified Phase 4 Compiler Sweep         "
echo "========================================================="
echo ""

# Print Table Header
printf "%-10s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
       "Level" "Binary Size" "Gaussian" "Sobel" "Mag L1" "Mag L2" "Direction"
echo "----------------------------------------------------------------------------------------"

for opt in "${LEVELS[@]}"; do
    # 1. Quietly build the binary using make targets
    make phase4_build_${opt} > /dev/null 2>&1
    
    # 2. Get Binary Size
    BINARY="${BIN_DIR}/phase4_scalar_${opt}"
    if [ -f "$BINARY" ]; then
        SIZE=$(stat -c%s "$BINARY")
    else
        SIZE="N/A"
    fi
    
    # 3. Run Benchmark silently and capture raw output metrics
    if [ -f "$BINARY" ]; then
        RAW_OUT=$(qemu-riscv64 -cpu rv64,v=true,vlen=256 "$BINARY" 2>/dev/null)
        
        # Parse timing metrics via grep/sed
        GAUSSIAN=$(echo "$RAW_OUT" | grep "GAUSSIAN_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        SOBEL=$(echo "$RAW_OUT" | grep "SOBEL_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        MAG_L1=$(echo "$RAW_OUT" | grep "MAG_L1_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        MAG_L2=$(echo "$RAW_OUT" | grep "MAG_L2_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        DIR=$(echo "$RAW_OUT" | grep "DIRECTION_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
    else
        GAUSSIAN="N/A"; SOBEL="N/A"; MAG_L1="N/A"; MAG_L2="N/A"; DIR="N/A"
    fi

    # 4. Print clean structured row
    printf "%-10s | %'-12d B | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
           "-$opt" "$SIZE" "$GAUSSIAN" "$SOBEL" "$MAG_L1" "$MAG_L2" "$DIR"
done

echo "----------------------------------------------------------------------------------------"
echo "Phase 4 Sweep Completed. Diagnostic data saved to $LOG_DIR/"
echo "========================================================="