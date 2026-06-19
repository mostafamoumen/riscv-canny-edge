#!/bin/bash

# Configuration
BIN_DIR="./bin_rv"
LOG_DIR="logs_phase4"
LEVELS=("O0" "O2" "O3" "Os" "Ofast")

# Ensure all output and log paths exist before running compiler steps
mkdir -p "$BIN_DIR"
mkdir -p "$LOG_DIR"

# Temporary file to build the summary log simultaneously
SUMMARY_LOG="$LOG_DIR/sweep_summary.txt"
echo "=========================================================" > "$SUMMARY_LOG"
echo "       Executing Purified Phase 4 Compiler Sweep         " >> "$SUMMARY_LOG"
echo "=========================================================" >> "$SUMMARY_LOG"
echo "" >> "$SUMMARY_LOG"
printf "%-10s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
       "Level" "Binary Size" "Gaussian" "Sobel" "Mag L1" "Mag L2" "Direction" >> "$SUMMARY_LOG"
echo "----------------------------------------------------------------------------------------" >> "$SUMMARY_LOG"

clear
echo "========================================================="
echo "       Executing Purified Phase 4 Compiler Sweep         "
echo "========================================================="
echo ""

# Print Table Header to Terminal
printf "%-10s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
       "Level" "Binary Size" "Gaussian" "Sobel" "Mag L1" "Mag L2" "Direction"
echo "----------------------------------------------------------------------------------------"

for opt in "${LEVELS[@]}"; do
    # 1. Quietly build the binary using make targets
    make phase4_build_${opt} > /dev/null 2>&1
    
    # 2. Get Binary Size and Process Outputs safely
    BINARY="${BIN_DIR}/phase4_scalar_${opt}"
    if [ -f "$BINARY" ]; then
        SIZE_BYTES=$(stat -c%s "$BINARY")
        SIZE_STR="$(printf "%'d" $SIZE_BYTES) B"
        
        # 3. Run Benchmark silently and capture raw output metrics
        RAW_OUT=$(qemu-riscv64 -cpu rv64,v=true,vlen=256 "$BINARY" 2>/dev/null)
        
        # Save the full raw stdout dump of the binary for your records
        echo "$RAW_OUT" > "$LOG_DIR/raw_run_${opt}.txt"
        
        # Parse timing metrics via grep/sed
        GAUSSIAN=$(echo "$RAW_OUT" | grep "GAUSSIAN_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        SOBEL=$(echo "$RAW_OUT" | grep "SOBEL_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        MAG_L1=$(echo "$RAW_OUT" | grep "MAG_L1_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        MAG_L2=$(echo "$RAW_OUT" | grep "MAG_L2_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        DIR=$(echo "$RAW_OUT" | grep "DIRECTION_TIME:" | cut -d':' -f2 | awk '{printf "%.4f ms", $1}')
        
        # Handle cases where the binary built but QEMU execution threw an error
        [ -z "$GAUSSIAN" ] && GAUSSIAN="N/A"
        [ -z "$SOBEL" ] && SOBEL="N/A"
        [ -z "$MAG_L1" ] && MAG_L1="N/A"
        [ -z "$MAG_L2" ] && MAG_L2="N/A"
        [ -z "$DIR" ] && DIR="N/A"
    else
        SIZE_STR="Missing Bin"
        GAUSSIAN="N/A"; SOBEL="N/A"; MAG_L1="N/A"; MAG_L2="N/A"; DIR="N/A"
    fi

    # 4. Print clean structured row to both terminal and summary file
    printf "%-10s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
           "-$opt" "$SIZE_STR" "$GAUSSIAN" "$SOBEL" "$MAG_L1" "$MAG_L2" "$DIR"
           
    printf "%-10s | %-12s | %-10s | %-10s | %-10s | %-10s | %-10s\n" \
           "-$opt" "$SIZE_STR" "$GAUSSIAN" "$SOBEL" "$MAG_L1" "$MAG_L2" "$DIR" >> "$SUMMARY_LOG"
done

echo "----------------------------------------------------------------------------------------"
echo "----------------------------------------------------------------------------------------" >> "$SUMMARY_LOG"
echo "Phase 4 Sweep Completed. Diagnostic data saved to $LOG_DIR/"
echo "Phase 4 Sweep Completed. Diagnostic data saved to $LOG_DIR/" >> "$SUMMARY_LOG"
echo "========================================================="
echo "=========================================================" >> "$SUMMARY_LOG"