#!/bin/bash
# run_phase5.sh: Automated profiling execution and logging helper

mkdir -p logs_phase5
echo "Compiling and running Phase 5 baseline profile..."

# Build via edited Makefile
make benchmark_rv

# Run profile and capture clean standard output
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./bin_rv/benchmark_rv > ./logs_phase5/scalar_O3_profile_results.txt

# Print results to console
cat ./logs_phase5/scalar_O3_profile_results.txt