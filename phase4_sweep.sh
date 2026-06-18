#!/bin/bash
set -e

FLAGS=("-O0" "-O2" "-O3" "-Os" "-Ofast")
SOURCE_FILES="benchmark.cpp image_io.cpp sobel.cpp magnitude.cpp direction.cpp"

echo "=========================================="
echo " Starting Compiler Optimization Sweep "
echo "=========================================="

for FLAG in "${FLAGS[@]}"
do
    echo "------------------------------------------"
    echo "Compiling with ${FLAG}..."
    
    # Compile
    riscv64-unknown-elf-g++ ${FLAG} -ftree-vectorize -std=c++17 -march=rv64gcv -mabi=lp64d ${SOURCE_FILES} -o benchmark_${FLAG}.elf
    
    # Measure Binary Size
    # We use 'size' to get the actual instruction (.text) footprint, avoiding ELF header bloat
    BIN_SIZE=$(riscv64-unknown-elf-size benchmark_${FLAG}.elf | awk 'NR==2 {print $1}')
    echo "Binary .text size: ${BIN_SIZE} bytes"
    
    # Execute on QEMU
    qemu-riscv64 -cpu rv64,v=true,vlen=128 ./benchmark_${FLAG}.elf
done