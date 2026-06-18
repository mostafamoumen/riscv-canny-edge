# Existing declarations preserved 
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
RV_CXXFLAGS = $(CXXFLAGS) -march=rv64gcv 

# Append these targets to the bottom of your existing Makefile

# Compile profiling harness with Phase 4's highest scalar optimization flag
benchmark_rv: benchmark.cpp image_io.cpp gaussian_blur.cpp sobel.cpp magnitude.cpp direction.cpp
	mkdir -p bin_rv
	$(RV_CXX) $(RV_CXXFLAGS) -O3 $^ -o ./bin_rv/benchmark_rv

# Execute Phase 5 Profiling Harness inside QEMU with defined vector extensions
profile: benchmark_rv
	@echo "Launching Phase 5 Profile Session under QEMU Emulation Layer..."
	qemu-riscv64 -cpu rv64,v=true,vlen=128 ./bin_rv/benchmark_rv