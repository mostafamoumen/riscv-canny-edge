# Compilers
HOST_CXX = g++
RV_CXX = riscv64-unknown-elf-g++

# Compiler Flags
CXXFLAGS = -Wall -Wextra -std=c++17 -O2
RV_CXXFLAGS = $(CXXFLAGS) -march=rv64gcv 

# Test Flags (Linking GoogleTest)
GTEST_LIBS = -L$(HOME)/gtest/lib -lgtest -lgtest_main -pthread
GTEST_INCLUDES = -I$(HOME)/gtest/include

# Executable names
HOST_BIN = test_host
RV_BIN = canny_rv

# 1. Compile and run GoogleTest suite natively on the host
test: test.cpp direction.cpp sobel.cpp magnitude.cpp
	$(HOST_CXX) test.cpp direction.cpp sobel.cpp magnitude.cpp $(GTEST_INCLUDES) $(GTEST_LIBS) -o test_host
	./test_host

# 2. Cross-compile the pipeline for RISC-V
canny_rv: main.cpp canny.cpp direction.cpp
	$(RV_CXX) $(RV_CXXFLAGS) main.cpp canny.cpp direction.cpp -o $(RV_BIN)

# 3. Execute the RISC-V binary on QEMU (using vlen=256 as an example)
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./$(RV_BIN)

# 4. Compile profiling harness with Phase 4's highest scalar optimization flag
benchmark_rv: benchmark.cpp image_io.cpp gaussian_blur.cpp sobel.cpp magnitude.cpp direction.cpp
	mkdir -p bin_rv
	$(RV_CXX) $(RV_CXXFLAGS) -O3 $^ -o ./bin_rv/benchmark_rv

# 5. Execute Phase 5 Profiling Harness inside QEMU with defined vector extensions
profile: benchmark_rv
	@echo "Launching Phase 5 Profile Session under QEMU Emulation Layer..."
	qemu-riscv64 -cpu rv64,v=true,vlen=128 ./bin_rv/benchmark_rv

# Execute an automated profiling sweep across multiple VLEN settings
profile_sweep: benchmark_rv
	@echo "========================================="
	@echo "Profiling LMUL Sweep at VLEN = 128"
	@echo "========================================="
	qemu-riscv64 -cpu rv64,v=true,vlen=128 ./bin_rv/benchmark_rv
	@echo "========================================="
	@echo "Profiling LMUL Sweep at VLEN = 256"
	@echo "========================================="
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./bin_rv/benchmark_rv
	@echo "========================================="
	@echo "Profiling LMUL Sweep at VLEN = 512"
	@echo "========================================="
	qemu-riscv64 -cpu rv64,v=true,vlen=512 ./bin_rv/benchmark_rv

# Target to cross-compile your entire pipeline equivalence test for RISC-V
equivalence_rv: equivalence_tests.cpp image_io.cpp gaussian_blur.cpp sobel.cpp magnitude.cpp direction.cpp
	mkdir -p bin_rv
	$(RV_CXX) $(RV_CXXFLAGS) -O2 $^ -o ./bin_rv/equivalence_tests

# Target to execute the VLEN sweep on QEMU automatically
run_equivalence_sweep: equivalence_rv
	@echo "========================================="
	@echo "Running Equivalence Sweep at VLEN = 128"
	@echo "========================================="
	qemu-riscv64 -cpu rv64,v=true,vlen=128 ./bin_rv/equivalence_tests
	@echo "========================================="
	@echo "Running Equivalence Sweep at VLEN = 256"
	@echo "========================================="
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./bin_rv/equivalence_tests
	@echo "========================================="
	@echo "Running Equivalence Sweep at VLEN = 512"
	@echo "========================================="
	qemu-riscv64 -cpu rv64,v=true,vlen=512 ./bin_rv/equivalence_tests

# 6. Remove all generated files
clean:
	rm -f $(HOST_BIN) $(RV_BIN) *.o
	rm -rf bin_rv


# =========================================================================
# PHASE 4: COMPLETELY ISOLATED SCALAR COMPILER SWEEP RULES
# =========================================================================

PHASE4_SRCS = phase4_benchmark.cpp image_io.cpp gaussian_blur.cpp sobel.cpp magnitude.cpp direction.cpp
PHASE4_LOG_DIR = logs_phase4

phase4_build_O0: $(PHASE4_SRCS)
	$(RV_CXX) -Wall -Wextra -std=c++17 -march=rv64gcv -O0 -fopt-info-vec-all=$(PHASE4_LOG_DIR)/vectorization_O0.txt $^ -o ./bin_rv/phase4_scalar_O0

phase4_build_O2: $(PHASE4_SRCS)
	$(RV_CXX) -Wall -Wextra -std=c++17 -march=rv64gcv -O2 -fno-tree-vectorize -fopt-info-vec-all=$(PHASE4_LOG_DIR)/vectorization_O2.txt $^ -o ./bin_rv/phase4_scalar_O2

phase4_build_O3: $(PHASE4_SRCS)
	$(RV_CXX) -Wall -Wextra -std=c++17 -march=rv64gcv -O3 -fno-tree-vectorize -fopt-info-vec-all=$(PHASE4_LOG_DIR)/vectorization_O3.txt $^ -o ./bin_rv/phase4_scalar_O3

phase4_build_Os: $(PHASE4_SRCS)
	$(RV_CXX) -Wall -Wextra -std=c++17 -march=rv64gcv -Os -fno-tree-vectorize -fopt-info-vec-all=$(PHASE4_LOG_DIR)/vectorization_Os.txt $^ -o ./bin_rv/phase4_scalar_Os

phase4_build_Ofast: $(PHASE4_SRCS)
	$(RV_CXX) -Wall -Wextra -std=c++17 -march=rv64gcv -Ofast -fno-tree-vectorize -fopt-info-vec-all=$(PHASE4_LOG_DIR)/vectorization_Ofast.txt $^ -o ./bin_rv/phase4_scalar_Ofast

phase4_run_sweep:
	chmod +x phase4_sweep.sh
	./phase4_sweep.sh