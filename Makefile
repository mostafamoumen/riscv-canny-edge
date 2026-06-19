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

# =========================================================================
# Image input configuration (override on command line, e.g.:
#   make run IMG=cool_picture.jpg
#   make run IMG=vacation.png WIDTH=256 HEIGHT=256
# =========================================================================
IMG ?= cool_picture.jpg
WIDTH ?= 512
HEIGHT ?= 512

# 1. Compile and run GoogleTest suite natively on the host
test: test.cpp direction.cpp sobel.cpp magnitude.cpp
	$(HOST_CXX) test.cpp direction.cpp sobel.cpp magnitude.cpp $(GTEST_INCLUDES) $(GTEST_LIBS) -o test_host
	./test_host

# 2. Cross-compile the pipeline for RISC-V
canny_rv: main.cpp image_io.cpp gaussian_blur.cpp sobel.cpp magnitude.cpp direction.cpp syscalls.cpp
	$(RV_CXX) $(RV_CXXFLAGS) $^ -o $(RV_BIN)

# =========================================================================
# Convert any input image into the raw grayscale format the simulator
# expects. Marked .PHONY so it ALWAYS re-runs regardless of file
# timestamps -- otherwise Make sees my_image.raw already exists and
# silently reuses the old one even after you change IMG=.
# =========================================================================
.PHONY: my_image.raw
my_image.raw:
	@echo "Converting $(IMG) to my_image.raw ($(WIDTH)x$(HEIGHT) grayscale)..."
	ffmpeg -y -i $(IMG) -vf "scale=$(WIDTH):$(HEIGHT),format=gray" -f rawvideo my_image.raw

# =========================================================================
# UPDATED: Run simulation with specific dimensions AND auto-convert to PNG
# Now also auto-converts the chosen IMG into my_image.raw as a prerequisite.
# =========================================================================
.PHONY: run
run: canny_rv my_image.raw
	@echo "Running QEMU Simulation with vlen=256 and elen=64..."
	qemu-riscv64 -cpu rv64,v=true,vlen=256,elen=64 ./$(RV_BIN) $(WIDTH) $(HEIGHT) my_image.raw
	@echo "Converting output_scalar.raw to scalar_result.png..."
	ffmpeg -y -f rawvideo -pixel_format gray -video_size $(WIDTH)x$(HEIGHT) -i output_scalar.raw scalar_result.png
	@echo "Converting output_rvv.raw to rvv_result.png..."
	ffmpeg -y -f rawvideo -pixel_format gray -video_size $(WIDTH)x$(HEIGHT) -i output_rvv.raw rvv_result.png
	@echo "Done! You can now view your PNG images."

# 4. Compile profiling harness with Phase 4's highest scalar optimization flag
benchmark_rv: benchmark.cpp image_io.cpp gaussian_blur.cpp sobel.cpp magnitude.cpp direction.cpp syscalls.cpp
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
equivalence_rv: equivalence_tests.cpp image_io.cpp gaussian_blur.cpp sobel.cpp magnitude.cpp direction.cpp syscalls.cpp
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

# 6. Remove all generated files (Updated to tidy up raw and png files too)
clean:
	rm -f $(HOST_BIN) $(RV_BIN) *.o *.raw *.png
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