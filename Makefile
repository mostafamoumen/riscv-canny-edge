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

# 6. Remove all generated files
clean:
	rm -f $(HOST_BIN) $(RV_BIN) *.o
	rm -rf bin_rv