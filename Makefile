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
test: test.cpp canny.cpp
	$(HOST_CXX) $(CXXFLAGS) $(GTEST_INCLUDES) test.cpp canny.cpp -o $(HOST_BIN) $(GTEST_LIBS)
	./$(HOST_BIN)

# 2. Cross-compile the pipeline for RISC-V
canny_rv: main.cpp canny.cpp
	$(RV_CXX) $(RV_CXXFLAGS) main.cpp canny.cpp -o $(RV_BIN)

# 3. Execute the RISC-V binary on QEMU (using vlen=256 as an example)
run: canny_rv
	qemu-riscv64 -cpu rv64,v=true,vlen=256 ./$(RV_BIN)

# 4. Remove all generated files
clean:
	rm -f $(HOST_BIN) $(RV_BIN) *.o
