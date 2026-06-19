# RISC-V Vector-Accelerated Canny Edge Detection Pipeline

An end-to-end, high-performance Embedded Systems project executing a Canny Edge Detection pipeline on the RISC-V 64-bit architecture with Vector Extensions (`rv64gcv`).

This project establishes a clean, portable C++ scalar baseline, performs compiler flag sweeping analyses, profiles structural execution hotspots, and implements hand-optimized vector pipelines using **RISC-V Vector (RVV) 1.0 Intrinsics** evaluated on user-mode QEMU emulation across varying Vector Lengths (`VLEN`).

---

## 1. Project Overview & Architecture

The minimum required pipeline deliverable covers **Stages 1 and 2** of the Canny Edge Detection algorithm. To guarantee optimal hardware vector utilization, the codebase shifts away from standard data layouts and implements advanced embedded design paradigms:

1. **Gaussian Blur ($5 \times 5$):** Implemented using fully integer arithmetic to eliminate floating-point overhead on the embedded target. To optimize processing loops, the 2D matrix convolution is factored into a **Separable Filter** framework—decomposing a $5 \times 5$ pass into consecutive horizontal ($1 \times 5$) and vertical ($5 \times 1$) 1D vector operations, lowering calculations from 25 down to 10 operations per pixel.
2. **Sobel Filter ($Gx$ and $Gy$):** Computes directional intensity derivatives.
3. **Structure of Arrays (SoA) Data Layout:** Intermediate images and derivative structures ($Gx, Gy$) are maintained in separate contiguous memory buffers rather than interleaved structures (Array of Structures - AoS). This allows the vector engine to utilize unit-stride vector memory loads (`vle16.v`) instead of costly gather-scatter operations.
4. **Magnitude Block:** Supports both lightweight $L_1$ norms ($|Gx| + |Gy|$) and precise integer-bounded $L_2$ norms.
5. **Direction Quantization:** Avoids slow floating-point trigonometric evaluations (`atan2`) by deploying scaled integer tangent approximations ($\tan(22.5^\circ) \approx \frac{2}{5}$ and $\tan(67.5^\circ) \approx \frac{12}{5}$) resolved via cross-multiplication.
6. **Boundary Conditions:** Standardized to a deterministic **Zero-Padding** strategy across image borders.

---

## 2. Repository Structure

```text
├── logs_phase4              # Raw evaluation dumps from compiler sweep metrics
├── logs_phase5              # Profiling metrics and vector execution trace readouts
├── .gitignore               # Excludes binary artifacts and generated raw assets
├── Makefile                 # Dual-target build automation engine
├── main.cpp                 # Execution driver for the edge detection application
├── benchmark.cpp            # Profiling harness checking runtime across 100 loops
├── phase4_benchmark.cpp     # Targeted performance compiler loop evaluation engine
├── test.cpp                 # Main native test runner mapping unit operations
├── equivalence_tests.cpp    # Cross-validation tests (Scalar vs. RVV correctness)
├── generate_test_images.cpp # Automation helper producing mock non-power-of-two assets
├── image_io.cpp / .h        # Native raw unheaded binary disk input/output management
├── gaussian_blur.cpp / .h   # Scalar and RVV implementations for Gaussian processing
├── sobel.cpp / .h           # Scalar and RVV implementations for Sobel processing
├── magnitude.cpp / .h       # Scalar and RVV implementations for Intensity Magnitude
├── direction.cpp / .h       # Scalar and RVV implementations for Directional Quantization
├── syscalls.cpp             # Embedded system configuration adjustments for bare environment
├── phase4_sweep.sh          # Orchestration script checking compiler output efficiencies
├── run_phase5.sh            # Standard execution script for standalone profiling runs
└── run_vlen_sweep.sh        # Validation loops sweeping VLEN boundaries (128, 256, 512)

```

---

## 3. Environment Setup & Prerequisites

Development requires a Linux environment. Windows users must configure WSL2, while macOS users must run inside a structured Docker container.

### 3.1 Dependencies Installation

Install the necessary compilation tools and library components for building both QEMU and the native cross-compiler toolchain:

```bash
sudo apt update && sudo apt install -y \
  autoconf automake build-essential bison flex texinfo gperf libtool patchutils bc git make \
  libglib2.0-dev libpixman-1-dev libslirp-dev ninja-build \
  libmpc-dev libmpfr-dev libgmp-dev zlib1g-dev libexpat1-dev

```

### 3.2 Building the Cross-Compiler Toolchain from Source

Standard distribution compilers do not reliably provide stable `<riscv_vector.h>` support for RVV 1.0. The toolchain must be built manually from source with explicit vector configuration flags:

```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain
cd riscv-gnu-toolchain
./configure --prefix=/opt/riscv --with-arch=rv64gcv --with-abi=lp64d
make -j$(nproc)

```

Add the compiled binaries to your environment path variable (e.g., in `~/.bashrc`):

```bash
export PATH=/opt/riscv/bin:$PATH

```

### 3.3 Building QEMU User-Mode Emulation

To run and benchmark the cross-compiled RISC-V binaries on your host machine, install and configure user-mode QEMU:

```bash
git clone https://github.com/qemu/qemu.git
cd qemu
mkdir build && cd build
../configure --target-list=riscv64-linux-user --enable-slirp
make -j$(nproc)
sudo make install

```

---

## 4. Compilation & Build Orchestration

The Makefile implements a robust dual-target build infrastructure, utilizing the local host compiler (`g++`) for native validation testing and the cross-compiler (`riscv64-unknown-elf-g++`) for targeted emulation runs.

| Command Target  | Execution Architecture   | Purpose and Functionality                                                                  |
| --------------- | ------------------------ | ------------------------------------------------------------------------------------------ |
| `make test`     | Native Host (`g++`)      | Compiles and runs the full GoogleTest testing suite to verify pipeline integrity.          |
| `make canny_rv` | RISC-V Cross-Compilation | Compiles the target application using specific compiler flags for RISC-V target execution. |
| `make run`      | QEMU User Emulation      | Executes the cross-compiled edge detection pipeline inside `qemu-riscv64`.                 |
| `make clean`    | Housekeeping Utility     | Cleans out compiled object files (`.o`), local test binaries, and output images.           |

---

## 5. Execution Guide (Single-Step Image Loading)

The pipeline abstracts data pre-processing and format extraction via an internal automated wrapper interface in the build layer. Standard web formats (`.jpg`, `.png`, `.heic`) are parsed, handled dynamically with 64-byte memory alignment blocks (`aligned_alloc`), and normalized to a headerless raw grayscale array structure ($\text{Width} \times \text{Height}$ bytes) for processing.

### Execution Examples

You can pass target assets directly to the runner interface. If your test image features custom or arbitrary non-power-of-two dimensions, pass the explicit boundaries directly as command parameters:

```bash
# Example 1: Standard JPEG execution pass
make run IMG=cool_picture.jpg

# Example 2: Run verification against PNG format
make run IMG=vacation.png

# Example 3: Running an arbitrary size non-power-of-two HEIC asset with explicit dimension specifications
make run IMG=another_photo.heic WIDTH=256 HEIGHT=256

```

### Vector-Length Agnosticism (VLA) Verification

To test and verify the robust vector-length agnosticism of your strip-mining tail loop configurations, manually alter the underlying hardware parameters inside the QEMU call window by supplying different bit-widths to the `-cpu` environment flag:

```bash
# Simulating a tight 128-bit vector registers system
qemu-riscv64 -cpu rv64,v=true,vlen=128 ./bin_rv/main_rv

# Simulating an advanced 512-bit enterprise vector registers infrastructure
qemu-riscv64 -cpu rv64,v=true,vlen=512 ./bin_rv/main_rv

```

---

## 6. Performance Optimization Matrix

The table below shows execution metrics gathered across compiler sweep assessments and hardware configuration runs.

> **Note 1:** Standard `-O3` options serve as the functional reference for baseline Compiler Auto-vectorization behavior.
> **Note 2:** To keep metrics uniform, profiling sweeps across Phase 5 and 6 (which evaluate a loop of 100 frames) have been normalized to reflect single-frame runtime values ($\text{Total} / 100$).
> **Note 3:** RVV operational configurations indicate maximum acceleration efficiency achieved across sweeping register allocation footprints ($\text{LMUL}=4$).

| Pipeline Stage              | -O0       | -O2       | -O3 (Auto-vec) | -Os       | -Ofast    | RVV (VLEN=128) | RVV (VLEN=256) | RVV (VLEN=512) |
| --------------------------- | --------- | --------- | -------------- | --------- | --------- | -------------- | -------------- | -------------- |
| **Gaussian Blur (5x5)**     | 0.8577 ms | 0.2081 ms | 0.0903 ms      | 0.2171 ms | 0.0892 ms | 0.3800 ms      | 0.3193 ms      | 0.2675 ms      |
| **Sobel Derivative Filter** | 0.1254 ms | 0.1883 ms | 0.0292 ms      | 0.0326 ms | 0.0289 ms | 0.1889 ms      | 0.1612 ms      | 0.2152 ms      |
| **Magnitude Processing**    | 1.4327 ms | 0.9238 ms | 0.9080 ms      | 0.8516 ms | 0.2966 ms | 0.2077 ms      | 0.1181 ms      | 0.1084 ms      |
| **Direction Quantization**  | 0.0573 ms | 0.0237 ms | 0.0250 ms      | 0.0339 ms | 0.1882 ms | _Scalar_       | _Scalar_       | _Scalar_       |
| **Binary Output Size**      | 1225.8 KB | 1176.7 KB | 1180.7 KB      | 1176.9 KB | 1180.5 KB | 1176.9 KB      | 1176.9 KB      | 1176.9 KB      |

### Performance Analysis Insights

- **The Amdahl's Law Profiling Reality:** Initial profiling indicated that direction quantization accounted for less than 14% of computational overhead. Writing intensive vector intrinsics for this stage yields diminishing returns; engineering effort was instead targeted at high-overhead operations like Magnitude extraction.
- **In-Flight Data Widening Pitfalls:** During Gaussian and Sobel calculations, data ranges expand beyond standard byte sizes. The RVV implementation utilizes specialized vector widening multipliers (`vwmul`) to scale 8-bit unsigned pixels safely into 16-bit or 32-bit registers. This operational choice dynamically inflates the active register allocation multiplier factor ($\text{LMUL}$), requiring precise variable type mapping across structural vector chains.

---

## 7. AI Tool Usage Log

In accordance with course academic accountability guidelines, key highlights from the AI assistance interactions are detailed below:

### Log Entry 1

- **Prompt/Query Enacted:** `"Rewrite a standard 2D 5x5 integer convolution kernel matrix into a pair of separable 1D horizontal and vertical pass equations to decrease looping cycles inside C++."`
- **AI Tool Recommendation:** Provided a structured code pattern that isolates the 1D row buffer additions into a temporary array before executing a secondary vertical sweep pass.
- **Adjustments/Modifications Made:** Refined the border tracking conditions to match a strict zero-padding model across image corners.
- **Core Insight Gathered:** Understood how decoupling 2D operations minimizes algorithmic execution constraints from $\mathcal{O}(K^2)$ to $\mathcal{O}(2K)$ per single image pixel location.

### Log Entry 2

- **Prompt/Query Enacted:** `"Provide the exact syntax sequence for taking an unaligned input pointer array and forcing 64-byte hardware boundary restrictions inside modern C++ infrastructure."`
- **AI Tool Recommendation:** Suggested deploying POSIX compliant structure extensions or using the modern standardized `std::aligned_alloc(64, size)` call.
- **Adjustments/Modifications Made:** Swapped raw `malloc` instances across memory setup blocks with `aligned_alloc`, sizing allocations to clean multiples of 64.
- **Core Insight Gathered:** Aligned memory boundaries are essential for SIMD optimization, preventing penalties from unaligned memory cache line splits.

### Log Entry 3

- **Prompt/Query Enacted:** `"My RVV vector logic compilation is throwing hard failures on lines mixing vwmul operations with baseline 8-bit input vectors. Debug register size tracking issues inside standard RISC-V GNU compilers."`
- **AI Tool Recommendation:** Highlighted that widening operations (`vwmul`) double the working data bit-widths, meaning the logical register allocation multiplier (`LMUL`) is scaled upward simultaneously.
- **Adjustments/Modifications Made:** Explicitly updated destination variable declarations to utilize expanded types matching the tracking configuration (e.g., `vint16m2_t` targets).
- **Core Insight Gathered:** Hardware register multiplier tracking requires strict compiler alignment; tracking layout changes across operations is mandatory.

### Log Entry 4

- **Prompt/Query Enacted:** `"Draft a clean Makefile template capable of handling standard host g++ targets alongside an independent cross-compilation toolchain path structure for RISC-V user emulation testing loops."`
- **AI Tool Recommendation:** Provided a standard Makefile layout utilizing distinct compiler assignment flags (`CXX` vs `RV_CXX`) alongside explicit invocation wrappers for execution paths via `qemu-riscv64`.
- **Adjustments/Modifications Made:** Customized processing arguments to directly support raw input specifications (`IMG`, `WIDTH`, `HEIGHT`).
- **Core Insight Gathered:** Separating tool paths inside automated build definitions avoids host execution target contamination during deployment routines.

