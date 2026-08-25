# Kernel Substring Search: An Exploration of Ring 0 Vectorization

![Platform: Windows Kernel](https://img.shields.io/badge/Platform-Windows%20Kernel%20(Ring%200)-blue)
![Language: C++20](https://img.shields.io/badge/Language-C%2B%2B20-orange)
![SIMD: AVX2 | NEON | SWAR](https://img.shields.io/badge/SIMD-AVX2%20%7C%20ARM64%20NEON%20%7C%20SWAR-green)

An empirical case study and exploration into high-performance UTF-8 (8-bit) and UTF-16 (16-bit `wchar_t`) substring searching within the Windows kernel environment. The study examines how the cost of entering vector execution in Ring 0 changes the optimal substring-search strategy across buffer sizes, pattern characteristics, and execution environments. Combining Boyer-Moore-Horspool, SWAR, AVX2, and ARM64 NEON, this project examines the low-level trade-offs of kernel-mode vector optimization, cache locality, and general-purpose register (GPR) fallback algorithms. While not strictly benchmark-driven during its design, the repository includes comprehensive hardware telemetry to validate its architectural findings.

* **Dynamic Ring 0 Routing:** Bypasses vector registers for small buffers to eliminate the `KeSaveExtendedProcessorState` extended processor-state save/restore overhead.
* **Hybrid Boyer-Moore-Horspool (BMH):** Blends sub-linear bad-character skipping with 4x unrolled SIMD memory sliding.
* **L1 Cache–Conscious Golden Ratio Hashing:** Folds wide UTF-16 skip tables from an unfeasible 256KB footprint down to 8KB to ensure residency in the L1 D-Cache.
* **Hardware-Agnostic SWAR Fallbacks:** Implements 64-bit SIMD-Within-A-Register bit-manipulation hacks to process 8 bytes (or 4 wide characters) per cycle without engaging FPU/YMM state.
* **Cross-Architecture Vector Pipelines:** Native implementations for x86_64 (AVX2) and ARM64 (NEON).
* **Experimental BNDM Evaluation:** Features Backward Nondeterministic DAWG Matching implementations with an analysis of why bit-parallel automata underperform BMH in kernel workloads.

---

## Table of Contents
1. [The Kernel Optimization Landscape & Ring 0 Trade-offs](#1-the-kernel-optimization-landscape--ring-0-trade-offs)
2. [Algorithm Architecture: Hybrid BMH](#2-algorithm-architecture-hybrid-bmh)
3. [Execution Engines: SIMD vs. SWAR](#3-execution-engines-simd-vs-swar)
4. [The Experimental Branch: Why BNDM Falls Behind](#4-the-experimental-branch-why-bndm-falls-behind)
5. [Benchmark Analysis](#5-benchmark-analysis)
6. [Repository Structure](#6-repository-structure)
7. [API Reference & Usage](#7-api-reference--usage)
8. [Building & Testing](#8-building--testing)
9. [License](#9-license)

---

## 1. The Kernel Optimization Landscape & Ring 0 Trade-offs

In user-mode applications, maximizing substring search performance is often a matter of streaming vectors directly through AVX-512 or AVX2 pipelines. In Windows kernel mode (Ring 0), vector optimization introduces strict architectural trade-offs:

~~~text
+-------------------------------------------------------------------------+
|                              Call: Find()                               |
+------------------------------------+------------------------------------+
                                     |
                         Buffer <= KERNEL_THRESHOLD?
                                     |
                  +------------------+------------------+
                  | YES                                 | NO
                  v                                     v
      +---------------------------+         +-----------------------+
      |      GPR SWAR /           |         |  KeSaveExtendedState  |
      |     BMH Fast Path         |         +-----------+-----------+
      | (Zero FPU State Overhead) |                     |
      +---------------------------+         +-----------v-----------+
                                            |  AVX2 / NEON 4x Slide |
                                            +-----------+-----------+
                                                        |
                                            +-----------v-----------+
                                            | KeRestoreExtendedState|
                                            +-----------------------+
~~~

### The Ring 0 Extended Processor State Overhead
The Windows kernel does not preserve floating-point and extended vector (XMM/YMM/ZMM) registers across thread context switches by default. To safely issue SIMD instructions in Ring 0 without corrupting user-mode thread state, the caller must allocate an `XSTATE_SAVE` structure and invoke `KeSaveExtendedProcessorState(XSTATE_MASK_AVX, ...)` (or `KeSaveFloatingPointState` on ARM64). 

This state preservation routine writes processor context to memory via XSAVE/XRSTOR, introducing fixed microsecond overheads that severely penalize short operations.

### Smart Auto Routing & Bypass
Smart Auto is the default runtime routing mode that selects between the GPR/SWAR/BMH path and the vectorized path according to the implementation's buffer-size threshold and processor capabilities. For small search buffers (e.g., < 512 characters), the XSTATE setup and teardown latency exceeds the execution time of the search itself. The engine uses the KERNEL_THRESHOLD_CHARS heuristic to dynamically balance throughput and context-switch costs:
* **Small Buffers (<= Threshold):** Bypasses vector registers completely and routes directly to the GPR-based SWAR fallback, incurring zero floating-point save overhead.
* **Large Buffers (> Threshold):** Wraps execution in safe processor state-saving calls to unleash the 4x unrolled vector pipeline across memory.

### IRQL Constraints & Memory Paging
Table precomputation in `Initialize()` relies on `ExAllocatePool2` with `POOL_FLAG_PAGED` and `POOL_FLAG_CACHE_ALIGNED`. The initialization routines are annotated with `PAGED_CODE()` and must execute at `PASSIVE_LEVEL` or `APC_LEVEL`.

---

## 2. Algorithm Architecture: Hybrid BMH

The primary production engines (`CKmStrSearch8` and `CKmStrSearch16`) combine Boyer-Moore-Horspool bad-character shift rules with unrolled vector sliding.

### 8-Bit Architecture (`CKmStrSearch8`)
* **Flat 256-Entry Lookup Table:** Precomputes a 1KB bad-character jump table for all ASCII/extended-ASCII byte values.
* **Pattern Length Bifurcation (>= 64 bytes):** Long patterns benefit from sub-linear jumping because the skip distance overcomes the latency of memory table lookups.
* **Memory Slide Mode (< 64 bytes):** Short patterns bypass table lookups entirely to avoid pipeline-stalling memory dependencies, executing a 4x unrolled pure vector slide across the text buffer.

### 16-Bit Architecture (`CKmStrSearch16`)
UTF-16 characters (`wchar_t`) span a 65,536 value space. A naive BMH table would require 256KB (65536 * 4 bytes), guaranteed to blow out the CPU's L1 Data Cache (typically 32KB–48KB per core), leading to continuous L2/L3 cache misses.

~~~text
16-Bit Character (ch) 
        |
        v
 [ ch * 0x9E3779B9U ]  ---> Multiplicative Hash (Golden Ratio 2^32 / phi)
        |
   (hash >> 21)        ---> 11-Bit Extraction (0 .. 2047)
        |
        v
  [ 8KB Table ]        ---> 2,048-entry skip table designed to remain L1-cache resident
~~~

* **Golden Ratio Multiplicative Hashing:** Characters are hashed into an 11-bit index (2048 entries).
* **Cache Residency:** The table occupies 8KB (2048 * 4 bytes), keeping the table within a typical L1 D-cache footprint.
* **Collision Safety:** On bucket collisions, the smallest jump distance is stored. While hash collisions slightly reduce skip aggressiveness, correctness is preserved.
* **Wide Bifurcation (>= 128 characters):** Because hashing adds arithmetic latency, vector streaming is maintained up to 128 wide characters to maximize sequential throughput.

### Repetitive Substring Protection
To prevent worst-case O(N * M) degradation on highly repetitive streams (e.g., searching for `aaaaab` within `aaaaaaaaaaaaaa...`), the SIMD and SWAR loaders broadcast and check the **last character** of the needle rather than the first. This creates instant mismatch rejections in homogeneous text streams.

---

## 3. Execution Engines: SIMD vs. SWAR

~~~text
SIMD Vectorization Pipeline (4x Unrolled AVX2 / NEON)
+-----------------------------------------------------------------------------+
| Lane 0: [ 32 Bytes YMM0 ] === cmpeq(vLast) ===> Mask0                       |
| Lane 1: [ 32 Bytes YMM1 ] === cmpeq(vLast) ===> Mask1                       |
| Lane 2: [ 32 Bytes YMM2 ] === cmpeq(vLast) ===> Mask2  ===> OR ===> testz   |
| Lane 3: [ 32 Bytes YMM3 ] === cmpeq(vLast) ===> Mask3                       |
+-----------------------------------------------------------------------------+
~~~

### Hardware SIMD Engines (AVX2 & ARM64 NEON)
* **AVX2 (x86_64):** Evaluates 128 bytes per iteration using four unrolled YMM registers (`_mm256_cmpeq_epi8` / `_mm256_cmpeq_epi16`), combined through `_mm256_or_si256` and fast-rejected via `_mm256_testz_si256`. Candidate offsets are isolated with `_BitScanForward`.
* **NEON (ARM64):** Processes 64 bytes per iteration using four 128-bit vector registers (`vceqq_u8` / `vceqq_u16`). Emulates x86 movemask functionality by applying a power-of-two bitshift array (`vandq_u8`) followed by an across-vector vector addition (`vaddvq_u16` / `vaddv_u8`).

### SWAR Fallback (SIMD Within A Register)
When operating beneath the vector threshold or on platforms lacking AVX2, the engine uses 64-bit general-purpose registers to evaluate memory without invoking FPU state.

Using Mycroft’s bit-twiddling zero-byte detection algorithm:

~~~cpp
// 8-Bit SWAR Byte Matching:
ULONGLONG chunk = *reinterpret_cast<const ULONGLONG*>(ptr);
ULONGLONG v     = chunk ^ c8; // Matching bytes become 0x00
if (((v - 0x0101010101010101ULL) & ~v & 0x8080808080808080ULL) != 0) 
{
    // Zero-byte match detected in 8-byte word
}
~~~

Wide characters adapt the same technique across four 16-bit integers simultaneously using `0x0001000100010001ULL` and `0x8000800080008000ULL` masks.

---

## 4. The Experimental Branch: Why BNDM Falls Behind

Located in the `Experimental/` directory, `KmBndmSearch8` and `KmBndmSearch16` explore an alternative algorithmic theory based on **Backward Nondeterministic DAWG Matching (BNDM)**. 

These implementations are **not pure BNDM**. They are heavily hybridized to survive the constraints of the kernel environment:

* **SWAR Fast-Forwarding:** To prevent CPU branch prediction stalls and severe O(N*M) degradation on repetitive inputs, the algorithm does not blindly execute the automaton. Instead, it uses the GPR-based SWAR technique to rapidly scan for the *last* character of the pattern before engaging the backward automaton state machine. 
* **State Capacity Bifurcation:** The non-deterministic suffix automaton's state is packed entirely into a single 64-bit General Purpose Register (`ULONGLONG uD`). Therefore, patterns exceeding 64 characters bypass the BNDM logic entirely, defaulting to a fast SWAR linear slide or explicit vector streaming.
* **Bitmask Bloom Merging (16-bit):** To avoid an unfeasible 65,536-entry (512KB) array for 16-bit characters, the wide engine hashes characters into a 16KB (2,048-entry) table. Hash collisions are handled by logically ORing (`|=`) the bitmasks together. This essentially acts as a Bloom filter, allowing false positive prefix matches that are later discarded by a full-string verification loop.
* **SIMD Bypass:** BNDM is utilized strictly as the scalar fallback. For buffers large enough to justify the context switch overhead, BNDM is bypassed entirely in favor of a pure 4x unrolled AVX2/NEON memory slide.

Despite its favorable theoretical/algorithmic characteristics, **the BNDM variant consistently lags behind the Hybrid BMH implementation in kernel evaluations**. Hardware telemetry reveals four persistent architectural bottlenecks:

1. **Scalar Inner Loop Overhead:** The Boyer-Moore-Horspool algorithm computes multi-byte jumps with a single bad-character table lookup, whereas BNDM must repeatedly step backward, compute hash lookups, and execute bitwise AND/shift operations per character (`uD &= Mask[Hash(ch)]; uD <<= 1;`).
2. **Loop-Carried Data Dependencies:** BMH streams linear memory chunks that interact predictably with CPU branch predictors and prefetchers, while BNDM creates a strict sequential dependency on the single register state `uD`, stalling instruction-level parallelism (ILP).
3. **Wide-Character Hash Collision Penalties:** In BMH, 16-bit hash collisions only reduce the optimal jump distance, but in wide-character BNDM, collisions force bitmasks to logically merge (`|=`), generating false prefix matches that require expensive full-string verification loops.
4. **Pre-SIMD Scalar Threshold Deficit:** While both algorithms rely on identical 4x unrolled vector sliding for large buffers, BNDM lags in smaller buffers before the SIMD threshold is reached due to the heavy overhead of the automaton fallback.

---

## 5. Benchmark Analysis

To evaluate the exploration, testing was performed using a custom multi-threaded kernel test harness (`KmStrSearchShared.h`). 
The benchmark numbers are comparative measurements: each implementation is evaluated under the same workload, hardware, and test harness. They are not intended as universal throughput claims or as a replacement benchmark for arbitrary strstr() implementations. CRT strstr/wcsstr serves as the baseline because it provides a familiar reference implementation against which the experimental engines can be compared on identical workloads.

### Test Methodology & Patterns

* **Thread Pool Saturation:** Performance is measured by spinning up a pool of up to 32 kernel worker threads (via a custom `TEST_THREAD_MANAGER`).
* **Synchronized Execution:** Threads synchronize on a fast mutex before initiating the search loops simultaneously, ensuring the CPU and memory bus are completely saturated to calculate accurate GB/s throughput.
* **Buffer Scales:** Telemetry spans across four distinct buffer boundaries to evaluate performance both below and above the SIMD bifurcation thresholds: Tiny (80B/40B), Small (512B/256B), Medium (20KB/10KB), and Large (2MB/1MB).

The suite repeatedly executes three distinct corpus patterns to assess best-case, worst-case, and real-world scenarios:

1. **Standard Execution:** The text buffer is filled uniformly with `'A'` characters. The algorithm searches for `"Needle In The Haystack"` embedded five bytes from the end of the buffer to measure standard linear throughput.
2. **Mismatched Dense Text:** The text buffer is flooded with `'a'` characters. The search pattern is the highly repetitive prefix `"aaaaaaaaab"`, also embedded near the end. This tests pathological/repetitive mismatch workload speed and resilience against pathological CPU pipeline stalls.
3. **Realistic Log Corpus:** The buffer is populated with repeating simulated kernel logs: `"2026-08-24 10:56:00 [INFO] Process svchost.exe (PID: 1024) requested memory allocation. "`. The target pattern `"CRITICAL: USE_AFTER_FREE BUGCHECK"` is placed near the end of the log stream. This validates skip table entropy and evaluates the algorithm against realistic jump distances in the field.

### Environment 1: Intel Core i7-1165G7 (Bare-Metal Windows 11)

#### 16-Bit UTF-16 Benchmark Highlights

| Buffer Size | Scenario | Engine | Throughput | vs CRT Baseline |
| :--- | :--- | :--- | ---: | ---: |
| **Tiny (80 B)** | Standard | CRT (`wcsstr`) | 6.72 GB/s | Baseline |
| | | Scalar Hybrid (BMH) | 3.77 GB/s | 0.56x |
| | | AVX2 (Explicit) | 1.40 GB/s | **0.20x (XSTATE overhead)** |
| | Mismatched Dense | CRT (`wcsstr`) | 0.81 GB/s | Baseline |
| | | Scalar Hybrid (BMH) | 3.80 GB/s | **4.64x** |
| **Medium (20 KB)** | Standard | CRT (`wcsstr`) | 24.63 GB/s | Baseline |
| | | Smart Auto | 56.59 GB/s | **2.29x** |
| | Mismatched Dense | CRT (`wcsstr`) | 0.30 GB/s | Baseline |
| | | Smart Auto | 56.84 GB/s | **183.42x** |
| **Large (2 MB)** | Realistic Log | CRT (`wcsstr`) | 16.44 GB/s | Baseline |
| | | Smart Auto | 41.55 GB/s | **2.52x** |

#### 8-Bit UTF-8 Benchmark Highlights

* **Large Buffer Realistic Log:** Smart Auto achieved **81.22 GB/s** (3.91x speedup over CRT 20.75 GB/s).
* **Large Buffer Mismatched Dense:** Smart Auto reached **48.96 GB/s** vs. CRT 0.17 GB/s (**273.71x speedup**).

---

### Environment 2: Intel Core i7-8086K (VMware Guest Windows 11)

In the tested VMware environment, CRT strstr/wcsstr exhibited substantially lower throughput, consistent with an un-vectorized fallback path, magnifying the benefits of explicit Ring 0 vector sliding.

| Buffer Size | Scenario | Engine | Throughput | vs CRT Baseline |
| :--- | :--- | :--- | ---: | ---: |
| **Medium (20 KB)** | Standard (16-bit) | CRT (`wcsstr`) | 1.49 GB/s | Baseline |
| | | Smart Auto | 31.46 GB/s | **21.00x** |
| | | AVX2 (Explicit) | 48.66 GB/s | **32.47x** |
| **Large (2 MB)** | Mismatched Dense (8-bit) | CRT (`strstr`) | 0.22 GB/s | Baseline |
| | | Smart Auto | 49.72 GB/s | **222.87x** |
| | Realistic Log (8-bit) | CRT (`strstr`) | 1.35 GB/s | Baseline |
| | | Smart Auto | 52.73 GB/s | **38.99x** |

### Telemetry Takeaways
1. **The XSTATE overhead is Quantifiable:** On tiny 80-byte buffers, forcing AVX2 yields only 1.40 GB/s compared to 3.77 GB/s for the GPR-based SWAR fallback (a 63% penalty). The Smart Auto engine successfully navigates this.
2. **Degenerate Pattern Immunity:** Pathological inputs collapse standard C-runtime routines to ~0.2 GB/s. The last-character scanning heuristic maintains 40–56 GB/s, delivering over **270x speedups**.
3. **BMH vs. BNDM Verification:** As analyzed above, on realistic logs with small buffers (512 bytes), BMH scalar processes **7.94 GB/s** while BNDM’s automaton manages only **4.05 GB/s** (~96% faster for BMH).

### Interpretation
The measurements suggest that kernel substring-search performance is dominated not by a single universally superior algorithm, but by the interaction between buffer size, pattern characteristics, memory locality, and the cost of entering vector execution. The results motivate the hybrid routing strategy and explain why the BNDM variant, despite its attractive bit-parallel structure, did not outperform the simpler BMH-based design in the tested workloads.

---

## 6. Repository Structure

~~~text
├── Experimental/
│   ├── KmBndmSearch8.h      # 8-bit BNDM bit-parallel class declaration
│   ├── KmBndmSearch8.cpp    # 8-bit BNDM implementation
│   ├── KmBndmSearch16.h     # 16-bit BNDM bitmask bloom merger declaration
│   └── KmBndmSearch16.cpp   # 16-bit BNDM implementation
├── Shared/
│   └── KmStrSearchShared.h  # Unified Kernel/User multi-threaded test harness
├── StrSearch/
│   ├── KmStrSearch8.h       # 8-bit substring search class declaration
│   ├── KmStrSearch8.cpp     # AVX2, NEON, SWAR, and BMH implementation (8-bit)
│   ├── KmStrSearch16.h      # UTF-16 Hybrid BMH class declaration
│   └── KmStrSearch16.cpp    # Golden Ratio Hash, AVX2, NEON, SWAR (16-bit)
├── TestKm/
│   └── KmStrSearchDrv.cpp   # Kernel driver for test execution
└── TestUm/
    └── KmStrSearchUm.cpp    # User-mode test suite wrapper
~~~

---

## 7. API Reference & Usage

The classes `CKmStrSearch8` (for `char` / UTF-8) and `CKmStrSearch16` (for `wchar_t` / UTF-16) provide matching APIs.

~~~cpp
#include <ntddk.h>
#include "KmStrSearch16.h"

VOID SearchExample(const wchar_t* pKernelLogBuffer, size_t cchLogLength)
{
    PAGED_CODE(); // Required: Class allocates from Paged Pool

    CKmStrSearch16 searchEngine;
    const wchar_t needle[] = L"BUGCHECK_CODE_CRITICAL";

    // 1. Initialize table (PASSIVE_LEVEL or APC_LEVEL)
    if (!searchEngine.Initialize(needle, wcslen(needle)))
    {
        return; // Allocation failure or invalid length
    }

    // 2. Execute Search using 'Auto' routing
    int matchIndex = searchEngine.Find(pKernelLogBuffer, cchLogLength);

    if (matchIndex != -1)
    {
        // Pattern matched at pKernelLogBuffer[matchIndex]
    }
}
~~~

### Explicit Engine Selection

Callers can override the auto-bypass heuristic by explicitly specifying an execution engine:

~~~cpp
// Force scalar GPR execution (guarantees zero XSTATE context saving)
int idxScalar = searchEngine.Find(pBuffer, cchLen, CKmStrSearch16::SearchEngine::Scalar);

// Explicitly execute AVX2 (automatically wraps KeSaveExtendedProcessorState)
int idxAVX2 = searchEngine.Find(pBuffer, cchLen, CKmStrSearch16::SearchEngine::AVX2);
~~~

---

## 8. Building & Testing

### Prerequisites
* **Visual Studio 2026**
* **C++20 Toolset Support:** Ensure `/std:c++20` is enabled in your compilation flags.
* **Windows Driver Kit (WDK):** The latest MS WDK is required to compile the kernel driver targets.

### Compilation
Native MSVC 2026 solution and project files are provided in the repository to build both the test kernel driver and the user mode test suite code.

* **TestKm (Kernel Driver):** Compiles the `KmStrSearchDrv.sys` test kernel driver containing the complete performance and correctness test suite. This represents the primary evaluation environment.
* **TestUm (User Mode):** Compiles the user mode test suite code wrapping the shared test logic, designed exclusively for rapid functional validation, unit testing, and debugging.

### Execution & Telemetry Output
* **Live Tracing:** The kernel driver outputs benchmark results and correctness validations in real-time using `DbgPrintEx` traces. These logs are broadcast under the `DPFLTR_IHVDRIVER_ID` component filter using `DPFLTR_INFO_LEVEL` and `DPFLTR_ERROR_LEVEL` levels, making them easily viewable via WinDbg or Sysinternals DebugView.
* **Persistent Logging:** At the conclusion of the test suite, the driver flushes all accumulated output strings and saves the complete telemetry report to a text file located at `\SystemRoot\Temp\KmStrSearchPerf.txt`.

**⚠️ Performance Caveat:** While the user-mode (`TestUm`) test suite code is provided for convenience and logic verification, its performance metrics **cannot be trusted** to reflect true system capabilities. The search engines, specifically the dynamic SIMD/SWAR routing logic and state-saving bypasses, are designed exclusively for the Windows kernel architecture. User-mode environments handle thread context switching and vector register preservation entirely differently. Always refer to the `KmStrSearchDrv.sys` kernel driver telemetry for accurate Ring 0 performance evaluations.

---

## 9. License

This project is licensed under the MIT License. See the `LICENSE` file for details.