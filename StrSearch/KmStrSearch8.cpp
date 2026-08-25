// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// KmStrSearch8.cpp
//
// ALGORITHM OVERVIEW:
// This file implements a Hybrid Boyer-Moore-Horspool (BMH) string searching 
// algorithm, optimized for the Windows kernel environment. It dynamically 
// selects between a GPR-based SWAR (SIMD Within A Register) scalar fallback 
// and true hardware SIMD (AVX2/NEON) vectorization based on buffer sizes and 
// pattern lengths.
//
// THE HYBRID APPROACH:
// 1. Skip Table (BMH): A flat 256-entry array precomputes the bad-character 
//    jumps for all possible 8-bit ASCII/UTF-8 values. 
// 2. Length Bifurcation: 
//    - Long Patterns (>= 128 bytes): Benefit massively from BMH sub-linear 
//      skipping. The algorithm uses the skip table to jump ahead, and upon 
//      finding a potential match, uses SIMD to verify the surrounding bytes.
//    - Short Patterns (< 128 bytes): BMH skip lookups create memory data 
//      dependencies that stall the CPU pipeline. For short strings, we ignore 
//      the skip table and execute a 4x unrolled pure SIMD memory slide (128 bytes 
//      for AVX2, 64 bytes for NEON), maximizing memory bandwidth over algorithmic
//      complexity.
// 3. SWAR Fallback: For environments where FPU/SIMD state saving is too costly,
//    we broadcast the 8-bit target across a 64-bit General Purpose Register (GPR).
//    Using bitwise XOR and magic masks (0x0101... and 0x8080...), we detect 
//    zero-bytes in 8-byte chunks without invoking XMM/YMM registers.
//
// KERNEL ARCHITECTURE CONSIDERATIONS:
// 1. Context Switch Overhead: Using floating-point/SIMD registers in Ring 0 
//    requires explicitly saving the thread's processor state via KPCR 
//    (KeSaveExtendedProcessorState). For small buffers, this overhead vastly 
//    exceeds the gains of vectorization. Thus, an auto-bypass is implemented.
// 2. IRQL Constraints: ExAllocatePool2 with POOL_FLAG_PAGED and PAGED_CODE() 
//    macros enforce that this class must be used at PASSIVE_LEVEL or APC_LEVEL.
// 3. Pipeline Stalls: To prevent branch mispredictions and stalls on repeating 
//    patterns (e.g., "aaaaab"), the SIMD engines broadcast and scan for the 
//    LAST character of the pattern rather than the first.
// -------------------------------------------------------------------------------------
#include <ntddk.h>
#include <intrin.h>
#include <limits.h>
#include "KmStrSearch8.h"

#if defined(_M_X64) || defined(_M_IX86)
    #include <immintrin.h>
#elif defined(_M_ARM64)
    #include <arm64_neon.h>

    // Satisfy MSVC compiler dependency for NEON usage in Ring 0.
    #ifdef __cplusplus
    extern "C" 
    {
    #endif
        __declspec(selectany) int _fltused = 0;
    #ifdef __cplusplus
    }
    #endif
#endif

// -------------------------------------------------------------------------------------
// CKmStrSearch8 classImplementation
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// SWAR (SIMD Within A Register) fallback.
// This executes a fast memory scan utilizing only General Purpose Registers (GPRs).
// By avoiding XMM/YMM registers, we completely bypass the need for 
// KeSaveExtendedProcessorState, making this optimal for short localized scans or
// legacy hardware.
// -------------------------------------------------------------------------------------
inline const char* CKmStrSearch8::MemChrScalar(const char* __restrict ptr,
                                               char                   val, 
                                               size_t                 cchNum)
{
    PAGED_CODE();

    const char* pEnd = ptr + cchNum;

#if defined(_M_X64) || defined(_M_ARM64)
    // 64-bit optimization: Broadcast the 8-bit target byte across a 64-bit GPR integer.
    // This allows us to evaluate 8 bytes per CPU cycle without utilizing SSE/AVX.
    ULONGLONG c8 = static_cast<unsigned char>(val);
    c8 |= c8 << 8;
    c8 |= c8 << 16;
    c8 |= c8 << 32;

    // Magic numbers for SWAR zero-byte detection (Mycroft's bit-twiddling hack)
    const ULONGLONG magic1 = 0x0101010101010101ULL;
    const ULONGLONG magic8 = 0x8080808080808080ULL;

    // Process 8 bytes at a time (strictly bounds-checked to prevent kernel page faults)
    while (ptr + 8 <= pEnd)
    {
        ULONGLONG chunk = *reinterpret_cast<const ULONGLONG*>(ptr);
        
        // XOR sets the matching bytes to 0x00. We then use SWAR to find any 0x00 byte.
        ULONGLONG v = chunk ^ c8; 

        // Evaluates to non-zero if any byte in the 64-bit word is 0x00
        if (((v - magic1) & ~v & magic8) != 0) [[unlikely]]
        {
            // A match exists in this 8-byte chunk. Break out to pinpoint it sequentially.
            break; 
        }

        ptr += 8;
    }
#endif

    // Pinpoint the exact match location (or handle short tails / 32-bit architectures)
    while (ptr < pEnd)
    {
        if (*ptr == val) [[unlikely]]
        {
            return ptr;
        }

        ++ptr;
    }

    return nullptr;
}

// -------------------------------------------------------------------------------------
// Validates the inner section of a pattern once the start/end bounds are confirmed.
// -------------------------------------------------------------------------------------
inline bool CKmStrSearch8::VerifyMiddleMatch(const char* __restrict Text,
                                             const char* __restrict Pattern, 
                                             size_t                 cchPatternLen)
{
    PAGED_CODE();

    // If pattern is 1 or 2 chars, the start/end bounds checks already validated the entire string.
    if (cchPatternLen <= 2) [[unlikely]]
    {
        return true;
    }

    // Linear scan of the inner bytes. Branch predictor expects matches here.
    for (size_t uI = 1; uI < cchPatternLen - 1; ++uI)
    {
        if (Text[uI] != Pattern[uI]) [[likely]]
        {
            return false;
        }
    }

    return true;
}

// -------------------------------------------------------------------------------------
// Constructor initializes internal state flags for CPU feature support.
// -------------------------------------------------------------------------------------
CKmStrSearch8::CKmStrSearch8() : m_Pattern(nullptr), 
                                 m_cchPatternLen(0), 
                                 m_SkipTable(nullptr)
{
    PAGED_CODE();

#if defined(_M_X64) || defined(_M_IX86)    
    m_bHasAVX2 = false;
#endif
}

// Destructor cleans up the BMH skip table from the paged pool.
CKmStrSearch8::~CKmStrSearch8()
{
    if (m_SkipTable)
    {
        ExFreePoolWithTag(m_SkipTable, POOL_TAG);
    }
}

// -------------------------------------------------------------------------------------
// Initializes the search engine by precomputing the Boyer-Moore-Horspool skip table.
// Precomputes the Boyer-Moore-Horspool bad-character jump table.
// This function must be called before invoking Find() and should be executed 
// at PASSIVE_LEVEL or APC_LEVEL due to kernel pool allocations.
//
// Parameters:
//   Pattern   - A pointer to the 8-bit character array to search for.
//               WARNING: The engine performs a shallow copy of this pointer. 
//               The caller MUST ensure this memory remains valid and resident 
//               (non-paged or locked) for the entire lifetime of this object 
//               to prevent Use-After-Free (UAF) bugchecks.
//   cchLength - The length of the Pattern in characters. Must not exceed INT_MAX.
//
// Returns:
//   true if the skip table was successfully allocated and populated; otherwise, false.
// -------------------------------------------------------------------------------------

bool CKmStrSearch8::Initialize(const char* __restrict Pattern, 
                               size_t                 cchLength)
{
    PAGED_CODE();

    if (!Pattern || cchLength == 0) [[unlikely]]
    {
        return false;
    }

    // EDGE CASE: Ensure length doesn't overflow our signed int skip table index.
    // Prevents integer wrap-around vulnerabilities during kernel memory arithmetic.
    if (cchLength > INT_MAX) [[unlikely]]
    {
        return false; 
    }

#if defined(_M_X64) || defined(_M_IX86)    
    // Dynamically query the KUSER_SHARED_DATA or CPUID wrapper for AVX2 support
    m_bHasAVX2 = ExIsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE);
#endif

    m_Pattern       = Pattern;
    m_cchPatternLen = cchLength;

    // Allocate 1KB flat array for the 256-byte ASCII/UTF-8 lookup map.
    // Uses POOL_FLAG_CACHE_ALIGNED to prevent false sharing and ensure optimal L1 D-Cache loads.
    m_SkipTable = static_cast<int*>(ExAllocatePool2(POOL_FLAG_PAGED | POOL_FLAG_UNINITIALIZED | POOL_FLAG_CACHE_ALIGNED, 
                                                    TABLE_SIZE * sizeof(int), 
                                                    POOL_TAG));    
    if (!m_SkipTable) [[unlikely]]
    {
        return false;
    }

    int iDefaultSkip = static_cast<int>(m_cchPatternLen);
    
    for (ULONG i = 0; i < TABLE_SIZE; ++i)
    {
        m_SkipTable[i] = iDefaultSkip;
    }

    // Populate Boyer-Moore bad-character jumps. 
    // Further occurrences of a char overwrite earlier ones (rightmost takes precedence).
    for (size_t uI = 0; uI < m_cchPatternLen - 1; ++uI)
    {
        unsigned char ch = static_cast<unsigned char>(Pattern[uI]);
        m_SkipTable[ch]  = static_cast<int>(m_cchPatternLen - 1 - uI);
    }

    return true;
}

// -------------------------------------------------------------------------------------
// Executes the string search algorithm against the provided text buffer.
// Dynamically routes to the optimal execution engine (Scalar, AVX2, or NEON) 
// based on the CPU architecture, buffer length, and configuration.
//
// Parameters:
//   Text       - A pointer to the 8-bit character buffer to be scanned.
//                Must be resident in non-paged memory if called at DISPATCH_LEVEL.
//   cchTextLen - The length of the Text buffer in characters.
//   Engine     - The execution strategy to use (defaults to Auto).
//                'Auto' intelligently bypasses the SIMD XSTATE context 
//                switch overhead for small buffer sizes.
//
// Returns:
//   The zero-based index of the first character of the matched pattern 
//   within the Text buffer, or -1 if the pattern was not found.
// -------------------------------------------------------------------------------------
int CKmStrSearch8::Find(const char* __restrict Text,
                        size_t                 cchTextLen, 
                        SearchEngine           Engine) const
{
    PAGED_CODE();

    if (!Text || m_cchPatternLen == 0 || cchTextLen < m_cchPatternLen) [[unlikely]]
    {
        return -1;
    }

    if (Engine == SearchEngine::Auto) [[likely]]
    {
        return FindKernel(Text, cchTextLen);
    }

    if (Engine == SearchEngine::Scalar) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    return RunEngineWithSafety(Text, cchTextLen, Engine);
}

// -------------------------------------------------------------------------------------
// Routes execution based on buffer size heuristics.
// -------------------------------------------------------------------------------------
int CKmStrSearch8::FindKernel(const char* __restrict Text,
                              size_t                 cchTextLen) const
{
    PAGED_CODE();

    // 1. Small Buffer Processing (Bypass the XSTATE AVX saving overhead)
    // Invoking KeSaveExtendedProcessorState is heavily penalizing on the CPU pipeline.
    if (cchTextLen <= KERNEL_THRESHOLD_CHARS) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    // 2. Large Buffer Processing
#if defined(_M_X64) || defined(_M_IX86)
    if (m_bHasAVX2) [[likely]]
    {
        return RunEngineWithSafety(Text, cchTextLen, SearchEngine::AVX2);
    }

    // 3. Absolute Fallback (Legacy Hardware without AVX2/NEON support)
    return FindScalar(Text, cchTextLen);

#elif defined(_M_ARM64)
    return RunEngineWithSafety(Text, cchTextLen, SearchEngine::NEON);
#endif
}

// -------------------------------------------------------------------------------------
// Manages the OS context switch mechanics required to use SIMD instruction sets 
// safely in Ring 0
// -------------------------------------------------------------------------------------
int CKmStrSearch8::RunEngineWithSafety(const char* __restrict Text, 
                                       size_t                 cchTextLen, 
                                       SearchEngine           Engine) const
{
    PAGED_CODE();

#if defined(_M_X64) || defined(_M_IX86)
    if (Engine == SearchEngine::AVX2 && !m_bHasAVX2) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    XSTATE_SAVE SaveState;    
    
    // Explicitly reserve the YMM (AVX) registers. This backs up the current thread's floating point state.
    if (NT_SUCCESS(KeSaveExtendedProcessorState(XSTATE_MASK_AVX, &SaveState))) [[likely]]
    {
        int iResult = FindAVX2(Text, cchTextLen);
        KeRestoreExtendedProcessorState(&SaveState);
        
        return iResult;
    }
#elif defined(_M_ARM64)
    if (Engine == SearchEngine::NEON) [[likely]]
    {
        KFLOATING_SAVE SaveState;
        
        // Save ARM64 NEON state (FPSR/FPCR).
        if (NT_SUCCESS(KeSaveFloatingPointState(&SaveState))) [[likely]]
        {
            int iResult = FindNEON(Text, cchTextLen);
            KeRestoreFloatingPointState(&SaveState);
            
            return iResult;
        }
    }
#endif

    // Fallback to integer instructions if kernel refuses to save float state (e.g., IRQL issues).
    return FindScalar(Text, cchTextLen);
}

// -------------------------------------------------------------------------------------
// Pure integer based Boyer-Moore-Horspool algorithm.
// -------------------------------------------------------------------------------------
int CKmStrSearch8::FindScalar(const char* __restrict Text, 
                              size_t                 cchTextLen) const
{
    PAGED_CODE();

    // Fast-path for single char patterns.
    if (m_cchPatternLen == 1) [[unlikely]]
    {
        const char* pMatch = MemChrScalar(Text, m_Pattern[0], cchTextLen);
        
        if (pMatch) [[unlikely]]
        {
            return static_cast<int>(pMatch - Text);
        }
        else
        {
            return -1;
        }
    }

    const size_t uLastIdx = m_cchPatternLen - 1;
    const char   lastChar = m_Pattern[uLastIdx];

    size_t uK = uLastIdx;

    // Scan Text using BMH skips.
    while (uK < cchTextLen)
    {
        unsigned char ch = static_cast<unsigned char>(Text[uK]);

        if (Text[uK] != lastChar) [[likely]]
        {
            int skip = m_SkipTable[ch];

            // OPTIMIZATION: Fast-forward dense sub-match areas via GPR scanning.
            // If the BMH skip is poor (<= 2), we are likely thrashing in dense partial matches.
            // Switch to SWAR processing to linearly advance without branch penalties.
            if (skip <= 2) [[unlikely]]
            {
                const char* pMatch = MemChrScalar(Text + uK + 1, lastChar, cchTextLen - uK - 1);                
                
                if (!pMatch) [[unlikely]]
                {
                    return -1;
                }

                uK = pMatch - Text;
            }
            else
            {
                uK += skip;
            }

            continue;
        }

        // Potential match found. Verify backwards.
        size_t uJ = uLastIdx - 1;
        size_t uI = uK - 1;

        while (Text[uI] == m_Pattern[uJ])
        {
            if (uJ == 0) [[unlikely]]
            {
                return static_cast<int>(uI);
            }

            --uI;
            --uJ;
        }

        // Shift by the bad character rule on failure.
        uK += m_SkipTable[ch];
    }

    return -1;
}

#if defined(_M_X64) || defined(_M_IX86)

// -------------------------------------------------------------------------------------
// High throughput SIMD implementation using AVX2.
// -------------------------------------------------------------------------------------
int CKmStrSearch8::FindAVX2(const char* __restrict Text, 
                            size_t                 cchTextLen) const
{
    PAGED_CODE();

    // Requires minimum 128 bytes due to the 4-register unroll strategy
    if (cchTextLen < 128 || m_cchPatternLen < 2) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    const size_t uLastIdx = m_cchPatternLen - 1;
    
    // Broadcast the LAST character instead of the FIRST to prevent 'aaaaab' pipeline stalls
    __m256i vLast = _mm256_set1_epi8(m_Pattern[uLastIdx]);
    size_t  uI    = 0;
    
    // Helper lambda to process match mask.
    auto CheckMask = [&](__m256i cmp, 
                         size_t  uLaneOffset) -> int 
    {
        // Extract 32-bit mask from YMM comparison result.
        unsigned int final_mask = _mm256_movemask_epi8(cmp);

        while (final_mask != 0)
        {
            ULONG bitIndex;
            _BitScanForward(&bitIndex, final_mask); // Use hardware BSF to find exact match byte.
            
            // The offset points to where the LAST character matched. Subtract uLastIdx to find pattern start.
            size_t matchEndOffset = uLaneOffset + bitIndex;
            size_t startOffset    = matchEndOffset - uLastIdx;

            // Fast scalar check of the FIRST character before invoking the middle-match loop
            if (Text[startOffset] == m_Pattern[0] && 
                VerifyMiddleMatch(Text + startOffset, m_Pattern, m_cchPatternLen)) [[unlikely]]
            {
                return static_cast<int>(startOffset);
            }
            
            final_mask &= (final_mask - 1); // Clear the lowest set bit and continue.
        }
        
        return -1;
    };

    // BIFURCATION: Long patterns benefit from Boyer-Moore skipping,
    // while short-to-medium patterns stall out under skip processing and execute faster as a pure memory slide.
    // Raised to 128 bytes so BMH jumps mathematically exceed the 128-byte unrolled SIMD stride.
    if (m_cchPatternLen >= 128)
    {
        // 2x Unroll (64 chars) for safer chunking before a jump
        while (uI + 63 + uLastIdx < cchTextLen)
        {
            // Shift the YMM load forward by uLastIdx
            __m256i b0    = _mm256_loadu_si256((const __m256i*)&Text[uI + uLastIdx]);
            __m256i b1    = _mm256_loadu_si256((const __m256i*)&Text[uI + 32 + uLastIdx]);

            __m256i c0    = _mm256_cmpeq_epi8(b0, vLast);
            __m256i c1    = _mm256_cmpeq_epi8(b1, vLast);
            __m256i cBoth = _mm256_or_si256(c0, c1);

            if (_mm256_testz_si256(cBoth, cBoth) == 0) [[unlikely]]
            {
                int iResult;
                
                if ((iResult = CheckMask(c0, uI + uLastIdx)) != -1) [[unlikely]]
                {
                    return iResult;
                }
                
                if ((iResult = CheckMask(c1, uI + 32 + uLastIdx)) != -1) [[unlikely]]
                {
                    return iResult;
                }
            }

            // Retrieve the last scanned byte to look up the jump skip.
            unsigned char badChar = static_cast<unsigned char>(Text[uI + 63 + uLastIdx]);
            uI += 63 + m_SkipTable[badChar];
        }
    }
    else
    {
        // Short patterns: Maximize Memory Bandwidth with 4x Unrolled (128-char) scanning.
        // Pure memory slide without BMH skip checking to keep instruction pipeline completely saturated.
        while (uI + 127 + uLastIdx < cchTextLen)
        {
            // Shift the YMM load forward by uLastIdx
            __m256i b0 = _mm256_loadu_si256((const __m256i*)&Text[uI + uLastIdx]);
            __m256i b1 = _mm256_loadu_si256((const __m256i*)&Text[uI + 32 + uLastIdx]);
            __m256i b2 = _mm256_loadu_si256((const __m256i*)&Text[uI + 64 + uLastIdx]);
            __m256i b3 = _mm256_loadu_si256((const __m256i*)&Text[uI + 96 + uLastIdx]);

            __m256i c0 = _mm256_cmpeq_epi8(b0, vLast);
            __m256i c1 = _mm256_cmpeq_epi8(b1, vLast);
            __m256i c2 = _mm256_cmpeq_epi8(b2, vLast);
            __m256i c3 = _mm256_cmpeq_epi8(b3, vLast);

            __m256i or01  = _mm256_or_si256(c0, c1);
            __m256i or23  = _mm256_or_si256(c2, c3);
            __m256i orAll = _mm256_or_si256(or01, or23);

            if (_mm256_testz_si256(orAll, orAll) == 0) [[unlikely]]
            {
                int iResult;
                
                if ((iResult = CheckMask(c0, uI + uLastIdx)) != -1) [[unlikely]]
                {
                    return iResult;
                }
                
                if ((iResult = CheckMask(c1, uI + 32 + uLastIdx)) != -1) [[unlikely]]
                {
                    return iResult;
                }
                
                if ((iResult = CheckMask(c2, uI + 64 + uLastIdx)) != -1) [[unlikely]]
                {
                    return iResult;
                }
                
                if ((iResult = CheckMask(c3, uI + 96 + uLastIdx)) != -1) [[unlikely]]
                {
                    return iResult;
                }
            }
            
            uI += 128; 
        }
    }

    // Tail processing for trailing bytes outside the unrolled alignment.
    if (uI < cchTextLen) [[unlikely]]
    {
        int iTailRes = FindScalar(Text + uI, cchTextLen - uI);        
        
        if (iTailRes != -1) [[unlikely]]
        {
            return static_cast<int>(uI + iTailRes);
        }
    }

    return -1;
}
#elif defined(_M_ARM64)

// -------------------------------------------------------------------------------------
// ARM64 specific optimization utilizing 128-bit vector NEON instructions.
// -------------------------------------------------------------------------------------
int CKmStrSearch8::FindNEON(const char* __restrict Text, 
                            size_t                 cchTextLen) const
{
    PAGED_CODE();

    if (cchTextLen < 128 || m_cchPatternLen < 2) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    const size_t uLastIdx = m_cchPatternLen - 1;
    
    // Broadcast the LAST character instead of the FIRST to prevent 'aaaaab' pipeline stalls
    uint8x16_t vLast = vdupq_n_u8(static_cast<unsigned char>(m_Pattern[uLastIdx]));

    // Emulate x86 _mm_movemask_epi8 since NEON doesn't have an exact equivalent.
    auto neon_movemask_u8 = [](uint8x16_t cmp) -> unsigned int 
    {
        alignas(16) static const unsigned char shift_arr[16] = { 1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128 };
        uint8x16_t shift  = vld1q_u8(shift_arr);
        uint8x16_t masked = vandq_u8(cmp, shift);
        
        unsigned int low  = vaddv_u8(vget_low_u8(masked));
        unsigned int high = vaddv_u8(vget_high_u8(masked));
        
        return low | (high << 8);
    };

    auto CheckLane = [&](uint8x16_t cmp, 
                         size_t     uLaneOffset) -> int 
    {
        if (vmaxvq_u32(vreinterpretq_u32_u8(cmp)) != 0) [[unlikely]]
        {
            unsigned int final_mask = neon_movemask_u8(cmp);

            while (final_mask != 0)
            {
                ULONG bitIndex;
                _BitScanForward(&bitIndex, final_mask);
                
                // The offset points to the LAST character match. Subtract uLastIdx.
                size_t matchEndOffset = uLaneOffset + bitIndex;
                size_t startOffset    = matchEndOffset - uLastIdx;
                
                if (Text[startOffset] == m_Pattern[0] && 
                    VerifyMiddleMatch(Text + startOffset, m_Pattern, m_cchPatternLen)) [[unlikely]]
                {
                    return static_cast<int>(startOffset);
                }
                
                final_mask &= (final_mask - 1); 
            }
        }
        
        return -1;
    };

    auto ProcessBlock = [&](size_t uCurrentI) -> int 
    {
        // Shift the NEON load forward by uLastIdx
        // 4x Vector Unroll (64 bytes total for NEON)
        uint8x16_t v0 = vld1q_u8(reinterpret_cast<const unsigned char*>(&Text[uCurrentI + uLastIdx]));
        uint8x16_t v1 = vld1q_u8(reinterpret_cast<const unsigned char*>(&Text[uCurrentI + 16 + uLastIdx]));
        uint8x16_t v2 = vld1q_u8(reinterpret_cast<const unsigned char*>(&Text[uCurrentI + 32 + uLastIdx]));
        uint8x16_t v3 = vld1q_u8(reinterpret_cast<const unsigned char*>(&Text[uCurrentI + 48 + uLastIdx]));

        uint8x16_t cmp0 = vceqq_u8(v0, vLast);
        uint8x16_t cmp1 = vceqq_u8(v1, vLast);
        uint8x16_t cmp2 = vceqq_u8(v2, vLast);
        uint8x16_t cmp3 = vceqq_u8(v3, vLast);
        
        uint8x16_t cmp01  = vorrq_u8(cmp0, cmp1);
        uint8x16_t cmp23  = vorrq_u8(cmp2, cmp3);
        uint8x16_t cmpAll = vorrq_u8(cmp01, cmp23);

        // Fast vector rejection checking multiple lanes concurrently.
        if (vmaxvq_u32(vreinterpretq_u32_u8(cmpAll)) != 0) [[unlikely]]
        {
            int iResult;
            
            if ((iResult = CheckLane(cmp0, uCurrentI + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }

            if ((iResult = CheckLane(cmp1, uCurrentI + 16 + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }

            if ((iResult = CheckLane(cmp2, uCurrentI + 32 + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }

            if ((iResult = CheckLane(cmp3, uCurrentI + 48 + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }
        }
        
        return -1;
    };

    size_t uI = 0;

    // BIFURCATION: Raised to 128 bytes so BMH jumps mathematically exceed the 64-byte unrolled NEON stride.
    if (m_cchPatternLen >= 128)
    {
        while (uI + 63 + uLastIdx < cchTextLen)
        {
            int iMatch = ProcessBlock(uI);            
            
            if (iMatch != -1) [[unlikely]]
            {
                return iMatch;
            }

            unsigned char badChar = static_cast<unsigned char>(Text[uI + 63 + uLastIdx]);
            uI += 63 + m_SkipTable[badChar];
        }
    }
    else
    {
        while (uI + 63 + uLastIdx < cchTextLen)
        {
            int iMatch = ProcessBlock(uI);            
            
            if (iMatch != -1) [[unlikely]]
            {
                return iMatch;
            }

            uI += 64; 
        }
    }

    // Scalar fail-safe for remaining tail bytes not cleanly divisible by vector strides.
    if (uI < cchTextLen) [[unlikely]]
    {
        int iTailRes = FindScalar(Text + uI, cchTextLen - uI);        
        
        if (iTailRes != -1) [[unlikely]]
        {
            return static_cast<int>(uI + iTailRes);
        }
    }

    return -1;
}

#endif