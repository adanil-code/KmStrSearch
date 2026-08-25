// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// KmStrSearch16.cpp
//
// ALGORITHM OVERVIEW:
// Extends the Hybrid Boyer-Moore-Horspool (BMH) algorithm to process 16-bit 
// wide characters (wchar_t), handling Unicode strings common in the Windows Kernel.
// 
// THE WIDE-CHARACTER HYBRID APPROACH:
// 1. Golden Ratio Hash Skip Table: A traditional BMH table for wchar_t requires 
//    65,536 entries (256KB), which guarantees L1 cache misses and degrades 
//    performance. Instead, we use a multiplicative Golden Ratio hash to fold 
//    the 16-bit characters into a dense 2048-entry table (8KB) that locks 
//    into the L1 D-Cache.
// 2. Length Bifurcation: 
//    - Long Patterns (>= 128 wchar_ts): Utilize the hashed BMH bad-character 
//      rule to jump forward, mitigating data dependency stalls by avoiding 
//      redundant contiguous memory reads.
//    - Short Patterns (< 128 wchar_ts): Bypass the hash lookups entirely and 
//      leverage an unrolled SIMD vector slide (checking 128 bytes / 64 wchar_ts 
//      per iteration using AVX2 or 64 bytes / 32 wchar_ts using NEON).
// 3. Wide SWAR Fallback: The scalar path broadcasts a 16-bit wchar_t across a 
//    64-bit GPR. We XOR the memory chunk and use wide magic masks 
//    (0x00010001... and 0x80008000...) to evaluate four wide characters per 
//    clock cycle without engaging the floating-point unit.
//
// KERNEL ARCHITECTURE CONSIDERATIONS:
// 1. Cache Thrashing: The 8KB BMH skip table fits inside L1 D-Cache, 
//    retaining fast lookup operations without invalidating the TLB.
// 2. State-Saving Overheads: Identical to the 8-bit implementation, AVX2 and 
//    NEON vectorization paths validate buffer lengths to bypass 
//    XSTATE / KFLOATING_SAVE saving when memory footprints are too small to 
//    justify the context switch overhead.
// -------------------------------------------------------------------------------------
#include <ntddk.h>
#include <intrin.h>
#include <limits.h>
#include "KmStrSearch16.h"

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
// CKmStrSearch16 class Implementation
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// Hand-tuned SWAR scalar fallback that stays strictly within general purpose registers
// (GPRs). Accelerates wide character searching without incurring XMM/YMM context switch
// costs.
// -------------------------------------------------------------------------------------
inline const wchar_t* CKmStrSearch16::WMemChrScalar(const wchar_t* __restrict ptr,
                                                    wchar_t                   val, 
                                                    size_t                    cchNum)
{
    PAGED_CODE();

    const wchar_t* pEnd = ptr + cchNum;

#if defined(_M_X64) || defined(_M_ARM64)
    // 64-bit optimization: Broadcast the 16-bit target wchar_t across an 8-byte integer.
    ULONGLONG c4 = static_cast<USHORT>(val);
    c4 |= c4 << 16;
    c4 |= c4 << 32;

    const ULONGLONG magic1 = 0x0001000100010001ULL;
    const ULONGLONG magic8 = 0x8000800080008000ULL;

    // Process 4 wchar_ts (8 bytes) at a time to maximize memory throughput.
    while (ptr + 4 <= pEnd)
    {
        ULONGLONG chunk = *reinterpret_cast<const ULONGLONG*>(ptr);
        
        // XOR isolates the matching words to 0x0000.
        ULONGLONG v = chunk ^ c4;

        // SWAR technique checks if any 16-bit block in the 64-bit word is 0x0000.
        if (((v - magic1) & ~v & magic8) != 0) [[unlikely]]
        {
            break;
        }

        ptr += 4;
    }
#endif

    // Iterate the final remaining elements precisely.
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
// Linear scalar verification loop for inner characters of a matched candidate.
// -------------------------------------------------------------------------------------
inline bool CKmStrSearch16::VerifyMiddleMatch(const wchar_t* __restrict Text,
                                              const wchar_t* __restrict Pattern, 
                                              size_t                    cchPatternLen)
{
    PAGED_CODE();

    if (cchPatternLen <= 2) [[unlikely]]
    {
        return true;
    }

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
// Default Constructor handles processor feature identification state.
// -------------------------------------------------------------------------------------
CKmStrSearch16::CKmStrSearch16() : m_Pattern(nullptr), 
                                   m_cchPatternLen(0), 
                                   m_SkipTable(nullptr)
{
    PAGED_CODE();

#if defined(_M_X64) || defined(_M_IX86)    
    m_bHasAVX2 = false;
#endif
}

// -------------------------------------------------------------------------------------
// Destructor ensures non-leaking memory release from the Windows pool mechanism.
// -------------------------------------------------------------------------------------
CKmStrSearch16::~CKmStrSearch16()
{
    if (m_SkipTable)
    {
        ExFreePoolWithTag(m_SkipTable, POOL_TAG);
    }
}

// -------------------------------------------------------------------------------------
// Initializes the search engine by precomputing a hashed Boyer-Moore-Horspool skip
// table.
// This function must be called before invoking Find() and should be executed 
// at PASSIVE_LEVEL or APC_LEVEL due to kernel pool allocations.
//
// Parameters:
//   Pattern   - A pointer to the 16-bit wide character array to search for.
//               WARNING: The engine performs a shallow copy of this pointer. 
//               The caller MUST ensure this memory remains valid and resident 
//               (non-paged or locked) for the entire lifetime of this object 
//               to prevent Use-After-Free (UAF) bugchecks.
//   cchLength - The length of the Pattern in wide characters. Must not exceed INT_MAX.
//
// Returns:
//   true if the hash skip table was successfully allocated and populated; otherwise, 
//   false.
// -------------------------------------------------------------------------------------
bool CKmStrSearch16::Initialize(const wchar_t* __restrict Pattern, 
                                size_t                    cchLength)
{
    PAGED_CODE();

    if (!Pattern || cchLength == 0) [[unlikely]]
    {
        return false;
    }

    // EDGE CASE PROTECTION: Ensure pattern length doesn't overflow the 32-bit signed int skip table.
    // Guards against buffer arithmetic wrap-arounds within the driver address space.
    if (cchLength > INT_MAX) [[unlikely]]
    {
        return false; 
    }

#if defined(_M_X64) || defined(_M_IX86)    
    m_bHasAVX2 = ExIsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE);
#endif

    m_Pattern       = Pattern;
    m_cchPatternLen = cchLength;

    // Allocate 8KB flat array. Fits neatly in L1 D-cache footprint.
    // Pool flags indicate cache-aligned memory mapping in the driver Paged Pool.
    m_SkipTable = static_cast<int*>(ExAllocatePool2(POOL_FLAG_PAGED | POOL_FLAG_UNINITIALIZED | POOL_FLAG_CACHE_ALIGNED, 
                                                    HASH_TABLE_SIZE * sizeof(int), 
                                                    POOL_TAG));    
    if (!m_SkipTable) [[unlikely]]
    {
        return false;
    }

    int defaultSkip = static_cast<int>(m_cchPatternLen);
    
    for (ULONG i = 0; i < HASH_TABLE_SIZE; ++i)
    {
        m_SkipTable[i] = defaultSkip;
    }

    // Populate Boyer-Moore bad-character rule with hashing.
    // The last character is intentionally omitted so it doesn't resolve to a skip of 0.
    for (size_t i = 0; i < m_cchPatternLen - 1; ++i)
    {
        USHORT ch = static_cast<USHORT>(Pattern[i]);
        m_SkipTable[HashChar(ch)] = static_cast<int>(m_cchPatternLen - 1 - i);
    }

    return true;
}

// -------------------------------------------------------------------------------------
// Executes the wide-character string search algorithm against the provided text buffer.
// Dynamically routes to the optimal execution engine (Scalar, AVX2, or NEON) 
// based on the CPU architecture, buffer length, and configuration.
//
// Parameters:
//   Text       - A pointer to the 16-bit wide character buffer to be scanned.
//                Must be resident in non-paged memory if called at DISPATCH_LEVEL.
//   cchTextLen - The length of the Text buffer in wide characters.
//   Engine     - The execution strategy to use (defaults to Auto).
//                'Auto' intelligently bypasses the SIMD XSTATE context 
//                switch overhead for small buffer sizes.
//
// Returns:
//   The zero-based index of the first wide character of the matched pattern 
//   within the Text buffer, or -1 if the pattern was not found.
// -------------------------------------------------------------------------------------
int CKmStrSearch16::Find(const wchar_t* __restrict Text, 
                         size_t                    cchTextLen, 
                         SearchEngine              Engine) const
{
    PAGED_CODE();

    // 1. Fast Validation
    if (!Text || m_cchPatternLen == 0 || cchTextLen < m_cchPatternLen) [[unlikely]]
    {
        return -1;
    }

    // 2. Most common scenario (Auto) first to improve branch prediction
    if (Engine == SearchEngine::Auto) [[likely]]
    {
        return FindKernel(Text, cchTextLen);
    }

    // 3. Explicit engine requests
    if (Engine == SearchEngine::Scalar) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    // 4. Run requested SIMD engine with state safety
    return RunEngineWithSafety(Text, cchTextLen, Engine);
}

// -------------------------------------------------------------------------------------
// Bypasses SIMD execution for buffers falling under the configured memory threshold.
// -------------------------------------------------------------------------------------
int CKmStrSearch16::FindKernel(const wchar_t* __restrict Text, 
                               size_t                    cchTextLen) const
{
    PAGED_CODE();

    // 1. Small Buffer Processing (Bypass the XSTATE AVX saving overhead)
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

    // 3. Absolute Fallback (Legacy Hardware without AVX2/NEON)
    return FindScalar(Text, cchTextLen);

#elif defined(_M_ARM64)
    return RunEngineWithSafety(Text, cchTextLen, SearchEngine::NEON);
#endif
}

// -------------------------------------------------------------------------------------
// Internal private helper to handle the state-saving "overhead"
// Ensures KeSaveExtendedProcessorState isolates Ring 0 threads from corrupting User Mode vectors.
// -------------------------------------------------------------------------------------
int CKmStrSearch16::RunEngineWithSafety(const wchar_t* __restrict Text, 
                                        size_t                    cchTextLen, 
                                        SearchEngine              Engine) const
{
    PAGED_CODE();

#if defined(_M_X64) || defined(_M_IX86)
    // Validate support for requested engine
    if (Engine == SearchEngine::AVX2 && !m_bHasAVX2) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    XSTATE_SAVE SaveState;    
    
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
        
        if (NT_SUCCESS(KeSaveFloatingPointState(&SaveState))) [[likely]]
        {
            int iResult = FindNEON(Text, cchTextLen);
            KeRestoreFloatingPointState(&SaveState);
            
            return iResult;
        }
    }
#endif

    return FindScalar(Text, cchTextLen);
}

// -------------------------------------------------------------------------------------
// Executes an integer-based Boyer-Moore-Horspool algorithm with SWAR fallback 
// integration.
// -------------------------------------------------------------------------------------
int CKmStrSearch16::FindScalar(const wchar_t* __restrict Text, 
                               size_t                    cchTextLen) const
{
    PAGED_CODE();

    if (m_cchPatternLen == 1) [[unlikely]]
    {
        const wchar_t* pMatch = WMemChrScalar(Text, m_Pattern[0], cchTextLen);
        return pMatch ? static_cast<int>(pMatch - Text) : -1;
    }

    const size_t  uLastIdx = m_cchPatternLen - 1;
    const wchar_t lastChar = m_Pattern[uLastIdx];
    size_t        uK       = uLastIdx;

    while (uK < cchTextLen)
    {
        USHORT ch = static_cast<USHORT>(Text[uK]);

        if (Text[uK] != lastChar) [[likely]]
        {
            int skip = m_SkipTable[HashChar(ch)];

            // OPTIMIZATION: If the Horsepool skip is poor (<= 2), we are likely thrashing in 
            // a dense area of partial matches. Break out and fast-forward using SWAR.
            if (skip <= 2) [[unlikely]]
            {
                const wchar_t* pMatch = WMemChrScalar(Text + uK + 1, lastChar, cchTextLen - uK - 1);
                
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

        size_t uJ = uLastIdx - 1;
        size_t uI = uK - 1;

        // Verify the rest of the pattern backwards
        while (Text[uI] == m_Pattern[uJ])
        {
            if (uJ == 0) [[unlikely]]
            {
                return static_cast<int>(uI);
            }

            --uI;
            --uJ;
        }

        uK += m_SkipTable[HashChar(ch)];
    }

    return -1;
}

#if defined(_M_X64) || defined(_M_IX86)

// -------------------------------------------------------------------------------------
// Processes 16-bit wide strings via heavily unrolled AVX2 memory mapping.
// -------------------------------------------------------------------------------------
int CKmStrSearch16::FindAVX2(const wchar_t* __restrict Text, 
                             size_t                    cchTextLen) const
{
    PAGED_CODE();

    if (cchTextLen < 64 || m_cchPatternLen < 2) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    const size_t uLastIdx = m_cchPatternLen - 1;
    
    // Broadcast the LAST character instead of the FIRST to prevent 'aaaaab' pipeline stalls.
    // Wide char requires the epi16 AVX2 broadcast.
    __m256i vLast = _mm256_set1_epi16(m_Pattern[uLastIdx]);
    size_t  uI    = 0;

    auto CheckMask = [&](__m256i cmp, 
                         size_t  offset_base) -> int 
    {
        // _mm256_movemask_epi8 extracts 32 bits from 32-byte chunks.
        // 0x55555555 explicitly strips the upper byte overlap since wchar_t is strictly 2 bytes.
        unsigned int m = _mm256_movemask_epi8(cmp) & 0x55555555;
        
        while (m != 0) 
        {
            ULONG bit; 
            _BitScanForward(&bit, m);
            
            // The offset points to where the LAST character matched. Subtract uLastIdx.
            // Divide by 2 because hardware bits reflect bytes rather than 16-bit indices.
            size_t matchEndOffset = offset_base + bit / 2;
            size_t startOffset    = matchEndOffset - uLastIdx;
            
            // Fast scalar check of the FIRST character before invoking the middle-match function
            if (Text[startOffset] == m_Pattern[0] && 
                VerifyMiddleMatch(Text + startOffset, m_Pattern, m_cchPatternLen)) [[unlikely]]
            {
                return static_cast<int>(startOffset);
            }
            
            m &= (m - 1);
        }
        
        return -1;
    };

    // BIFURCATION: Long patterns benefit from BMH skipping. Short patterns are stalled 
    // by memory data dependencies, so they run a pure SIMD linear slide.
    // Raised from 32 to 128 wchar_t so that the BMH hash jump overcomes the latency 
    // of the multiplicative hash calculation and table lookup compared to pure vector streaming.
    if (m_cchPatternLen >= 128)
    {
        // 2x Unroll (32 wchar_ts) for safer chunking before a jump
        while (uI + 31 + uLastIdx < cchTextLen)
        {
            // Shift the YMM load forward by uLastIdx
            __m256i b0    = _mm256_loadu_si256((const __m256i*)&Text[uI + uLastIdx]);
            __m256i b1    = _mm256_loadu_si256((const __m256i*)&Text[uI + 16 + uLastIdx]);

            __m256i c0    = _mm256_cmpeq_epi16(b0, vLast);
            __m256i c1    = _mm256_cmpeq_epi16(b1, vLast);
            __m256i cBoth = _mm256_or_si256(c0, c1);

            if (_mm256_testz_si256(cBoth, cBoth) == 0) [[unlikely]]
            {
                int res;
                
                if ((res = CheckMask(c0, uI + uLastIdx)) != -1) [[unlikely]]
                {
                    return res;
                }
                
                if ((res = CheckMask(c1, uI + 16 + uLastIdx)) != -1) [[unlikely]]
                {
                    return res;
                }
            }

            // Retrieve the last scanned char, pass it through the custom golden-ratio hash, and skip.
            USHORT badChar = static_cast<USHORT>(Text[uI + 31 + uLastIdx]);
            uI += 31 + m_SkipTable[HashChar(badChar)];
        }
    }
    else
    {
        // 4x Unrolled (64 wchar_ts / 128 bytes)
        while (uI + 63 + uLastIdx < cchTextLen)
        {
            // Shift the YMM load forward by uLastIdx
            __m256i b0 = _mm256_loadu_si256((const __m256i*)&Text[uI + uLastIdx]);
            __m256i b1 = _mm256_loadu_si256((const __m256i*)&Text[uI + 16 + uLastIdx]);
            __m256i b2 = _mm256_loadu_si256((const __m256i*)&Text[uI + 32 + uLastIdx]);
            __m256i b3 = _mm256_loadu_si256((const __m256i*)&Text[uI + 48 + uLastIdx]);

            __m256i c0 = _mm256_cmpeq_epi16(b0, vLast);
            __m256i c1 = _mm256_cmpeq_epi16(b1, vLast);
            __m256i c2 = _mm256_cmpeq_epi16(b2, vLast);
            __m256i c3 = _mm256_cmpeq_epi16(b3, vLast);

            __m256i or01  = _mm256_or_si256(c0, c1);
            __m256i or23  = _mm256_or_si256(c2, c3);
            __m256i orAll = _mm256_or_si256(or01, or23);

            if (_mm256_testz_si256(orAll, orAll) == 0) [[unlikely]]
            {
                int res;
                
                if ((res = CheckMask(c0, uI + uLastIdx)) != -1) [[unlikely]]
                {
                    return res;
                }
                
                if ((res = CheckMask(c1, uI + 16 + uLastIdx)) != -1) [[unlikely]]
                {
                    return res;
                }
                
                if ((res = CheckMask(c2, uI + 32 + uLastIdx)) != -1) [[unlikely]]
                {
                    return res;
                }
                
                if ((res = CheckMask(c3, uI + 48 + uLastIdx)) != -1) [[unlikely]]
                {
                    return res;
                }
            }
            
            uI += 64; 
        }
    }

    // Tail processing for trailing bytes natively failing within 64 wchar boundaries.
    if (uI < cchTextLen) [[unlikely]]
    {
        int tailRes = FindScalar(Text + uI, cchTextLen - uI);
        
        if (tailRes != -1) [[unlikely]]
        {
            return static_cast<int>(uI + tailRes);
        }
    }

    return -1;
}

#elif defined(_M_ARM64)

// -------------------------------------------------------------------------------------
// 128-bit vector NEON implementation scaling up the wchar_t requirements natively 
// mapped to ARM hardware.
// -------------------------------------------------------------------------------------
int CKmStrSearch16::FindNEON(const wchar_t* __restrict Text, 
                             size_t                    cchTextLen) const
{
    PAGED_CODE();

    // Requires minimum 64 wchar_t due to the 4-register unroll strategy (32 wchar_t / 64 bytes per block)
    if (cchTextLen < 64 || m_cchPatternLen < 2) [[unlikely]]
    {
        return FindScalar(Text, cchTextLen);
    }

    const size_t uLastIdx = m_cchPatternLen - 1;
    
    // Broadcast the LAST character instead of the FIRST to prevent 'aaaaab' pipeline stalls
    uint16x8_t vLast = vdupq_n_u16(static_cast<USHORT>(m_Pattern[uLastIdx]));

    // NEON equivalent for _mm256_movemask_epi8. 
    // Emulates a bitmask where bit 'i' is set if lane 'i' is true.
    auto neon_movemask_u16 = [](uint16x8_t cmp) -> unsigned int 
    {
        static const USHORT shift_arr[8] = { 1, 2, 4, 8, 16, 32, 64, 128 };
        uint16x8_t shift = vld1q_u16(shift_arr);
        
        // cmp contains 0xFFFF for matches, 0x0000 for mismatches.
        // Bitwise AND leaves only the shift array value (e.g., 4) on matching lanes.
        uint16x8_t masked = vandq_u16(cmp, shift);
        
        // Sums all lanes across the vector. Since shift values are powers of 2, 
        // their arithmetic sum perfectly emulates a bitwise OR mask assembly.
        return vaddvq_u16(masked);
    };

    auto CheckLane = [&](uint16x8_t cmp, 
                         size_t     uLaneOffset) -> int 
    {
        // vmaxvq_u32 treats 4x 32-bit elements, returning the maximum. 
        // If cmp is completely empty (all 0), the max is 0 (fast rejection).
        if (vmaxvq_u32(vreinterpretq_u32_u16(cmp)) != 0) [[unlikely]]
        {
            unsigned int final_mask = neon_movemask_u16(cmp);

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
        // 4x Vector Unroll (32 wchar_ts / 64 bytes total for NEON)
        uint16x8_t v0 = vld1q_u16(reinterpret_cast<const USHORT*>(&Text[uCurrentI + uLastIdx]));
        uint16x8_t v1 = vld1q_u16(reinterpret_cast<const USHORT*>(&Text[uCurrentI + 8 + uLastIdx]));
        uint16x8_t v2 = vld1q_u16(reinterpret_cast<const USHORT*>(&Text[uCurrentI + 16 + uLastIdx]));
        uint16x8_t v3 = vld1q_u16(reinterpret_cast<const USHORT*>(&Text[uCurrentI + 24 + uLastIdx]));

        uint16x8_t cmp0 = vceqq_u16(v0, vLast);
        uint16x8_t cmp1 = vceqq_u16(v1, vLast);
        uint16x8_t cmp2 = vceqq_u16(v2, vLast);
        uint16x8_t cmp3 = vceqq_u16(v3, vLast);
        
        uint16x8_t cmp01  = vorrq_u16(cmp0, cmp1);
        uint16x8_t cmp23  = vorrq_u16(cmp2, cmp3);
        uint16x8_t cmpAll = vorrq_u16(cmp01, cmp23);

        // Fast path rejection
        if (vmaxvq_u32(vreinterpretq_u32_u16(cmpAll)) != 0) [[unlikely]]
        {
            int iResult;
            
            if ((iResult = CheckLane(cmp0, uCurrentI + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }

            if ((iResult = CheckLane(cmp1, uCurrentI + 8 + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }

            if ((iResult = CheckLane(cmp2, uCurrentI + 16 + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }

            if ((iResult = CheckLane(cmp3, uCurrentI + 24 + uLastIdx)) != -1) [[unlikely]]
            {
                return iResult;
            }
        }
        
        return -1;
    };

    size_t uI = 0;

    // BIFURCATION: Raised to 128 wchar_t to bypass the multiplicative hash & L1 table lookup
    // latency for short-to-medium patterns, maximizing sequential streaming throughput.
    if (m_cchPatternLen >= 128)
    {
        while (uI + 31 + uLastIdx < cchTextLen)
        {
            int iMatch = ProcessBlock(uI);
            
            if (iMatch != -1) [[unlikely]]
            {
                return iMatch;
            }

            USHORT badChar = static_cast<USHORT>(Text[uI + 31 + uLastIdx]);
            uI += 31 + m_SkipTable[HashChar(badChar)];
        }
    }
    else
    {
        while (uI + 31 + uLastIdx < cchTextLen)
        {
            int iMatch = ProcessBlock(uI);
            
            if (iMatch != -1) [[unlikely]]
            {
                return iMatch;
            }

            uI += 32; 
        }
    }

    if (uI < cchTextLen) [[unlikely]]
    {
        int iTailResult = FindScalar(Text + uI, cchTextLen - uI);
        
        if (iTailResult != -1) [[unlikely]]
        {
            return static_cast<int>(uI + iTailResult);
        }
    }

    return -1;
}
#endif