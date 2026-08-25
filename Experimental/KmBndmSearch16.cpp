// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// KmBndmSearch16.cpp
//
// ALGORITHM OVERVIEW:
// Implements a Hybrid search utilizing Backward Nondeterministic DAWG Matching (BNDM) 
// for the 64-bit scalar fallback and Pure SIMD memory sliding for large buffers.
//
// THE WIDE-CHARACTER BNDM BIT-PARALLELISM DESIGN:
// 1. Bitmask Bloom Merging: Instead of maintaining an unfeasible 65,536-entry array
//    which causes L1 cache thrashing, we map 16-bit characters into a 2,048 entry hash 
//    table (16KB). When hash collisions occur, bitmasks are logically OR'd.
// 2. Pure SIMD Streaming: For buffers > KERNEL_THRESHOLD_CHARS, BNDM is bypassed 
//    entirely in favor of unrolled AVX2/NEON vector streaming.
// 3. Hybrid Bifurcation for Long Patterns: Patterns > 64 chars bypass BNDM allocation
//    and route to vector streams or a fast SWAR scalar slide fallback automatically.
//
// KERNEL ARCHITECTURE CONSIDERATIONS:
// 1. Context Switch Overhead: Using floating-point/SIMD registers in Ring 0 requires 
//    explicitly saving the thread's processor state via KPCR. For small buffers, this 
//    overhead vastly exceeds the gains of vectorization. Thus, an auto-bypass evaluates 
//    buffer bounds.
// 2. IRQL Constraints: ExAllocatePool2 with POOL_FLAG_PAGED and PAGED_CODE() macros 
//    enforce that this class must be used at PASSIVE_LEVEL or APC_LEVEL.
// 3. Pipeline Stalls: To prevent CPU branch prediction stalls and severe O(N*M) 
//    degradation on repetitive inputs, the SWAR fallback explicitly scans for the LAST 
//    character of the pattern before engaging the backward automaton.
// -------------------------------------------------------------------------------------
#include <ntddk.h>
#include <intrin.h>
#include <limits.h>
#include "KmBndmSearch16.h"

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
// SWAR scalar fallback for single wide characters.
// Tuned SWAR scalar fallback that stays strictly within general purpose registers
// (GPRs). Accelerates wide-character searching without incurring XMM/YMM context switch
// costs.
// -------------------------------------------------------------------------------------
inline const wchar_t* CKmBndmSearch16::WMemChrScalar(const wchar_t* __restrict ptr,
                                                     wchar_t                   val, 
                                                     size_t                    cchNum)
{
    PAGED_CODE();

    const wchar_t* pEnd = ptr + cchNum;

#if defined(_M_X64) || defined(_M_ARM64)
    // 64-bit optimization: Broadcast the 16-bit target wchar_t across an 8-byte integer.
    // This allows us to evaluate 4 wide-characters per CPU cycle without utilizing SSE/AVX.
    ULONGLONG c4 = static_cast<USHORT>(val);
    c4 |= c4 << 16;
    c4 |= c4 << 32;

    // Magic numbers for SWAR zero-byte detection (Mycroft's bit-twiddling hack)
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
// Exact candidate verification loops
// -------------------------------------------------------------------------------------
inline bool CKmBndmSearch16::VerifyMiddleMatch(const wchar_t* __restrict Text,
                                               const wchar_t* __restrict Pattern, 
                                               size_t                    cchPatternLen)
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
// Exact candidate verification loop to eliminate hash collision false positives.
// -------------------------------------------------------------------------------------
inline bool CKmBndmSearch16::VerifyFullMatch(const wchar_t* __restrict Text,
                                             const wchar_t* __restrict Pattern, 
                                             size_t                    cchPatternLen)
{
    PAGED_CODE();

    for (size_t uI = 0; uI < cchPatternLen; ++uI)
    {
        if (Text[uI] != Pattern[uI]) [[likely]]
        {
            return false;
        }
    }

    return true;
}

// -------------------------------------------------------------------------------------
// Constructor & Destructor
// -------------------------------------------------------------------------------------
CKmBndmSearch16::CKmBndmSearch16() : m_MaskTable(nullptr), 
                                     m_cchPatternLen(0), 
                                     m_Pattern(nullptr)
{
    PAGED_CODE();

#if defined(_M_X64) || defined(_M_IX86)    
    m_bHasAVX2 = false;
#endif
}

CKmBndmSearch16::~CKmBndmSearch16()
{
    if (m_MaskTable)
    {
        ExFreePoolWithTag(m_MaskTable, POOL_TAG);
    }
}

// -------------------------------------------------------------------------------------
// Initializes the wide character BNDM engine with a Golden Ratio hashed mask table.
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
bool CKmBndmSearch16::Initialize(const wchar_t* __restrict Pattern, 
                                 size_t                    cchLength)
{
    PAGED_CODE();

    if (!Pattern || cchLength == 0) [[unlikely]]
    {
        return false;
    }

    // EDGE CASE: Prevent INT_MAX overflow during kernel memory arithmetic.
    if (cchLength > INT_MAX) [[unlikely]]
    {
        return false;
    }

#if defined(_M_X64) || defined(_M_IX86)    
    m_bHasAVX2 = ExIsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE);
#endif

    m_Pattern       = Pattern;
    m_cchPatternLen = cchLength;

    // Option 2: Hybrid Bifurcation. If pattern > 64, bypass BNDM allocation
    // and rely purely on vector streams or fallback scalar slide.
    if (m_cchPatternLen > MAX_BNDM_PATTERN_LEN)
    {
        m_MaskTable = nullptr;
        return true;
    }

    // Allocate 16KB flat array for 64-bit wide bitmasks. Fits natively in L1 D-cache.
    // Uses POOL_FLAG_CACHE_ALIGNED to prevent false sharing and ensure optimal L1 D-Cache loads.
    m_MaskTable = static_cast<ULONGLONG*>(ExAllocatePool2(POOL_FLAG_PAGED | POOL_FLAG_UNINITIALIZED | POOL_FLAG_CACHE_ALIGNED, 
                                                          HASH_TABLE_SIZE * sizeof(ULONGLONG), 
                                                          POOL_TAG));    
    if (!m_MaskTable) [[unlikely]]
    {
        return false;
    }

    for (ULONG ulI = 0; ulI < HASH_TABLE_SIZE; ++ulI)
    {
        m_MaskTable[ulI] = 0;
    }

    // Populate hashed bitmasks: logical OR safely merges masks in case of bucket collisions.
    for (size_t uJ = 0; uJ < m_cchPatternLen; ++uJ)
    {
        USHORT ch = static_cast<USHORT>(Pattern[m_cchPatternLen - 1 - uJ]);
        m_MaskTable[HashChar(ch)] |= (1ULL << uJ);
    }

    return true;
}

// -------------------------------------------------------------------------------------
// Executes the string search algorithm against the provided text buffer.
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
//   The zero-based index of the first character of the matched pattern 
//   within the Text buffer, or -1 if the pattern was not found.
// -------------------------------------------------------------------------------------
int CKmBndmSearch16::Find(const wchar_t* __restrict Text, 
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
int CKmBndmSearch16::FindKernel(const wchar_t* __restrict Text, 
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
int CKmBndmSearch16::RunEngineWithSafety(const wchar_t* __restrict Text, 
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
// Executes wide-character BNDM search or Hybrid GPR Fallback
// -------------------------------------------------------------------------------------
int CKmBndmSearch16::FindScalar(const wchar_t* __restrict Text, 
                                size_t                    cchTextLen) const
{
    PAGED_CODE();

    if (cchTextLen < m_cchPatternLen) [[unlikely]]
    {
        return -1;
    }

    if (m_cchPatternLen == 1) [[unlikely]]
    {
        const wchar_t* pMatch = WMemChrScalar(Text, m_Pattern[0], cchTextLen);
        
        if (pMatch) [[unlikely]]
        {
            return static_cast<int>(pMatch - Text);
        }
        else
        {
            return -1;
        }
    }

    // Hybrid Bifurcation Fallback for Long Patterns (> 64)
    // Avoids complex multi-word BNDM logic by defaulting to a fast GPR slide.
    if (m_cchPatternLen > MAX_BNDM_PATTERN_LEN || !m_MaskTable)
    {
        const size_t  uLastIdx = m_cchPatternLen - 1;
        const wchar_t lastChar = m_Pattern[uLastIdx];
        size_t        uI       = 0;

        while (uI + uLastIdx < cchTextLen)
        {
            const wchar_t* pMatch = WMemChrScalar(Text + uI + uLastIdx, lastChar, cchTextLen - (uI + uLastIdx));
            
            if (!pMatch) [[unlikely]]
            {
                return -1;
            }

            size_t matchEndOffset = pMatch - Text;
            size_t startOffset    = matchEndOffset - uLastIdx;

            // Fast scalar check of the FIRST character before invoking the middle-match function
            if (Text[startOffset] == m_Pattern[0] &&
                VerifyMiddleMatch(Text + startOffset, m_Pattern, m_cchPatternLen)) [[unlikely]]
            {
                return static_cast<int>(startOffset);
            }

            uI = startOffset + 1;
        }

        return -1;
    }

    // Standard 64-bit BNDM Execution
    // The automaton state is maintained in a single 64-bit GPR (uD).
    const size_t    m          = m_cchPatternLen;
    const ULONGLONG uPrefixBit = (1ULL << (m - 1));
    const wchar_t   lastChar   = m_Pattern[m - 1]; 
    size_t          uPos       = 0;

    while (uPos <= cchTextLen - m)
    {
        // FAST FORWARD OPTIMIZATION:
        // Rapidly skip non-matching regions by utilizing SWAR GPR scanning
        // to find the last character of the pattern before engaging the BNDM automaton.        
        if (Text[uPos + m - 1] != lastChar) [[likely]]
        {
            const wchar_t* pMatch = WMemChrScalar(Text + uPos + m - 1, lastChar, cchTextLen - (uPos + m - 1));
            
            if (!pMatch) [[unlikely]]
            {
                return -1;
            }

            uPos = static_cast<size_t>(pMatch - Text) - (m - 1);
        }

        size_t    uJ    = m;
        size_t    uLast = m;
        ULONGLONG uD    = ~0ULL; // Initialize automaton to match any prefix

        // Evaluate character state and bit-shift
        while (uD != 0)
        {
            --uJ;
            USHORT ch = static_cast<USHORT>(Text[uPos + uJ]);
            
            // Advance the automaton state using the hashed bitmask for this character
            uD &= m_MaskTable[HashChar(ch)];

            // Check if the current prefix matches the beginning of the pattern
            if ((uD & uPrefixBit) != 0)
            {
                if (uJ == 0) [[unlikely]]
                {
                    // Validate against potential hash collisions
                    if (VerifyFullMatch(Text + uPos, m_Pattern, m)) [[likely]]
                    {
                        return static_cast<int>(uPos);
                    }

                    // Collision occurred; break backward scan and step forward
                    break;
                }

                uLast = uJ;
            }

            uD <<= 1;
        }

        uPos += uLast;
    }

    return -1;
}

#if defined(_M_X64) || defined(_M_IX86)
// -------------------------------------------------------------------------------------
// Processes 16-bit wide strings via heavily unrolled AVX2 memory mapping.
// -------------------------------------------------------------------------------------
int CKmBndmSearch16::FindAVX2(const wchar_t* __restrict Text, 
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

    // 4x Unrolled (64 wchar_ts / 128 bytes) Pure Vector Slide
    // Pure memory slide without algorithmic skip checking to keep instruction pipeline completely saturated.
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
int CKmBndmSearch16::FindNEON(const wchar_t* __restrict Text, 
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

    // 4x Unrolled Pure Vector Slide
    while (uI + 31 + uLastIdx < cchTextLen)
    {
        int iMatch = ProcessBlock(uI);
        
        if (iMatch != -1) [[unlikely]]
        {
            return iMatch;
        }

        uI += 32; 
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