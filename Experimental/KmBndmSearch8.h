// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// KmBndmSearch8.h
// -------------------------------------------------------------------------------------
#pragma once

class CKmBndmSearch8
{
public:
    // Allows the client to explicitly select the algorithm or use the smart threshold
    enum class SearchEngine
    {
        Auto = 0, // Smart wrapper (bypasses SIMD context switch for small buffers)
        Scalar,   // Forced Scalar (64-bit BNDM or GPR slide fallback)
        AVX2,     // Forced AVX2 (Will fallback to Scalar if CPU/Arch unsupported)
        NEON      // Forced ARM64 NEON (Will fallback to Scalar if CPU/Arch unsupported)
    };

public:
    CKmBndmSearch8();
    ~CKmBndmSearch8();

    // -------------------------------------------------------------------------------------
    // Initializes the wide character BNDM engine with a Golden Ratio hashed mask table.
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
    bool Initialize(const char* __restrict Pattern, 
                    size_t                 cchLength);

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
    int Find(const char* __restrict Text, 
             size_t                 cchTextLen, 
             SearchEngine           Engine = SearchEngine::Auto) const;

private:
    // Internal implementations
    int FindKernel(const char* __restrict Text, 
                   size_t                 cchTextLen) const;

    int FindScalar(const char* __restrict Text, 
                   size_t                 cchTextLen) const;

    int RunEngineWithSafety(const char* __restrict Text, 
                            size_t                 cchTextLen, 
                            SearchEngine           Engine) const;

#if defined(_M_X64) || defined(_M_IX86)
    int FindAVX2(const char* __restrict Text, 
                 size_t                 cchTextLen) const;
#elif defined(_M_ARM64)
    int FindNEON(const char* __restrict Text, 
                 size_t                 cchTextLen) const;
#endif

    static inline const char* MemChrScalar(const char* __restrict ptr,
                                           char                   val,
                                           size_t                 cchNum);

    static inline bool VerifyMiddleMatch(const char* __restrict Text,
                                         const char* __restrict Pattern,
                                         size_t                 cchPatternLen);

private:
    // With 8-bit chars (ASCII/UTF-8), there are only 256 possible byte values.
    static const ULONG  TABLE_SIZE           = 256;
    static const ULONG  POOL_TAG             = 'bdnH';
    static const size_t MAX_BNDM_PATTERN_LEN = 64;

    // State-Save Thresholds calculated in bytes
#if defined(_M_ARM64)
    #define KERNEL_THRESHOLD_BYTES 1024 
#else
    #define KERNEL_THRESHOLD_BYTES 6144
#endif

    static const ULONG KERNEL_THRESHOLD_CHARS = KERNEL_THRESHOLD_BYTES / sizeof(char);

    ULONGLONG* m_MaskTable;
    size_t              m_cchPatternLen;
    const char*         m_Pattern;

#if defined(_M_X64) || defined(_M_IX86)    
    bool m_bHasAVX2;
#endif
};