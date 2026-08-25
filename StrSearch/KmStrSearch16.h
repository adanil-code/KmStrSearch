// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// KmStrSearch16.h
// -------------------------------------------------------------------------------------
#pragma once

class CKmStrSearch16
{
public:
    // Allows the client to explicitly select the algorithm or use the smart threshold
    enum class SearchEngine
    {
        Auto = 0, // Smart wrapper (bypasses SIMD context switch for small buffers)
        Scalar,   // Forced Scalar (Hybrid wmemchr fallback)
        AVX2,     // Forced AVX2 (Will fallback to Scalar if CPU/Arch unsupported)    
        NEON      // Forced ARM64 NEON (Will fallback to Scalar if CPU/Arch unsupported)
    };

public:
    CKmStrSearch16();
    ~CKmStrSearch16();

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
    bool Initialize(const wchar_t* __restrict Pattern, 
                    size_t                    cchLength);

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
    int Find(const wchar_t* __restrict Text, 
             size_t                    cchTextLen, 
             SearchEngine              Engine = SearchEngine::Auto) const;

private:
    // Internal implementations
    int FindKernel(const wchar_t* __restrict Text, 
                   size_t                    cchTextLen) const;

    int FindScalar(const wchar_t* __restrict Text, 
                   size_t                    cchTextLen) const;

    int RunEngineWithSafety(const wchar_t* __restrict Text, 
                            size_t                    cchTextLen, 
                            SearchEngine              Engine) const;

#if defined(_M_X64) || defined(_M_IX86)
    int FindAVX2(const wchar_t* __restrict Text, 
                 size_t                    cchTextLen) const;    
#elif defined(_M_ARM64)
    int FindNEON(const wchar_t* __restrict Text, 
                 size_t                    cchTextLen) const;
#endif

    static inline const wchar_t* WMemChrScalar(const wchar_t* __restrict ptr,
                                               wchar_t                   val,
                                               size_t                    cchNum);

    static inline bool VerifyMiddleMatch(const wchar_t* __restrict Text,
                                         const wchar_t* __restrict Pattern,
                                         size_t                    cchPatternLen);

private:
    // 2K Hash Table Size to fit entirely in the L1 Data Cache.
    static const ULONG HASH_TABLE_SIZE = 2048; 
    static const ULONG POOL_TAG        = 'psrH';

    // Fast hash function to map 16-bit Unicode characters into a 11-bit (2048) bucket.
    static inline ULONG HashChar(USHORT ch)
    {
        // Multiply by the 32-bit Golden Ratio, then shift down to leave exactly 11 bits.
        // 32 - 11 = 21. No bitwise AND mask is required.
        return (ch * 2654435769U) >> 21;        
    }

    // State-Save Thresholds calculated in bytes
#if defined(_M_ARM64)
    #define KERNEL_THRESHOLD_BYTES 1024 
#else
    #define KERNEL_THRESHOLD_BYTES 6144
#endif

    static const ULONG KERNEL_THRESHOLD_CHARS = KERNEL_THRESHOLD_BYTES / sizeof(wchar_t);

    int*           m_SkipTable;
    size_t         m_cchPatternLen;
    const wchar_t* m_Pattern;

#if defined(_M_X64) || defined(_M_IX86)    
    bool m_bHasAVX2;
#endif
};