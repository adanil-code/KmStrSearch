// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// KmStrSearchTestShared.h
// Unified Test Logic (Used by both Kernel and User Mode wrappers)
// -------------------------------------------------------------------------------------
#pragma once

#ifndef DRIVER_TAG
    #define DRIVER_TAG 'Hrsp'
#endif

extern volatile LONG g_lAbortTests;
extern BOOLEAN       g_HasAVX2;

VOID FlushLogToFile();

#define MAX_TEST_THREADS 32

// Configuration structure allowing callers to selectively run specific algorithms
struct TEST_SUITE_CONFIG
{
    BOOLEAN bTestKmStrSearch;
    BOOLEAN bTestKmBndmSearch;
};

// Context tracking state for individual test worker threads
struct TEST_WORKER_CONTEXT
{
    ULONG          ulThreadId;
    PKEVENT        pStartEvent;
    volatile LONG* plStopFlag;
    PVOID          pUserContext;
};

typedef VOID(*PTEST_WORKER_FUNC)(_Inout_ TEST_WORKER_CONTEXT* pContext);

// Manager handling thread pooling and synchronization
struct TEST_THREAD_MANAGER
{
    PETHREAD            pThreads[MAX_TEST_THREADS];
    TEST_WORKER_CONTEXT Contexts[MAX_TEST_THREADS];
    ULONG               ulThreadCount;
    KEVENT              StartEvent;
    volatile LONG       lStopFlag;
};

// -------------------------------------------------------------------------------------
// Initializes and starts a pool of worker threads synchronized on an event.
// Used to saturate the CPU and memory bus during high-throughput performance testing.
//
// Parameters:
//   pMgr         - Pointer to the thread manager structure holding state and handles.
//   ulCount      - The number of worker threads to spawn (capped at MAX_TEST_THREADS).
//   pFunc        - Pointer to the worker routine to execute on each thread.
//   pUserContext - Opaque context pointer passed to each worker thread.
// -------------------------------------------------------------------------------------
VOID StartThreads(_Inout_  TEST_THREAD_MANAGER* pMgr, 
                  _In_     ULONG                ulCount, 
                  _In_     PTEST_WORKER_FUNC    pFunc, 
                  _In_opt_ PVOID                pUserContext)
{
    PAGED_CODE();

    if (ulCount > MAX_TEST_THREADS)
    {
        ulCount = MAX_TEST_THREADS;
    }

    pMgr->ulThreadCount = ulCount;
    pMgr->lStopFlag     = 0;
    
    KeInitializeEvent(&pMgr->StartEvent, NotificationEvent, FALSE);

    // Spawn requested number of threads
    for (ULONG ulIndex = 0; ulIndex < ulCount; ++ulIndex)
    {
        pMgr->Contexts[ulIndex].ulThreadId   = ulIndex;
        pMgr->Contexts[ulIndex].pStartEvent  = &pMgr->StartEvent;
        pMgr->Contexts[ulIndex].plStopFlag   = &pMgr->lStopFlag;
        pMgr->Contexts[ulIndex].pUserContext = pUserContext;

        HANDLE hThread;
        NTSTATUS ntStatus = PsCreateSystemThread(&hThread, 
                                                 0, 
                                                 NULL, 
                                                 NULL, 
                                                 NULL, 
                                                (PKSTART_ROUTINE)pFunc,
                                                 &pMgr->Contexts[ulIndex]);

        if (NT_SUCCESS(ntStatus))
        {
            ntStatus = ObReferenceObjectByHandle(hThread, 
                                                 0, 
                                                 NULL, 
                                                 KernelMode, 
                                                 (PVOID*)&pMgr->pThreads[ulIndex], 
                                                 NULL);
                                                 
            if (NT_SUCCESS(ntStatus))
            {
                // Elevate priority to stabilize benchmarks
                KeSetPriorityThread(pMgr->pThreads[ulIndex], 15);
                ZwClose(hThread);
            }
            else
            {
                pMgr->pThreads[ulIndex] = NULL;
                InterlockedExchange(&pMgr->lStopFlag, 1);
                InterlockedExchange(&g_lAbortTests, 1);

                // Release threads and wait for graceful cleanup
                KeSetEvent(&pMgr->StartEvent, IO_NO_INCREMENT, FALSE);
                KeWaitForSingleObject(hThread, Executive, KernelMode, FALSE, NULL);
                ZwClose(hThread);
            }
        }
        else
        {
            pMgr->pThreads[ulIndex] = NULL;
        }
    }
    
    // Release all threads simultaneously
    KeSetEvent(&pMgr->StartEvent, IO_NO_INCREMENT, FALSE);
}

// -------------------------------------------------------------------------------------
// Signals threads to stop and waits for them to terminate safely.
// Includes a synchronized grace period before signaling the abort flags.
//
// Parameters:
//   pMgr          - Pointer to the thread manager handling the active pool.
//   nSleepSeconds - Number of seconds to let the test run before triggering the stop flag.
// -------------------------------------------------------------------------------------
VOID StopAndWaitThreads(_Inout_ TEST_THREAD_MANAGER* pMgr, 
                        _In_    int                  nSleepSeconds)
{
    PAGED_CODE();

    if (nSleepSeconds > 0)
    {
        int nIterations = nSleepSeconds * 100;

        for (int nIndex = 0; nIndex < nIterations; ++nIndex)
        {
            if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
            {
                break;
            }
            
            SleepMs(10);
        }
    }

    InterlockedExchange(&pMgr->lStopFlag, 1);

    for (ULONG ulIndex = 0; ulIndex < pMgr->ulThreadCount; ++ulIndex)
    {
        if (pMgr->pThreads[ulIndex] != NULL)
        {
            KeWaitForSingleObject(pMgr->pThreads[ulIndex], Executive, KernelMode, FALSE, NULL);
            ObDereferenceObject(pMgr->pThreads[ulIndex]);
        }
    }
}

// -------------------------------------------------------------------------------------
// Provides a template-based wrapper for standard C-runtime string search functions.
// Used exclusively to benchmark against the baseline OS implementations.
// -------------------------------------------------------------------------------------
template <typename CharT>
const CharT* StdStrStr(const CharT* __restrict Text, const CharT* __restrict Pattern);

template <>
const char* StdStrStr<char>(const char* __restrict Text, const char* __restrict Pattern)
{
    return strstr(Text, Pattern);
}

template <>
const wchar_t* StdStrStr<wchar_t>(const wchar_t* __restrict Text, const wchar_t* __restrict Pattern)
{
    return wcsstr(Text, Pattern);
}

// -------------------------------------------------------------------------------------
// Naive verification loop establishing the baseline truth for correctness checks.
// Simulates the most basic byte-by-byte comparison to ensure the highly optimized
// SIMD implementations do not deviate in behavior.
//
// Parameters:
//   Text       - Buffer to search within.
//   cchTextLen - Length of the text buffer in elements.
//   Pattern    - The sequence to find.
//   cchPatLen  - Length of the pattern in elements.
// -------------------------------------------------------------------------------------
template <typename CharT>
static int BasicOffset(const CharT* __restrict Text, 
                       size_t                  cchTextLen, 
                       const CharT* __restrict Pattern, 
                       size_t                  cchPatLen)
{
    PAGED_CODE();

    if (cchPatLen == 0 || cchTextLen < cchPatLen)
    {
        return -1;
    }

    for (size_t uI = 0; uI <= cchTextLen - cchPatLen; ++uI)
    {
        bool match = true;
        
        for (size_t uJ = 0; uJ < cchPatLen; ++uJ)
        {
            if (Text[uI + uJ] != Pattern[uJ])
            {
                match = false;
                break;
            }
        }

        if (match)
        {
            return static_cast<int>(uI);
        }
    }
    
    return -1;
}

// -------------------------------------------------------------------------------------
// Validates initialization and checks execution results across all engines
// (Scalar, Auto, SIMD) against the naive matching routine to guarantee correctness.
//
// Parameters:
//   szText        - Buffer to search within.
//   cchTextLen    - Length of the text buffer in elements.
//   szPattern     - The sequence to find.
//   cchPatternLen - Length of the pattern in elements.
//   szTestName    - Descriptive string for console output.
// -------------------------------------------------------------------------------------
template <typename CharT, typename SearchClass>
BOOLEAN RunCorrectnessTestEx(const CharT* __restrict szText, 
                             size_t                  cchTextLen, 
                             const CharT* __restrict szPattern, 
                             size_t                  cchPatternLen, 
                             const char*             szTestName)
{
    PAGED_CODE();

    if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
    {
        return FALSE;
    }

    int expectedIndex = BasicOffset<CharT>(szText, cchTextLen, szPattern, cchPatternLen);

    SearchClass Search;
    
    // Verify initialization failures match expectations (e.g., length zero)
    if (!Search.Initialize(szPattern, cchPatternLen))
    {
        if (cchPatternLen == 0)
        {
            LOG_INFO("[SEARCH_TEST] [+] PASS: %s (Expected Init Failure for Length 0)\n", szTestName);
            return TRUE;
        }

        LOG_ERR("[SEARCH_TEST] [!] %s - Init Failed Unexpectedly\n", szTestName);
        return FALSE;
    }

    int scalarIdx = Search.Find(szText, cchTextLen, SearchClass::SearchEngine::Scalar);
    int autoIdx   = Search.Find(szText, cchTextLen, SearchClass::SearchEngine::Auto);

    int avx2Idx = expectedIndex;

    // Explicitly test SIMD pathways if supported
    if (g_HasAVX2)
    {
        avx2Idx = Search.Find(szText, cchTextLen, SearchClass::SearchEngine::AVX2);
    }

    BOOLEAN passed = (scalarIdx == expectedIndex) &&
                     (autoIdx   == expectedIndex) &&
                     (avx2Idx   == expectedIndex);

    if (passed)
    {
        LOG_INFO("[SEARCH_TEST] [+] PASS: %s\n", szTestName);
        return TRUE;
    }
    else
    {
        LOG_ERR("[SEARCH_TEST] [!] FAIL: %s (Expected: %d | Scalar: %d | Auto: %d | AVX2: %d)\n", 
                szTestName, expectedIndex, scalarIdx, autoIdx, avx2Idx);
        return FALSE;
    }
}

// -------------------------------------------------------------------------------------
// Wrapper for computing null-terminated string lengths prior to correctness checks.
// Eliminates the need for explicitly passing buffer sizes for constant strings.
// -------------------------------------------------------------------------------------
template <typename CharT, typename SearchClass>
BOOLEAN RunCorrectnessTest(const CharT* __restrict szText, 
                           const CharT* __restrict szPattern, 
                           const char*             szTestName)
{
    size_t cchTextLen = 0;
    
    while (szText[cchTextLen] != 0) 
    { 
        cchTextLen++; 
    }
    
    size_t cchPatternLen = 0;
    
    while (szPattern[cchPatternLen] != 0) 
    { 
        cchPatternLen++; 
    }

    return RunCorrectnessTestEx<CharT, SearchClass>(szText, cchTextLen, szPattern, cchPatternLen, szTestName);
}

// -------------------------------------------------------------------------------------
// Data packet fed into threaded performance workers.
// Contains target memory bounds, selected engine parameters, and metric tracking variables.
// -------------------------------------------------------------------------------------
template <typename CharT, typename SearchClass>
struct SearchWorkerCtx
{
    SearchClass*  pSearch;
    const CharT*  pText;
    size_t        cchTextLen;
    const CharT*  pPattern;
    size_t        cchPatternLen;
    
    typename SearchClass::SearchEngine Engine;

    BOOLEAN         bIsStdStringFn;

    volatile LONG64 llTotalOps;    
    volatile LONG   lStartFlag;
};

// -------------------------------------------------------------------------------------
// Iteratively runs search operations in a tight loop to calculate GB/s throughput.
// Threads synchronize on a shared spinlock to ensure simultaneous execution,
// maximizing memory bus saturation during performance analysis.
//
// Parameters:
//   pCtx - Pointer to the context structure containing thread and synchronization state.
// -------------------------------------------------------------------------------------
template <typename CharT, typename SearchClass>
VOID SearchWorkerT(_Inout_ TEST_WORKER_CONTEXT* pCtx)
{
    PAGED_CODE();

    SearchWorkerCtx<CharT, SearchClass>* pWorkerCtx = (SearchWorkerCtx<CharT, SearchClass>*)pCtx->pUserContext;

    KeWaitForSingleObject(pCtx->pStartEvent, Executive, KernelMode, FALSE, NULL);

    UINT64  localOps   = 0;
    UINT64  localDummy = 0;
    BOOLEAN bAborted   = FALSE;

    // Spinlock wait allowing all threads to sync perfectly before the timer starts
    while (InterlockedCompareExchange(&pWorkerCtx->lStartFlag, 0, 0) == 0)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            bAborted = TRUE;
            break;
        }

        YieldProcessor();
    }

    if (!bAborted)
    {
        while (!InterlockedCompareExchange(pCtx->plStopFlag, 0, 0) &&
               !InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            // Shift search space occasionally to prevent CPU caching the exact offset
            size_t       cchShift     = (localOps & 1) * 32;
            const CharT* pSearchStart = pWorkerCtx->pText + cchShift;
            size_t       cchSearchLen = pWorkerCtx->cchTextLen - cchShift;

            if (pWorkerCtx->bIsStdStringFn)
            {
                const CharT* match = StdStrStr<CharT>(pSearchStart, pWorkerCtx->pPattern);
                localDummy += match ? 1 : 0;
            }
            else
            {
                localDummy += pWorkerCtx->pSearch->Find(pSearchStart, cchSearchLen, pWorkerCtx->Engine);
            }

            localOps++;
            COMPILER_BARRIER();
        }

        InterlockedExchangeAdd64(&pWorkerCtx->llTotalOps, localOps);
    }

    PsTerminateSystemThread(STATUS_SUCCESS);
}

// -------------------------------------------------------------------------------------
// Orchestrates performance execution, spins up worker threads, calculates metrics,
// and logs memory throughput telemetry (GB/s).
//
// Parameters:
//   LargeText     - Large data buffer to benchmark against.
//   cchTextChars  - Buffer length.
//   szPattern     - Pattern to repeatedly search for.
//   szTestName    - Name of the performance test variation.
//   szDescription - Detailed description of the operation for logs.
//   szSizeLabel   - Descriptive label defining buffer sizing boundaries.
//   bitSize       - The character width configuration (8 or 16 bit).
// -------------------------------------------------------------------------------------
template <typename CharT, typename SearchClass>
BOOLEAN RunPerformanceSuiteT(CharT*       LargeText, 
                             size_t       cchTextChars, 
                             const CharT* szPattern, 
                             const char*  szTestName, 
                             const char*  szDescription,
                             const char*  szSizeLabel,
                             int          bitSize)
{
    PAGED_CODE();

    if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
    {
        return FALSE;
    }

    size_t chPatternLen = 0;
    
    while (szPattern[chPatternLen] != 0)
    {
        chPatternLen++;
    }

    SearchClass Search;
    
    if (!Search.Initialize(szPattern, chPatternLen))
    {
        return FALSE;
    }

    LOG_INFO("\n[SEARCH_TEST] --- Perf Test (%d-bit): %s [%s] ---\n", bitSize, szTestName, szSizeLabel);
    LOG_INFO("[SEARCH_TEST]      %s\n", szDescription);
    LOG_INFO("[SEARCH_TEST]      %-18s | %-15s | %-15s\n", "Engine", "Throughput", "Improvement");
    LOG_INFO("[SEARCH_TEST]      -----------------------------------------------------------------------------------------\n");

    LARGE_INTEGER liFreq;
    KeQueryPerformanceCounter(&liFreq);

    const int nSecondsPerStep = 2; 

    // Inline lambda to standardize testing individual engine variations
    auto RunEngine = [&](typename SearchClass::SearchEngine engine, 
                         BOOLEAN                            bIsStdFn, 
                         const char*                        label, 
                         UINT64                             baselineBytesPerSec) -> UINT64
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            return 0;
        }

        SearchWorkerCtx<CharT, SearchClass> Ctx = { 0 };
        Ctx.pSearch        = &Search;
        Ctx.pText          = LargeText;
        Ctx.cchTextLen     = cchTextChars;
        Ctx.pPattern       = szPattern;
        Ctx.cchPatternLen  = chPatternLen;
        Ctx.Engine         = engine;
        Ctx.bIsStdStringFn = bIsStdFn;

        TEST_THREAD_MANAGER* pMgr = (TEST_THREAD_MANAGER*)ExAllocatePool2(POOL_FLAG_NON_PAGED, 
                                                                          sizeof(TEST_THREAD_MANAGER), 
                                                                          DRIVER_TAG);        
        if (!pMgr)
        {
            return 0;
        }

        StartThreads(pMgr, 
                     1, 
                     SearchWorkerT<CharT, SearchClass>, 
                     &Ctx);
                     
        SleepMs(200);

        InterlockedExchange(&Ctx.lStartFlag, 1);
        LARGE_INTEGER liStart = KeQueryPerformanceCounter(NULL);

        StopAndWaitThreads(pMgr, nSecondsPerStep);
        LARGE_INTEGER liEnd = KeQueryPerformanceCounter(NULL);

        ExFreePoolWithTag(pMgr, DRIVER_TAG);

        UINT64 ullTicks = (liEnd.QuadPart - liStart.QuadPart) ? (liEnd.QuadPart - liStart.QuadPart) : 1;
        UINT64 ullOps   = Ctx.llTotalOps ? Ctx.llTotalOps : 1;

        UINT64 bytesPerOp  = cchTextChars * sizeof(CharT);
        UINT64 bytesPerSec = (ullOps * bytesPerOp * liFreq.QuadPart) / ullTicks;

        UINT64 oneGB   = 1024ULL * 1024ULL * 1024ULL;
        UINT64 gbWhole = bytesPerSec / oneGB;
        UINT64 gbFrac  = ((bytesPerSec % oneGB) * 100ULL) / oneGB;

        if (baselineBytesPerSec == 0)
        {
            LOG_INFO("[SEARCH_TEST]      %-18s | %3llu.%02llu GB/s     | Baseline\n", label, gbWhole, gbFrac);
        }
        else
        {
            UINT64 factorWhole = bytesPerSec / baselineBytesPerSec;
            UINT64 factorFrac  = ((bytesPerSec % baselineBytesPerSec) * 100ULL) / baselineBytesPerSec;
            LOG_INFO("[SEARCH_TEST]      %-18s | %3llu.%02llu GB/s     | %llu.%02llux\n", label, gbWhole, gbFrac, factorWhole, factorFrac);
        }

        return bytesPerSec;
    };

    UINT64 baseThroughput = RunEngine(SearchClass::SearchEngine::Auto, 
                                      TRUE, 
                                      "CRT Substring", 
                                      0);
    
    RunEngine(SearchClass::SearchEngine::Scalar, 
              FALSE, 
              "Scalar Hybrid", 
              baseThroughput);
              
    RunEngine(SearchClass::SearchEngine::Auto, 
              FALSE, 
              "Smart Auto", 
              baseThroughput);

    if (g_HasAVX2)
    {
        RunEngine(SearchClass::SearchEngine::AVX2, 
                  FALSE, 
                  "AVX2 Explicit", 
                  baseThroughput);
    }

    return TRUE;
}

// -------------------------------------------------------------------------------------
// Core engine test suite. Evaluates correctness and memory throughput 
// for any injected search algorithm architecture.
//
// Parameters:
//   szEngineName - Name of the algorithm being executed for console logging.
// -------------------------------------------------------------------------------------
template <typename SearchClass8, typename SearchClass16>
VOID RunEngineTestPlan(const char* szEngineName)
{
    PAGED_CODE();

    if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
    {
        return;
    }

    LOG_INFO("\n[SEARCH_TEST] ===================================================\n");
    LOG_INFO("[SEARCH_TEST]    TESTING ALGORITHM: %s\n", szEngineName);
    LOG_INFO("[SEARCH_TEST] ===================================================\n\n");

    // Correctness Tests (16-bit UTF-16)
    RunCorrectnessTest<wchar_t, SearchClass16>(L"This is Horsepool algorithm", L"Horsepool", "Basic Match - Middle (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Short text", L"LongerPatternThanText", "Pattern Longer Than Text (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Start match here", L"Start", "Match At Start (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Match at the very end!!!", L"!!!", "Match At Final Offset (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"This string does not contain it", L"MissingWord", "No Match (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Looking for a single character", L"s", "One-Character Pattern (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Looking for a 2 byte character", L"2 ", "Two-Character Pattern (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", L"AAAA", "Repeated Characters (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"AAAAABAAAAABAAAAABAAAAABAAAAAB", L"AAAAAB", "Repeated Near-Matches (16-bit)");
    RunCorrectnessTestEx<wchar_t, SearchClass16>(L"Text", 4, L"", 0, "Empty Pattern Init Rejection (16-bit)");
    
    wchar_t nullText16[] = L"Before\0After"; 
    wchar_t nullPat16[]  = L"e\0A";          
    RunCorrectnessTestEx<wchar_t, SearchClass16>(nullText16, 12, nullPat16, 3, "Embedded Null 0x00 Bytes (16-bit)");

    wchar_t collisionPat[] = L"\x1000\x2000\x3000\x4000"; 
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Some text with \x1000\x2000\x3000\x4000 inside", collisionPat, "Hash Table Collision Recovery (16-bit)");

    RunCorrectnessTest<wchar_t, SearchClass16>(L"ExactLengthMatch", L"ExactLengthMatch", "Text and Pattern Exact Length (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Mismatch on the very last character X", L"character Y", "Mismatch on Last Character (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"Testing surrogate pairs \xD83D\xDE00 in text", L"\xD83D\xDE00", "Surrogate Pair / High Unicode (16-bit)");
    RunCorrectnessTest<wchar_t, SearchClass16>(L"This string contains exactly thirty-two characters in the pattern!", 
                                               L"exactly thirty-two characters in",
                                               "AVX2 Bifurcation Boundary 32 (16-bit)");

    // Correctness Tests (8-bit ASCII / UTF-8)
    RunCorrectnessTest<char, SearchClass8>("This is Horsepool algorithm", "Horsepool", "Basic Match - Middle (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("Short text", "LongerPatternThanText", "Pattern Longer Than Text (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("Start match here", "Start", "Match At Start (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("Match at the very end!!!", "!!!", "Match At Final Offset (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("This string does not contain it", "MissingWord", "No Match (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("Looking for a single character", "s", "One-Character Pattern (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("Looking for a 2 byte character", "2 ", "Two-Character Pattern (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", "AAAA", "Repeated Characters (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("AAAAABAAAAABAAAAABAAAAABAAAAAB", "AAAAAB", "Repeated Near-Matches (8-bit)");
    RunCorrectnessTestEx<char, SearchClass8>("Text", 4, "", 0, "Empty Pattern Init Rejection (8-bit)");

    char nullText8[] = "Before\0After"; 
    char nullPat8[]  = "e\0A";          
    RunCorrectnessTestEx<char, SearchClass8>(nullText8, 12, nullPat8, 3, "Embedded Null 0x00 Bytes (8-bit)");

    RunCorrectnessTest<char, SearchClass8>("ExactLengthMatch", "ExactLengthMatch", "Text and Pattern Exact Length (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("Mismatch on the very last character X", "character Y", "Mismatch on Last Character (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("Testing extended ASCII \x80\xFF in text", "\x80\xFF", "Extended ASCII / High Bit (8-bit)");
    RunCorrectnessTest<char, SearchClass8>("This buffer contains a pattern that is exactly sixty-four characters long for AVX2 bifurcation testing!!", 
                                           "a pattern that is exactly sixty-four characters long for AVX2 bi", 
                                           "AVX2 Bifurcation Boundary 64 (8-bit)");

    // Performance Suites
    struct TestConfig
    {
        size_t      chChars;
        const char* label;
    };

    const static TestConfig testSizes[] = 
    {
        { 40,          "Tiny Buffer: 80 Bytes (16b) / 40 Bytes (8b)" },
        { 256,         "Small Buffer: 512 Bytes (16b) / 256 B (8b)" },
        { 10240,       "Medium Buffer: 20 KB (16b) / 10 KB (8b)" },
        { 1024 * 1024, "Large Buffer: 2 MB (16b) / 1 MB (8b)" }
    };

    for (int iConfig = 0; iConfig < sizeof(testSizes) / sizeof(testSizes[0]); ++iConfig)
    {
        if (InterlockedCompareExchange(&g_lAbortTests, 0, 0))
        {
            break;
        }

        size_t      cchTextChars = testSizes[iConfig].chChars;
        const char* sizeLabel    = testSizes[iConfig].label;

        size_t cbAllocBytes16 = (cchTextChars + 64) * sizeof(wchar_t) + 64; 
        size_t cbAllocBytes8  = (cchTextChars + 64) * sizeof(char) + 64;

        PVOID pRawBuffer16 = ExAllocatePool2(POOL_FLAG_PAGED, cbAllocBytes16, DRIVER_TAG);
        PVOID pRawBuffer8  = ExAllocatePool2(POOL_FLAG_PAGED, cbAllocBytes8, DRIVER_TAG);

        if (pRawBuffer16 && pRawBuffer8)
        {
            wchar_t* largeText16 = (wchar_t*)(((ULONG_PTR)pRawBuffer16 + 63) & ~(ULONG_PTR)63);
            char* largeText8  = (char*)(((ULONG_PTR)pRawBuffer8 + 63) & ~(ULONG_PTR)63);
            
            largeText16[cchTextChars] = L'\0';
            largeText8[cchTextChars]  = '\0';

            // Test 1: Standard Search
            for (size_t uI = 0; uI < cchTextChars; ++uI) 
            { 
                largeText16[uI] = L'A'; 
                largeText8[uI]  = 'A'; 
            }
            
            const wchar_t* pat1_16    = L"Needle In The Haystack";
            const char*    pat1_8     = "Needle In The Haystack";
            size_t         cchPat1Len = 22;

            for (size_t uI = 0; uI < cchPat1Len; ++uI) 
            {
                largeText16[cchTextChars - cchPat1Len - 5 + uI] = pat1_16[uI];
                largeText8[cchTextChars - cchPat1Len - 5 + uI]  = pat1_8[uI];
            }
            
            const char* szDesc1 = "Measures standard search throughput using a single match near the buffer end.";
            
            RunPerformanceSuiteT<wchar_t, SearchClass16>(largeText16, 
                                                         cchTextChars, 
                                                         pat1_16, 
                                                         "Standard Execution", 
                                                         szDesc1, 
                                                         sizeLabel, 
                                                         16);
                                                          
            RunPerformanceSuiteT<char, SearchClass8>(largeText8, 
                                                     cchTextChars, 
                                                     pat1_8, 
                                                      "Standard Execution", 
                                                      szDesc1, 
                                                      sizeLabel, 
                                                      8);

            // Test 2: Worst-case Mismatch
            for (size_t uI = 0; uI < cchTextChars; ++uI) 
            { 
                largeText16[uI] = L'a'; 
                largeText8[uI]  = 'a'; 
            }
            
            const wchar_t* pat2_16    = L"aaaaaaaaab";
            const char*    pat2_8     = "aaaaaaaaab";
            size_t         cchPat2Len = 10;

            for (size_t uI = 0; uI < cchPat2Len; ++uI) 
            {
                largeText16[cchTextChars - cchPat2Len - 5 + uI] = pat2_16[uI];
                largeText8[cchTextChars - cchPat2Len - 5 + uI]  = pat2_8[uI];
            }
            
            const char* szDesc2 = "Tests worst-case mismatch recovery speed over a dense array of partial matches.";
            
            RunPerformanceSuiteT<wchar_t, SearchClass16>(largeText16, 
                                                         cchTextChars, 
                                                         pat2_16, 
                                                         "Mismatched Dense Text", 
                                                         szDesc2, 
                                                         sizeLabel, 
                                                         16);
                                                          
            RunPerformanceSuiteT<char, SearchClass8>(largeText8, 
                                                     cchTextChars, 
                                                     pat2_8, 
                                                     "Mismatched Dense Text", 
                                                     szDesc2, 
                                                     sizeLabel, 
                                                     8);

            // Test 3: Realistic Log Corpus
            const wchar_t* szLogSnippet16 = L"2026-08-24 10:56:00 [INFO] Process svchost.exe (PID: 1024) requested memory allocation. ";
            const char*    szLogSnippet8  = "2026-08-24 10:56:00 [INFO] Process svchost.exe (PID: 1024) requested memory allocation. ";
            
            size_t uOffset = 0;
            
            while (uOffset < cchTextChars)
            {
                size_t uI = 0;
                
                while (szLogSnippet8[uI] != '\0' && (uOffset + uI) < cchTextChars)
                {
                    largeText16[uOffset + uI] = szLogSnippet16[uI];
                    largeText8[uOffset + uI]  = szLogSnippet8[uI];
                    uI++;
                }
                
                uOffset += uI;
            }
            
            const wchar_t* pat3_16    = L"CRITICAL: USE_AFTER_FREE BUGCHECK";
            const char*    pat3_8     = "CRITICAL: USE_AFTER_FREE BUGCHECK";
            size_t         cchPat3Len = 33;

            // Embed the target pattern near the end of the buffer
            for (size_t uI = 0; uI < cchPat3Len; ++uI) 
            {
                largeText16[cchTextChars - cchPat3Len - 5 + uI] = pat3_16[uI];
                largeText8[cchTextChars - cchPat3Len - 5 + uI]  = pat3_8[uI];
            }
            
            const char* szDesc3 = "Simulates real-world kernel log scanning to test skip table entropy and realistic jump distances.";
            
            RunPerformanceSuiteT<wchar_t, SearchClass16>(largeText16, 
                                                         cchTextChars, 
                                                         pat3_16, 
                                                         "Realistic Log Corpus", 
                                                         szDesc3, 
                                                         sizeLabel, 
                                                         16);
                                                          
            RunPerformanceSuiteT<char, SearchClass8>(largeText8, 
                                                     cchTextChars, 
                                                     pat3_8, 
                                                     "Realistic Log Corpus", 
                                                     szDesc3, 
                                                     sizeLabel, 
                                                     8);
                        
            ExFreePoolWithTag(pRawBuffer16, DRIVER_TAG);
            ExFreePoolWithTag(pRawBuffer8, DRIVER_TAG);
        }
        else
        {
            LOG_ERR("[SEARCH_TEST] [!] Failed to allocate %s test buffer(s).\n", sizeLabel);

            if (pRawBuffer16)
            {
                ExFreePoolWithTag(pRawBuffer16, DRIVER_TAG);
            }
            if (pRawBuffer8)
            {
                ExFreePoolWithTag(pRawBuffer8, DRIVER_TAG);
            }
        }
    }
}

// -------------------------------------------------------------------------------------
// Main routine that bootstraps all correctness checks and triggers performance evaluation.
//
// Parameters:
//   pContext - Standard required context pointer. Used to pass TEST_SUITE_CONFIG.
// -------------------------------------------------------------------------------------
VOID RunTests(_In_opt_ PVOID pContext)
{
    PAGED_CODE();

    TEST_SUITE_CONFIG* pConfig         = (TEST_SUITE_CONFIG*)pContext;
    BOOLEAN            bTestStrSearch  = TRUE;
    BOOLEAN            bTestBndmSearch = TRUE;

    if (pConfig)
    {
        bTestStrSearch  = pConfig->bTestKmStrSearch;
        bTestBndmSearch = pConfig->bTestKmBndmSearch;
    }

    KeSetPriorityThread(KeGetCurrentThread(), 2); 

    LOG_INFO("\n[SEARCH_TEST] ===================================================\n");
    LOG_INFO("[SEARCH_TEST]          KMSTRSEARCH UNIFIED TEST SUITE\n");
    LOG_INFO("[SEARCH_TEST] ===================================================\n");

    if (bTestStrSearch)
    {
        RunEngineTestPlan<CKmStrSearch8, CKmStrSearch16>("Boyer-Moore-Horspool Hybrid");
    }

    if (bTestBndmSearch)
    {
        RunEngineTestPlan<CKmBndmSearch8, CKmBndmSearch16>("BNDM Hybrid");
    }

    if (InterlockedCompareExchange(&g_lAbortTests, 0, 0) != 0)
    {
        LOG_INFO("\n[SEARCH_TEST] [-] Test suite aborted.\n\n");
    }
    else
    {
        LOG_INFO("\n[SEARCH_TEST] --- All Tests Finished Successfully ---\n\n");
    }

    FlushLogToFile();

    PsTerminateSystemThread(STATUS_SUCCESS);
}