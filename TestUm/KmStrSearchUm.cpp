// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// KmStrSearchUm.cpp
// User Mode Wrapper Application
// -------------------------------------------------------------------------------------

#include <ntddk.h> 
#include <intrin.h>

#include "KmStrSearch16.h"
#include "KmStrSearch8.h"
#include "KmBndmSearch16.h"
#include "KmBndmSearch8.h"

// -------------------------------------------------------------------------------------
// Logging Setup & CPU Barriers
// -------------------------------------------------------------------------------------
#define LOG_INFO(...)      do { printf(__VA_ARGS__); fflush(stdout); } while(0)
#define LOG_ERR(...)       do { printf(__VA_ARGS__); fflush(stdout); } while(0)

#define COMPILER_BARRIER() _ReadWriteBarrier()

inline VOID SleepMs(_In_ ULONG ulMs)
{
    Sleep(ulMs);
}

// -------------------------------------------------------------------------------------
// Globals Required by the Shared Test Logic
// -------------------------------------------------------------------------------------
volatile LONG  g_lAbortTests            = 0;
BOOLEAN        g_HasAVX2                = FALSE;
PETHREAD       g_pMasterBenchmarkThread = NULL;

// -------------------------------------------------------------------------------------
// Stub file I/O for User Mode (Console output only)
// -------------------------------------------------------------------------------------
inline VOID FlushLogToFile()
{
}

// -------------------------------------------------------------------------------------
// Inject the Unified Test Logic
// -------------------------------------------------------------------------------------
#include "KmStrSearchShared.h"

// Instruct the shared suite what algorithm to execute
TEST_SUITE_CONFIG g_TestConfig = { TRUE, FALSE}; 

// -------------------------------------------------------------------------------------
// User-Mode Entry Point
// -------------------------------------------------------------------------------------
int main()
{
    LOG_INFO("[SEARCH_TEST] *** User-Mode Main Called.\n");
    LOG_INFO("[SEARCH_TEST] Performance numbers do not tell much for user mode test suite as the search classes are designed for windows kernel "
             "and should be tested against kernel implementation of strstr/wcsstr. "
             "Run the kernel test suite (KmStrSearchDrv.sys) for accurate performance metrics.\n\n");

    g_HasAVX2 = ExIsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE);

    LOG_INFO("[SEARCH_TEST] AVX2 Supported   : %s\n", g_HasAVX2 ? "YES" : "NO");

    HANDLE   hThread;
    NTSTATUS ntStatus = PsCreateSystemThread(&hThread, 
                                             0, 
                                             NULL, 
                                             NULL, 
                                             NULL, 
                                             (PKSTART_ROUTINE)RunTests,
                                             &g_TestConfig);

    if (NT_SUCCESS(ntStatus))
    {
        ntStatus = ObReferenceObjectByHandle(hThread, 
                                             0, 
                                             NULL, 
                                             0, 
                                             (PVOID*)&g_pMasterBenchmarkThread, 
                                             NULL);
                                  
        ZwClose(hThread);
        
        if (g_pMasterBenchmarkThread != NULL)
        {
            KeWaitForSingleObject(g_pMasterBenchmarkThread, Executive, KernelMode, FALSE, NULL);
            ObDereferenceObject(g_pMasterBenchmarkThread);
            g_pMasterBenchmarkThread = NULL;
        }
    }

    LOG_INFO("[SEARCH_TEST] *** User-Mode Test Application Exited Safely.\n");
    
    return 0;
}