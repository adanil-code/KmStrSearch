// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// HorsepoolDrv.cpp
// Kernel Mode Wrapper 
// -------------------------------------------------------------------------------------
#include <ntifs.h>
#include <ntddk.h>
#include <ntstrsafe.h>
#include <intrin.h>

#include "KmStrSearch16.h"
#include "KmStrSearch8.h"
#include "KmBndmSearch16.h"
#include "KmBndmSearch8.h"

#define DRIVER_TAG 'Hrsp'

// -------------------------------------------------------------------------------------
// Logging Infrastructure (Kernel Buffer & File Output)
// -------------------------------------------------------------------------------------
#define MAX_LOG_LINE 512

PCHAR          g_pLogBuffer        = NULL;
size_t         g_cbLogBufferMax    = 1024 * 1024; // 1 MB Allocation
size_t         g_cbLogBufferOffset = 0;
FAST_MUTEX     g_LogMutex;

// -------------------------------------------------------------------------------------
// Formats and records a log message to both the kernel debugger (DbgPrintEx)
// and a dynamically allocated memory buffer for later file flushing.
// Synchronized via a Fast Mutex to ensure log integrity across concurrent threads.
//
// Parameters:
//   pszFormat - A standard printf-style format string.
//   ...       - Variadic arguments corresponding to the format string.
// -------------------------------------------------------------------------------------
VOID RecordLog(_In_ _Printf_format_string_ PCSTR pszFormat, 
               ...)
{
    va_list args;
    va_start(args, pszFormat);

    char szLine[MAX_LOG_LINE];
    NTSTATUS status = RtlStringCbVPrintfA(szLine, sizeof(szLine), pszFormat, args);
    
    if (NT_SUCCESS(status))
    {
        DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_INFO_LEVEL, "%s", szLine);

        ExAcquireFastMutex(&g_LogMutex);
        
        if (g_pLogBuffer)
        {
            size_t cbLen = 0;
            RtlStringCbLengthA(szLine, MAX_LOG_LINE, &cbLen);
            
            if (g_cbLogBufferOffset + cbLen < g_cbLogBufferMax)
            {
                RtlCopyMemory(g_pLogBuffer + g_cbLogBufferOffset, szLine, cbLen);
                g_cbLogBufferOffset += cbLen;
                g_pLogBuffer[g_cbLogBufferOffset] = '\0';
            }
        }
        
        ExReleaseFastMutex(&g_LogMutex);
    }

    va_end(args);
}

// -------------------------------------------------------------------------------------
// Flushes the accumulated in-memory log buffer to a physical file on disk.
// This function executes standard Zw* file I/O operations and therefore 
// MUST be called at PASSIVE_LEVEL.
// -------------------------------------------------------------------------------------
VOID FlushLogToFile()
{
    PAGED_CODE();

    if (!g_pLogBuffer || g_cbLogBufferOffset == 0)
    {
        return;
    }

    UNICODE_STRING    uniName;
    OBJECT_ATTRIBUTES objAttr;
    
    RtlInitUnicodeString(&uniName, L"\\SystemRoot\\Temp\\KmStrSearchPerf.txt");
    InitializeObjectAttributes(&objAttr, 
                               &uniName, 
                               OBJ_CASE_INSENSITIVE | OBJ_KERNEL_HANDLE, 
                               NULL, 
                               NULL);

    HANDLE          hFile;
    IO_STATUS_BLOCK ioStatusBlock;
    
    NTSTATUS ntStatus = ZwCreateFile(&hFile,
                                     FILE_APPEND_DATA,
                                     &objAttr,
                                     &ioStatusBlock,
                                     NULL,
                                     FILE_ATTRIBUTE_NORMAL,
                                     0,
                                     FILE_OPEN_IF,
                                     FILE_SYNCHRONOUS_IO_NONALERT,
                                     NULL,
                                     0);

    if (NT_SUCCESS(ntStatus))
    {
        ZwWriteFile(hFile, 
                    NULL, 
                    NULL, 
                    NULL, 
                    &ioStatusBlock, 
                    g_pLogBuffer, 
                    (ULONG)g_cbLogBufferOffset, 
                    NULL, 
                    NULL);
                    
        ZwClose(hFile);
    }
    else
    {
        DbgPrintEx(DPFLTR_IHVDRIVER_ID, DPFLTR_ERROR_LEVEL, "[SEARCH_TEST] [!] Failed to create log file: 0x%08X\n", ntStatus);
    }
}

#define LOG_INFO(...)      RecordLog(__VA_ARGS__)
#define LOG_ERR(...)       RecordLog(__VA_ARGS__)

#define COMPILER_BARRIER() _ReadWriteBarrier()

// -------------------------------------------------------------------------------------
// Pauses the current thread execution for a specified duration.
// Wraps KeDelayExecutionThread to provide a standard timing delay mechanism.
// 
// Parameters:
//   ulMs - The amount of time to delay, in milliseconds.
// -------------------------------------------------------------------------------------
inline VOID SleepMs(_In_ ULONG ulMs)
{
    LARGE_INTEGER liDelay;
    liDelay.QuadPart = -(LONGLONG)(ulMs * 10000LL);
    KeDelayExecutionThread(KernelMode, FALSE, &liDelay);
}

// -------------------------------------------------------------------------------------
// Globals Required by the Shared Test Logic
// -------------------------------------------------------------------------------------
volatile LONG  g_lAbortTests = 0;
BOOLEAN        g_HasAVX2     = FALSE;

PETHREAD       g_pMasterBenchmarkThread = NULL;

// -------------------------------------------------------------------------------------
// Inject the Unified Test Logic
// -------------------------------------------------------------------------------------
#include "KmStrSearchShared.h"

// Instruct the shared suite what algorithm to execute
TEST_SUITE_CONFIG g_TestConfig = { TRUE, FALSE };

// -------------------------------------------------------------------------------------
// Driver Teardown & Entry
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// Driver unload routine. Signals all active background test threads
// to abort, waits for their graceful termination, flushes pending logs, and 
// frees all allocated pool memory before the driver is unloaded.
//
// Parameters:
//   pDriverObject - Pointer to the driver object representing this driver instance.
// -------------------------------------------------------------------------------------
VOID DriverUnload(_In_ PDRIVER_OBJECT pDriverObject)
{
    PAGED_CODE();
    UNREFERENCED_PARAMETER(pDriverObject);

    LOG_INFO("\n[SEARCH_TEST] *** Unload requested. Signaling abort...\n");

    InterlockedExchange(&g_lAbortTests, 1);

    if (g_pMasterBenchmarkThread != NULL)
    {
        KeWaitForSingleObject(g_pMasterBenchmarkThread, Executive, KernelMode, FALSE, NULL);
        ObDereferenceObject(g_pMasterBenchmarkThread);
        g_pMasterBenchmarkThread = NULL;
    }

    if (g_pLogBuffer)
    {
        ExFreePoolWithTag(g_pLogBuffer, DRIVER_TAG);
        g_pLogBuffer = NULL;
    }

    LOG_INFO("[SEARCH_TEST] *** Driver Unloaded Safely.\n");
}

extern "C" NTSTATUS DriverEntry(_In_ PDRIVER_OBJECT  pDriverObject, 
                                _In_ PUNICODE_STRING pRegistryPath);

#pragma alloc_text(INIT, DriverEntry)

// -------------------------------------------------------------------------------------
// Driver initialization entry point. Bootstraps the driver environment, 
// initializes synchronization primitives, detects hardware SIMD (AVX2) support, 
// and spawns the master background thread to execute the test suite asynchronously.
//
// Parameters:
//   pDriverObject - Pointer to the driver object created by the I/O manager.
//   pRegistryPath - Pointer to the registry path for this driver (unused).
//
// Returns:
//   STATUS_SUCCESS if the initial allocations and thread spawning succeed;  
//   otherwise, an appropriate NTSTATUS failure code.
// -------------------------------------------------------------------------------------
extern "C" NTSTATUS DriverEntry(_In_ PDRIVER_OBJECT  pDriverObject, 
                                _In_ PUNICODE_STRING pRegistryPath)
{
    PAGED_CODE();
    UNREFERENCED_PARAMETER(pRegistryPath);

    ExInitializeFastMutex(&g_LogMutex);

    g_pLogBuffer = (PCHAR)ExAllocatePool2(POOL_FLAG_PAGED, g_cbLogBufferMax, DRIVER_TAG);
    
    if (g_pLogBuffer)
    {
        g_pLogBuffer[0]     = '\0';
        g_cbLogBufferOffset = 0;
    }

    LOG_INFO("[SEARCH_TEST] *** DriverEntry Called.\n");

    pDriverObject->DriverUnload = DriverUnload;
    
    g_HasAVX2 = ExIsProcessorFeaturePresent(PF_AVX2_INSTRUCTIONS_AVAILABLE);    

    LOG_INFO("[SEARCH_TEST] AVX2 Supported   : %s\n", g_HasAVX2 ? "YES" : "NO");    

    HANDLE   hThread;
    NTSTATUS ntStatus = PsCreateSystemThread(&hThread, 
                                             THREAD_ALL_ACCESS, 
                                             NULL, 
                                             NULL, 
                                             NULL, 
                                             (PKSTART_ROUTINE)RunTests, 
                                             &g_TestConfig);

    if (NT_SUCCESS(ntStatus))
    {
        ObReferenceObjectByHandle(hThread, 
                                  THREAD_ALL_ACCESS, 
                                  NULL, 
                                  KernelMode, 
                                  (PVOID*)&g_pMasterBenchmarkThread, 
                                  NULL);
                                  
        ZwClose(hThread);
    }

    return STATUS_SUCCESS;
}