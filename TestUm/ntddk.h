// -------------------------------------------------------------------------------------
// Copyright (c) 2026 Alexander Danileiko
// SPDX-License-Identifier: MIT
// -------------------------------------------------------------------------------------

// -------------------------------------------------------------------------------------
// ntddk.h (USER MODE MOCK)
// Intercepts kernel inclusions to allow pristine algorithm compilation in UM.
// -------------------------------------------------------------------------------------
#pragma once

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>

// -------------------------------------------------------------------------------------
// Core Types & Memory Management
// -------------------------------------------------------------------------------------
#define PAGED_CODE()
#define POOL_FLAG_PAGED         0
#define POOL_FLAG_NON_PAGED     0
#define POOL_FLAG_CACHE_ALIGNED 0
#define POOL_FLAG_UNINITIALIZED 0

#define NTSTATUS                LONG
#define STATUS_SUCCESS          0L
#define STATUS_UNSUCCESSFUL     0xC0000001
#define NT_SUCCESS(Status)      (((NTSTATUS)(Status)) >= 0)

#define KernelMode              0
#define Executive               0
#define NotificationEvent       0
#define IO_NO_INCREMENT         0

inline PVOID ExAllocatePool2(ULONG  flags, 
                             SIZE_T size, 
                             ULONG  tag)
{
    UNREFERENCED_PARAMETER(flags);
    UNREFERENCED_PARAMETER(tag);
    return _aligned_malloc(size, 64);
}

inline VOID ExFreePoolWithTag(PVOID p, 
                              ULONG tag)
{
    UNREFERENCED_PARAMETER(tag);
    _aligned_free(p);
}

// -------------------------------------------------------------------------------------
// CPU & Processor State (AVX2 / NEON)
// -------------------------------------------------------------------------------------
#define PF_AVX2_INSTRUCTIONS_AVAILABLE 40

inline BOOLEAN ExIsProcessorFeaturePresent(ULONG feature) 
{
    return IsProcessorFeaturePresent(feature) ? TRUE : FALSE;
}

typedef struct _XSTATE_SAVE 
{ 
    int dummy; 
} XSTATE_SAVE;

inline NTSTATUS KeSaveExtendedProcessorState(ULONG64      Mask, 
                                             XSTATE_SAVE* SaveState)
{
    UNREFERENCED_PARAMETER(Mask);
    UNREFERENCED_PARAMETER(SaveState);
    return STATUS_SUCCESS;
}

inline VOID KeRestoreExtendedProcessorState(XSTATE_SAVE* SaveState)
{
    UNREFERENCED_PARAMETER(SaveState);
}

typedef struct _KFLOATING_SAVE 
{ 
    int dummy; 
} KFLOATING_SAVE;

inline NTSTATUS KeSaveFloatingPointState(KFLOATING_SAVE* SaveState)
{
    UNREFERENCED_PARAMETER(SaveState);
    return STATUS_SUCCESS;
}

inline VOID KeRestoreFloatingPointState(KFLOATING_SAVE* SaveState)
{
    UNREFERENCED_PARAMETER(SaveState);
}

// -------------------------------------------------------------------------------------
// Threading & Synchronization
// -------------------------------------------------------------------------------------
typedef HANDLE KEVENT;
typedef HANDLE* PKEVENT;

inline VOID KeInitializeEvent(PKEVENT Event, 
                              int     type, 
                              BOOLEAN state)
{
    UNREFERENCED_PARAMETER(type);
    *Event = CreateEvent(NULL, TRUE, state, NULL);
}

inline VOID KeSetEvent(PKEVENT Event, 
                       int     increment, 
                       BOOLEAN wait)
{
    UNREFERENCED_PARAMETER(increment);
    UNREFERENCED_PARAMETER(wait);
    SetEvent(*Event);
}

inline VOID KeWaitForSingleObject(HANDLE         handle, 
                                  int            waitReason, 
                                  int            waitMode, 
                                  BOOLEAN        alertable, 
                                  PLARGE_INTEGER timeout)
{
    UNREFERENCED_PARAMETER(waitReason);
    UNREFERENCED_PARAMETER(waitMode);
    UNREFERENCED_PARAMETER(alertable);
    UNREFERENCED_PARAMETER(timeout);

    WaitForSingleObject(handle, INFINITE);
}

typedef HANDLE  PETHREAD;
typedef HANDLE* PHANDLE;

typedef VOID (*PKSTART_ROUTINE)(PVOID StartContext);

struct THREAD_START_CTX 
{
    void (*func)(PVOID);
    PVOID ctx;
};

inline DWORD WINAPI ThreadStub(LPVOID p) 
{
    auto ctx = (THREAD_START_CTX*)p;
    ctx->func(ctx->ctx);
    delete ctx;
    return 0;
}

inline NTSTATUS PsCreateSystemThread(PHANDLE         hThread, 
                                     ULONG           access, 
                                     PVOID           objAttr, 
                                     HANDLE          hProcess, 
                                     PVOID           clientId, 
                                     PKSTART_ROUTINE startRoutine, 
                                     PVOID           context) 
{
    UNREFERENCED_PARAMETER(access);
    UNREFERENCED_PARAMETER(objAttr);
    UNREFERENCED_PARAMETER(hProcess);
    UNREFERENCED_PARAMETER(clientId);
    
    auto ctx = new THREAD_START_CTX{(void(*)(PVOID))startRoutine, context};
    *hThread = CreateThread(NULL, 0, ThreadStub, ctx, 0, NULL);
    
    return *hThread ? STATUS_SUCCESS : STATUS_UNSUCCESSFUL;
}

inline VOID PsTerminateSystemThread(NTSTATUS s) 
{
    ExitThread((DWORD)s);
}

inline VOID KeSetPriorityThread(HANDLE h, int prio) 
{
    UNREFERENCED_PARAMETER(prio);
    SetThreadPriority(h, THREAD_PRIORITY_HIGHEST);
}

inline HANDLE KeGetCurrentThread() 
{
    return GetCurrentThread(); 
}

inline NTSTATUS ObReferenceObjectByHandle(HANDLE h, 
                                          ULONG  m, 
                                          PVOID  objType, 
                                          char   p, 
                                          PVOID* outHandle, 
                                          PVOID  info) 
{
    UNREFERENCED_PARAMETER(m);
    UNREFERENCED_PARAMETER(objType);
    UNREFERENCED_PARAMETER(p);
    UNREFERENCED_PARAMETER(info);
    
    HANDLE process = GetCurrentProcess();
    DuplicateHandle(process, h, process, outHandle, 0, FALSE, DUPLICATE_SAME_ACCESS);
    return STATUS_SUCCESS;
}

inline VOID ObDereferenceObject(HANDLE h) 
{ 
    CloseHandle(h); 
}

inline VOID ZwClose(HANDLE h)
{ 
    CloseHandle(h); 
}

inline LARGE_INTEGER KeQueryPerformanceCounter(PLARGE_INTEGER perfFreq)
{
    LARGE_INTEGER res = { 0 };
    if (perfFreq)
    {
        QueryPerformanceFrequency(perfFreq);
        res = *perfFreq;
    }
    else
    {
        QueryPerformanceCounter(&res);
    }
    return res;
}

// -------------------------------------------------------------------------------------
// WDK Native String
// -------------------------------------------------------------------------------------
typedef struct _UNICODE_STRING 
{
    USHORT Length;
    USHORT MaximumLength;
    PWSTR  Buffer;
} UNICODE_STRING, *PUNICODE_STRING;

inline LONG RtlCompareUnicodeString(PUNICODE_STRING s1, PUNICODE_STRING s2, BOOLEAN caseIn) 
{
    return caseIn ? _wcsicmp(s1->Buffer, s2->Buffer) : wcscmp(s1->Buffer, s2->Buffer);
}
