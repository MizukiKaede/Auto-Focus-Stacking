#pragma once
#include <cstdint>
#ifdef _OPENMP
#include <omp.h>
#endif
// Each ctypes caller supplies its budget on the calling OS thread.
static thread_local int native_threads = 1;
API void native_set_threads(int count) { native_threads = count > 0 ? count : 1; }
API int native_openmp_enabled() {
#ifdef _OPENMP
    return 1;
#else
    return 0;
#endif
}
#ifdef _OPENMP
#ifdef _MSC_VER
#define NATIVE_PRAGMA(x) __pragma(x)
#else
#define NATIVE_STRING(x) #x
#define NATIVE_EXPAND(x) NATIVE_STRING(x)
#define NATIVE_PRAGMA(x) _Pragma(NATIVE_EXPAND(x))
#endif
#define NATIVE_FOR(work) NATIVE_PRAGMA(omp parallel for schedule(static) num_threads(native_threads) if(native_threads > 1 && (work) >= 262144))
#define NATIVE_SUM(work, ...) NATIVE_PRAGMA(omp parallel for schedule(static) num_threads(native_threads) if(native_threads > 1 && (work) >= 262144) reduction(+: __VA_ARGS__))
#else
#define NATIVE_FOR(work)
#define NATIVE_SUM(work, ...)
#endif
