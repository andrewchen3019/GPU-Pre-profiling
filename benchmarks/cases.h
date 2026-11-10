#ifndef MEMORY_BENCHMARK_CASES_H
#define MEMORY_BENCHMARK_CASES_H

#include <stddef.h>
#include <string.h>

#define MIB ((size_t)1024 * 1024)

/* Slot 0 = A, slot 1 = B, slot 2 = C. Expected values are in MiB. */
typedef struct {
    int allocate;  /* 1 = allocate, 0 = free */
    int slot;
    size_t size_mib;
    size_t current_mib;
    size_t peak_mib;
} Step;

typedef struct {
    const char *name;
    const Step *steps;
    size_t count;
} Benchmark;

static const Step single_steps[] = {
    {1, 0,  64,  64,  64},
    {0, 0,   0,   0,  64}
};

static const Step sequential_steps[] = {
    {1, 0,  64,  64,  64},
    {0, 0,   0,   0,  64},
    {1, 1, 128, 128, 128},
    {0, 1,   0,   0, 128}
};

static const Step interleaved_steps[] = {
    {1, 0,  64,  64,  64},
    {1, 1, 128, 192, 192},
    {0, 0,   0, 128, 192},
    {1, 2,  96, 224, 224},
    {0, 1,   0,  96, 224},
    {0, 2,   0,   0, 224}
};

static const Benchmark benchmarks[] = {
    {"single", single_steps, sizeof(single_steps) / sizeof(Step)},
    {"sequential", sequential_steps, sizeof(sequential_steps) / sizeof(Step)},
    {"interleaved", interleaved_steps, sizeof(interleaved_steps) / sizeof(Step)}
};

static const Benchmark *find_benchmark(const char *name) {
    size_t i;
    for (i = 0; i < sizeof(benchmarks) / sizeof(Benchmark); ++i) {
        if (strcmp(name, benchmarks[i].name) == 0) return &benchmarks[i];
    }
    return NULL;
}

#endif
