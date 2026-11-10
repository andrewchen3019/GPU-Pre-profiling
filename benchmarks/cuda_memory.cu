/* Real CUDA calls. This program does NOT measure the profiler's peak. */
#include <cuda_runtime_api.h>
#include <cstdio>
#include "cases.h"

int main(int argc, char **argv) {
    const Benchmark *test = argc == 2 ? find_benchmark(argv[1]) : NULL;
    if (!test) {
        std::fprintf(stderr, "Usage: %s single|sequential|interleaved\n", argv[0]);
        return 2;
    }

    void *pointers[3] = {NULL, NULL, NULL};
    cudaError_t result = cudaSetDevice(0);
    if (result != cudaSuccess) {
        std::fprintf(stderr, "cudaSetDevice failed: %s\n", cudaGetErrorString(result));
        return 1;
    }

    for (size_t i = 0; i < test->count; ++i) {
        const Step *step = &test->steps[i];
        if (step->allocate) {
            result = cudaMalloc(&pointers[step->slot], step->size_mib * MIB);
        } else {
            result = cudaFree(pointers[step->slot]);
            if (result == cudaSuccess) pointers[step->slot] = NULL;
        }
        if (result != cudaSuccess) {
            std::fprintf(stderr, "%s step %zu (%s) failed: %s\n", test->name,
                         i + 1, step->allocate ? "cudaMalloc" : "cudaFree",
                         cudaGetErrorString(result));
            for (int j = 0; j < 3; ++j) {
                if (pointers[j]) (void)cudaFree(pointers[j]);
            }
            return 1;
        }
        std::printf("%s %c: expected current=%zu MiB, peak=%zu MiB\n",
                    step->allocate ? "allocate" : "free", 'A' + step->slot,
                    step->current_mib, step->peak_mib);
    }

    std::printf("CUDA_CALLS_OK case=%s expected_peak_bytes=%zu\n", test->name,
                test->steps[test->count - 1].peak_mib * MIB);
    std::puts("All CUDA calls succeeded. Printed memory values are expectations, not measurements.");
    return 0;
}
