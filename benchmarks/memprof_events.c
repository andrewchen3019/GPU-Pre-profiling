/* Feed the same cases into profiler, without CUDA or a GPU.
 * JSON report goes to stdout; diagnostics go to stderr.
 */
#include <stdio.h>
#include "memprof.h"
#include "cases.h"

int main(int argc, char **argv) {
    const Benchmark *test = argc == 2 ? find_benchmark(argv[1]) : NULL;
    if (!test) {
        fprintf(stderr, "Usage: %s single|sequential|interleaved\n", argv[0]);
        return 2;
    }
    memprof_t *profiler = memprof_create(1);
    if (!profiler) {
        fprintf(stderr, "Could not create profiler\n");
        return 1;
    }

    /* Stable, distinct addresses identify allocations. No large buffers needed. */
    char tokens[3] = {0, 0, 0};
    for (size_t i = 0; i < test->count; ++i) {
        const Step *step = &test->steps[i];
        if (step->allocate) {
            memprof_alloc(profiler, 0, &tokens[step->slot],
                          step->size_mib * MIB, MEMPROF_DEVICE);
        } else {
            memprof_free(profiler, &tokens[step->slot]);
        }
        memprof_device_stats stats;
        if (memprof_get_device_stats(profiler, 0, &stats) != 0) {
            fprintf(stderr, "Could not read device statistics\n");
            memprof_destroy(profiler);
            return 1;
        }
        if (stats.current != step->current_mib * MIB ||
            stats.peak != step->peak_mib * MIB) {
            fprintf(stderr, "FAIL %s step %zu: current=%zu peak=%zu bytes; expected %zu %zu\n",
                    test->name, i + 1, stats.current, stats.peak,
                    step->current_mib * MIB, step->peak_mib * MIB);
            memprof_destroy(profiler);
            return 1;
        }
    }

    int failed = memprof_write_json(profiler, stdout) != 0;
    if (fflush(stdout) != 0) failed = 1;
    memprof_destroy(profiler);
    if (failed) {
        fprintf(stderr, "Could not write profiler report\n");
        return 1;
    }
    fprintf(stderr, "%s: all per-step measurements matched (synthetic events).\n", test->name);
    return 0;
}
