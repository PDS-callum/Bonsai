/*
 * membw: measure sustained memory bandwidth on the pot.
 *
 * Token generation on a CPU reads every active weight once per token, so the
 * read bandwidth measured here sets the ceiling in Bonsai's core estimate:
 *
 *     tokens per second  <=  read bandwidth / bytes of active weights
 *
 * Three kernels run over arrays far larger than the CPU caches:
 *
 *     read   sum += a[i]           bytes counted: 8 per element
 *     copy   c[i] = a[i]           bytes counted: 16 per element
 *     triad  a[i] = b[i] + s*c[i]  bytes counted: 24 per element
 *
 * Copy and triad follow the STREAM convention and do not count the extra
 * traffic a write-allocate cache may add. Every kernel runs `repeats` times;
 * each sample is printed so the spread can be studied, not just the best.
 *
 * Output is one JSON object per kernel, one per line.
 *
 * Build: gcc -O3 -march=native -fopenmp -o membw membw.c
 * Usage: membw [size-mib] [repeats]      (threads from OMP_NUM_THREADS)
 */
#define _POSIX_C_SOURCE 200809L

#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#define MAX_REPEATS 1000

static double now_seconds(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
}

static int compare_doubles(const void *x, const void *y) {
    double a = *(const double *)x;
    double b = *(const double *)y;
    return (a > b) - (a < b);
}

/* Sink for the read kernel's result, so the compiler cannot drop the loop. */
static volatile double read_sink;

static double kernel_read(const double *a, size_t n) {
    double total = 0.0;
    /* `simd` lets the reduction be vectorised; without it, floating-point
     * ordering rules would keep one core to a single add chain, too slow to
     * saturate memory from one thread. */
#pragma omp parallel for simd reduction(+ : total) schedule(static)
    for (size_t i = 0; i < n; i++) {
        total += a[i];
    }
    return total;
}

static void kernel_copy(double *restrict c, const double *restrict a, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        c[i] = a[i];
    }
}

static void kernel_triad(double *restrict a, const double *restrict b,
                         const double *restrict c, double s, size_t n) {
#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        a[i] = b[i] + s * c[i];
    }
}

static void report(const char *kernel, double bytes, double *seconds,
                   int repeats, size_t size_mib) {
    double gbs[MAX_REPEATS];
    for (int r = 0; r < repeats; r++) {
        gbs[r] = bytes / seconds[r] / 1e9;
    }

    double sorted[MAX_REPEATS];
    memcpy(sorted, gbs, sizeof(double) * (size_t)repeats);
    qsort(sorted, (size_t)repeats, sizeof(double), compare_doubles);
    double median = repeats % 2
                        ? sorted[repeats / 2]
                        : (sorted[repeats / 2 - 1] + sorted[repeats / 2]) / 2.0;

    printf("{\"kernel\": \"%s\", \"threads\": %d, \"size_mib\": %zu, "
           "\"repeats\": %d, \"best_gbs\": %.3f, \"median_gbs\": %.3f, "
           "\"worst_gbs\": %.3f, \"samples_gbs\": [",
           kernel, omp_get_max_threads(), size_mib, repeats,
           sorted[repeats - 1], median, sorted[0]);
    for (int r = 0; r < repeats; r++) {
        printf("%s%.3f", r ? ", " : "", gbs[r]);
    }
    printf("]}\n");
    fflush(stdout);
}

int main(int argc, char **argv) {
    size_t size_mib = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 256;
    int repeats = argc > 2 ? atoi(argv[2]) : 10;
    if (size_mib < 16 || repeats < 1 || repeats > MAX_REPEATS) {
        fprintf(stderr, "usage: %s [size-mib >= 16] [repeats 1..%d]\n",
                argv[0], MAX_REPEATS);
        return 2;
    }

    size_t n = size_mib * 1024 * 1024 / sizeof(double);
    double *a = malloc(n * sizeof(double));
    double *b = malloc(n * sizeof(double));
    double *c = malloc(n * sizeof(double));
    if (!a || !b || !c) {
        fprintf(stderr, "membw: cannot allocate 3 x %zu MiB\n", size_mib);
        return 1;
    }

#pragma omp parallel for schedule(static)
    for (size_t i = 0; i < n; i++) {
        a[i] = 1.0;
        b[i] = 2.0;
        c[i] = 0.0;
    }

    double seconds[MAX_REPEATS];
    double element = (double)sizeof(double);

    /* One untimed pass of each kernel warms up clocks and page tables. */
    read_sink = kernel_read(a, n);
    kernel_copy(c, a, n);
    kernel_triad(a, b, c, 3.0, n);

    for (int r = 0; r < repeats; r++) {
        double start = now_seconds();
        read_sink = kernel_read(a, n);
        seconds[r] = now_seconds() - start;
    }
    report("read", element * (double)n, seconds, repeats, size_mib);

    for (int r = 0; r < repeats; r++) {
        double start = now_seconds();
        kernel_copy(c, a, n);
        seconds[r] = now_seconds() - start;
    }
    report("copy", 2 * element * (double)n, seconds, repeats, size_mib);

    for (int r = 0; r < repeats; r++) {
        double start = now_seconds();
        kernel_triad(a, b, c, 3.0, n);
        seconds[r] = now_seconds() - start;
    }
    report("triad", 3 * element * (double)n, seconds, repeats, size_mib);

    free(a);
    free(b);
    free(c);
    return 0;
}
