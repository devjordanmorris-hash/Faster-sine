// triangle_sine_v2.c
// clang -O3 -march=native -o triangle_sine_v2 triangle_sine_v2.c -lm

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <math.h>
#include <time.h>

#define SEGMENTS    512
#define SAMPLES     1000000
#define ITERATIONS  10

static float table[SEGMENTS + 1];

static inline void build_table(void)
{
    const float pi2 = (float)(M_PI * 0.5);

    for (int i = 0; i <= SEGMENTS; i++) {
        float x = pi2 * ((float)i / SEGMENTS);
        table[i] = sinf(x);
    }
}

static inline float fast_sine(float x)
{
    const float pi     = (float)M_PI;
    const float pi2    = (float)(M_PI * 0.5);
    const float two_pi = (float)(2.0 * M_PI);

    // wrap into [0, 2pi)
    x = fmodf(x, two_pi);
    if (x < 0.0f)
        x += two_pi;

    // symmetry
    float sign = 1.0f;

    if (x > pi) {
        sign = -1.0f;
        x -= pi;
    }

    if (x > pi2)
        x = pi - x;

    // direct index
    const float inv_step = (float)SEGMENTS / pi2;

    float scaled = x * inv_step;

    int idx = (int)scaled;

    if (idx >= SEGMENTS)
        idx = SEGMENTS - 1;

    float frac = scaled - idx;

    float y0 = table[idx];
    float y1 = table[idx + 1];

    float y = y0 + frac * (y1 - y0);

    return sign * y;
}

static void benchmark(void)
{
    float *input = malloc(sizeof(float) * SAMPLES);

    for (int i = 0; i < SAMPLES; i++) {
        input[i] =
            ((float)rand() / RAND_MAX) *
            (float)(2.0 * M_PI);
    }

    printf("=====================================\n");
    printf("Triangle Sine V2\n");
    printf("=====================================\n\n");

    printf("Segments: %d\n", SEGMENTS);
    printf("Table size: %zu bytes\n\n", sizeof(table));

    // ------------------------------------------------
    // sinf benchmark
    // ------------------------------------------------

    clock_t t0 = clock();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < SAMPLES; i++) {
            volatile float y = sinf(input[i]);
            (void)y;
        }
    }

    clock_t t1 = clock();

    double sin_ms =
        1000.0 *
        (double)(t1 - t0) /
        CLOCKS_PER_SEC /
        ITERATIONS;

    double sin_mv =
        (SAMPLES / 1e6) /
        (sin_ms / 1000.0);

    printf("sinf()\n");
    printf("  %.2f ms\n", sin_ms);
    printf("  %.1f M vals/sec\n\n", sin_mv);

    // ------------------------------------------------
    // fast sine benchmark
    // ------------------------------------------------

    t0 = clock();

    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < SAMPLES; i++) {
            volatile float y = fast_sine(input[i]);
            (void)y;
        }
    }

    t1 = clock();

    double fast_ms =
        1000.0 *
        (double)(t1 - t0) /
        CLOCKS_PER_SEC /
        ITERATIONS;

    double fast_mv =
        (SAMPLES / 1e6) /
        (fast_ms / 1000.0);

    printf("fast_sine()\n");
    printf("  %.2f ms\n", fast_ms);
    printf("  %.1f M vals/sec\n\n", fast_mv);

    // ------------------------------------------------
    // accuracy
    // ------------------------------------------------

    float max_err = 0.0f;
    double rms = 0.0;

    for (int i = 0; i < SAMPLES; i++) {

        float exact = sinf(input[i]);
        float approx = fast_sine(input[i]);

        float err = fabsf(exact - approx);

        if (err > max_err)
            max_err = err;

        rms += (double)err * err;
    }

    rms = sqrt(rms / SAMPLES);

    printf("Accuracy\n");
    printf("  Max error : %.3e\n", max_err);
    printf("  RMS error : %.3e\n\n", rms);

    printf("Ratio (fast/sinf): %.2f\n",
           fast_mv / sin_mv);

    free(input);
}

int main(void)
{
    build_table();
    benchmark();
    return 0;
}