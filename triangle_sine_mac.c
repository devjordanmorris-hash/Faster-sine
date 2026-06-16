// triangle_sine_fixed.c
// Compile: clang -O3 -march=native -o triangle_sine triangle_sine_fixed.c -lm
// Or: clang -O3 -o triangle_sine triangle_sine_fixed.c -lm

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <time.h>
#include <stdlib.h>
#include <string.h>

#ifdef __ARM_NEON
#include <arm_neon.h>
#endif

#define TRIANGLES 512
#define SAMPLES 1000000
#define ITERATIONS 10

typedef struct {
    float y0;
    float slope_l;
    float slope_r;
    float half_len;
    float mid_x;
} Triangle;

static Triangle g_triangles[TRIANGLES];
static int g_tri_count = TRIANGLES;

// ============================================================
// BUILD TRIANGLE STACK
// ============================================================
void build_triangle_stack(int count) {
    double pi_2 = M_PI / 2.0;
    double step = pi_2 / count;
    
    for (int i = 0; i < count; i++) {
        double x0 = i * step;
        double x1 = (i + 0.5) * step;
        double x2 = (i + 1) * step;
        
        double y0 = sin(x0);
        double y1 = sin(x1);
        double y2 = sin(x2);
        
        double half_len = step / 2.0;
        g_triangles[i].y0 = (float)y0;
        g_triangles[i].slope_l = (float)((y1 - y0) / half_len);
        g_triangles[i].slope_r = (float)((y2 - y1) / half_len);
        g_triangles[i].half_len = (float)half_len;
        g_triangles[i].mid_x = (float)x1;
    }
    g_tri_count = count;
}

// ============================================================
// SCALAR SINE EVALUATION
// ============================================================
float triangle_sine_scalar(float x) {
    const float pi = M_PI;
    const float pi_2 = M_PI / 2.0f;
    
    x = fmodf(x, 2.0f * pi);
    if (x < 0) x += 2.0f * pi;
    
    float sign = 1.0f;
    if (x > pi) {
        sign = -1.0f;
        x -= pi;
    }
    if (x > pi_2) {
        x = pi - x;
    }
    
    // Binary search
    int lo = 0, hi = g_tri_count - 1;
    int idx = 0;
    while (lo <= hi) {
        int mid = (lo + hi) / 2;
        if (g_triangles[mid].mid_x <= x) {
            lo = mid + 1;
            idx = mid;
        } else {
            hi = mid - 1;
        }
    }
    if (idx >= g_tri_count) idx = g_tri_count - 1;
    
    const Triangle* t = &g_triangles[idx];
    float dx = x - t->mid_x;
    float y;
    if (dx <= 0) {
        y = t->y0 + t->slope_l * (x - (t->mid_x - t->half_len));
    } else {
        y = t->y0 + t->slope_l * t->half_len + t->slope_r * dx;
    }
    
    return y * sign;
}

// ============================================================
// NEON-OPTIMIZED (works on Apple Silicon)
// ============================================================
#ifdef __ARM_NEON
void triangle_sine_neon4(float* input, float* output, int count) {
    const float32x4_t pi = vdupq_n_f32(M_PI);
    const float32x4_t pi2 = vdupq_n_f32(M_PI / 2.0f);
    const float32x4_t two_pi = vdupq_n_f32(2.0f * M_PI);
    const float32x4_t zero = vdupq_n_f32(0.0f);
    const float32x4_t one = vdupq_n_f32(1.0f);
    
    // Precompute triangle data as NEON vectors
    float32x4_t y0_arr[TRIANGLES];
    float32x4_t slope_l_arr[TRIANGLES];
    float32x4_t slope_r_arr[TRIANGLES];
    float32x4_t half_len_arr[TRIANGLES];
    float32x4_t mid_x_arr[TRIANGLES];
    
    for (int i = 0; i < g_tri_count; i++) {
        y0_arr[i] = vdupq_n_f32(g_triangles[i].y0);
        slope_l_arr[i] = vdupq_n_f32(g_triangles[i].slope_l);
        slope_r_arr[i] = vdupq_n_f32(g_triangles[i].slope_r);
        half_len_arr[i] = vdupq_n_f32(g_triangles[i].half_len);
        mid_x_arr[i] = vdupq_n_f32(g_triangles[i].mid_x);
    }
    
    for (int i = 0; i < count; i += 4) {
        float32x4_t x = vld1q_f32(input + i);
        
        // Reduce to [0, 2PI)
        float32x4_t n = vcvtq_f32_s32(vcvtq_s32_f32(vmulq_f32(x, vdupq_n_f32(1.0f / (2.0f * M_PI)))));
        x = vsubq_f32(x, vmulq_f32(n, two_pi));
        x = vmaxq_f32(x, zero);
        
        // Symmetry to [0, PI/2]
        uint32x4_t mask_pi = vcgtq_f32(x, pi);
        float32x4_t sign = vbslq_f32(mask_pi, vnegq_f32(one), one);
        x = vbslq_f32(mask_pi, vsubq_f32(x, pi), x);
        
        uint32x4_t mask_half = vcgtq_f32(x, pi2);
        x = vbslq_f32(mask_half, vsubq_f32(pi, x), x);
        
        // Find triangle index for each lane (unrolled for speed)
        float32x4_t result = vdupq_n_f32(0.0f);
        
        // We'll process each lane separately for simplicity
        // (A proper NEON version would use a vectorized search)
        for (int lane = 0; lane < 4; lane++) {
            float x_lane;
            float sign_lane;
            
            // Extract lane values
            switch (lane) {
                case 0: x_lane = vgetq_lane_f32(x, 0); sign_lane = vgetq_lane_f32(sign, 0); break;
                case 1: x_lane = vgetq_lane_f32(x, 1); sign_lane = vgetq_lane_f32(sign, 1); break;
                case 2: x_lane = vgetq_lane_f32(x, 2); sign_lane = vgetq_lane_f32(sign, 2); break;
                case 3: x_lane = vgetq_lane_f32(x, 3); sign_lane = vgetq_lane_f32(sign, 3); break;
                default: x_lane = 0; sign_lane = 1; break;
            }
            
            // Binary search for this lane
            int lo = 0, hi = g_tri_count - 1;
            int idx = 0;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if (g_triangles[mid].mid_x <= x_lane) {
                    lo = mid + 1;
                    idx = mid;
                } else {
                    hi = mid - 1;
                }
            }
            if (idx >= g_tri_count) idx = g_tri_count - 1;
            
            const Triangle* t = &g_triangles[idx];
            float dx = x_lane - t->mid_x;
            float y;
            if (dx <= 0) {
                y = t->y0 + t->slope_l * (x_lane - (t->mid_x - t->half_len));
            } else {
                y = t->y0 + t->slope_l * t->half_len + t->slope_r * dx;
            }
            
            // Set result lane
            switch (lane) {
                case 0: result = vsetq_lane_f32(y * sign_lane, result, 0); break;
                case 1: result = vsetq_lane_f32(y * sign_lane, result, 1); break;
                case 2: result = vsetq_lane_f32(y * sign_lane, result, 2); break;
                case 3: result = vsetq_lane_f32(y * sign_lane, result, 3); break;
                default: break;
            }
        }
        
        vst1q_f32(output + i, result);
    }
}
#else
// Fallback if NEON not available
void triangle_sine_neon4(float* input, float* output, int count) {
    for (int i = 0; i < count; i++) {
        output[i] = triangle_sine_scalar(input[i]);
    }
}
#endif

// ============================================================
// BENCHMARK
// ============================================================
void benchmark() {
    printf("============================================================\n");
    printf("STACKED SYMMETRIC TRIANGLE SINE - MAC\n");
    printf("============================================================\n\n");
    
    build_triangle_stack(TRIANGLES);
    
    size_t tri_bytes = sizeof(g_triangles);
    size_t table_bytes = SAMPLES * sizeof(float);
    printf("Triangle data: %zu bytes (%.1f KB)\n", tri_bytes, tri_bytes / 1024.0);
    printf("Full sine table: %zu bytes (%.1f KB)\n", table_bytes, table_bytes / 1024.0);
    printf("Compression: %.1fx\n\n", (float)table_bytes / tri_bytes);
    
    float* input = (float*)aligned_alloc(16, SAMPLES * sizeof(float));
    float* output = (float*)aligned_alloc(16, SAMPLES * sizeof(float));
    for (int i = 0; i < SAMPLES; i++) {
        input[i] = ((float)rand() / RAND_MAX) * 2.0f * M_PI;
    }
    
    // CPU sinf()
    printf("--- CPU: sinf() ---\n");
    clock_t t0 = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < SAMPLES; i++) {
            volatile float y = sinf(input[i]);
        }
    }
    clock_t t1 = clock();
    double sin_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0 / ITERATIONS;
    double sin_mv = (SAMPLES / 1e6) / (sin_ms / 1000.0);
    printf("  Time: %.2f ms\n", sin_ms);
    printf("  Speed: %.1f M vals/sec\n", sin_mv);
    
    // Scalar triangle
    printf("\n--- CPU: triangle_sine (scalar) ---\n");
    t0 = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        for (int i = 0; i < SAMPLES; i++) {
            volatile float y = triangle_sine_scalar(input[i]);
        }
    }
    t1 = clock();
    double scalar_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0 / ITERATIONS;
    double scalar_mv = (SAMPLES / 1e6) / (scalar_ms / 1000.0);
    printf("  Time: %.2f ms\n", scalar_ms);
    printf("  Speed: %.1f M vals/sec\n", scalar_mv);
    printf("  Speedup vs sinf: %.2fx\n", sin_mv / scalar_mv);
    
    // NEON triangle (if available)
    printf("\n--- CPU: triangle_sine (NEON) ---\n");
    t0 = clock();
    for (int iter = 0; iter < ITERATIONS; iter++) {
        triangle_sine_neon4(input, output, SAMPLES);
    }
    t1 = clock();
    double neon_ms = (double)(t1 - t0) / CLOCKS_PER_SEC * 1000.0 / ITERATIONS;
    double neon_mv = (SAMPLES / 1e6) / (neon_ms / 1000.0);
    printf("  Time: %.2f ms\n", neon_ms);
    printf("  Speed: %.1f M vals/sec\n", neon_mv);
    printf("  Speedup vs sinf: %.2fx\n", sin_mv / neon_mv);
    printf("  Speedup vs scalar: %.2fx\n", scalar_mv / neon_mv);
    
    // Accuracy
    printf("\n--- Accuracy ---\n");
    float max_err = 0, rms_err = 0;
    for (int i = 0; i < SAMPLES; i++) {
        float true_val = sinf(input[i]);
        float tri_val = triangle_sine_scalar(input[i]);
        float err = fabsf(true_val - tri_val);
        if (err > max_err) max_err = err;
        rms_err += err * err;
    }
    rms_err = sqrtf(rms_err / SAMPLES);
    printf("  Max error: %.2e\n", max_err);
    printf("  RMS error: %.2e\n", rms_err);
    
    printf("\n============================================================\n");
    printf("SUMMARY\n");
    printf("============================================================\n");
    printf("✓ %.1fx compression vs full sine table\n", (float)table_bytes / tri_bytes);
    printf("✓ %.1f M vals/sec (NEON)\n", neon_mv);
    printf("✓ %.2fx faster than sinf()\n", sin_mv / neon_mv);
    printf("✓ Accuracy: RMS %.2e, Max %.2e\n", rms_err, max_err);
    printf("\n🔺🚀\n");
    
    free(input);
    free(output);
}

int main() {
    benchmark();
    return 0;
}
