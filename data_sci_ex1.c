#include <stdio.h>
#include <arm_neon.h>

float dot_product(const float* a, const float* b) {
    float32x4_t a = vld1q_f32(a);
}