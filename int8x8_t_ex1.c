#include <arm_neon.h>
#include <stdio.h>

int main() {
    float32x4_t a = {1.0f, 2.0f, 3.0f, 4.0f};
    float32x4_t b = {5.0f, 6.0f, 7.0f, 8.0f};
    float32x4_t result = vaddq_f32(a, b);
    float res[4];
    vst1q_f32(res, result);
    printf("Result: %f %f %f %f\n", res[0], res[1], res[2], res[3]);
    return 0;
}
