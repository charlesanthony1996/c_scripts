#include <stdio.h>
#include <arm_neon.h>

int main() {
    int8x16_t a = vdupq_n_s8(2);
    int8x16_t b = vdupq_n_s8(2);
    int8x16_t result = vaddq_s8(a, b);
    int8_t res[16];
    vst1q_s8(res, result);

    printf("Result: ");
    for (int i = 0; i < 16; i++) {
        printf("%d", res[i]);
    }
    printf("\n");
    return 0;
}

