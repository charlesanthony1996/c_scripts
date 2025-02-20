#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>

#define N 512

int main(void) {
    float *A = malloc(N * N * sizeof(float));
    float *B = malloc(N * N * sizeof(float));
    float *C = malloc(N * N * sizeof(float));

    if (!A || !B || !C) {
        printf("Memory allocation failed\n");
        return 1;
    }

    for (int i = 0; i < N; i++) {
        for (int j = 0; j < N; j++) {
            A[i * N + j] = (float)(i + j);
            B[i * N + j] = (float)(i - j);
            C[i * N + j] = 0.0f;
        }
    }

    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    for (int i = 0; i < N; i++) {
        for (int j = 0 ; j < N; j++) {
            float sum = 0.0f;
            
            for (int k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }

    clock_gettime(CLOCK_MONOTONIC, &end);
    double elapsed = (end.tv_sec - start.tv_sec) + (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("Elapsed time: %f seconds\n", elapsed);

    printf("C[0][0] = %f\n", C[0]);
    printf("C[N-1][N-1] = %f\n", C[N * N - 1]);

    free(A);
    free(B);
    free(C);

    return 0;
}

