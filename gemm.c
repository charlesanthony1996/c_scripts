#define _GNU_SOURCE

#include <stdint.h>
#include <time.h>
#include <sched.h>
#include <stdio.h>
#include <assert.h>
#include <math.h>
#include <string.h>
#include <arm_neon.h>
#include <pthread.h>
#include <unistd.h>
#include <stdatomic.h>

#ifndef N
  // Matrix dimensions; must be a multiple of 4.
  #define N 512
#endif

#ifndef NTHREADS
  #define NTHREADS 1
#endif

// Matrices stored in row-major order.
float A[N*N] __attribute__ ((aligned (64)));
float B[N*N] __attribute__ ((aligned (64)));
float C[N*N] __attribute__ ((aligned (64)));
float val[N*N] __attribute__ ((aligned (64)));

// Bf is a “preswizzled” version of B for better cache behavior.
// For NEON we use groups of 4 floats.
float Bf[N*N] __attribute__ ((aligned (64)));

// For the vectorized stores, we view C and Bf as arrays of 4 floats.
float32x4_t *Cm = (float32x4_t*)C;
float32x4_t *Bfm = (float32x4_t*)Bf;

// Timing helper
uint64_t nanos() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec*1000000000 + (uint64_t)start.tv_nsec;
}

// Blocking parameters (tuned for NEON: 4 floats per vector)
#define BLOCK 4      // vector width
#define BLOCK_Y 4    // number of rows processed per block
#define BLOCK_X 2    // number of vector registers processed per block in x

// Matrix multiplication routine
// Computes C[sy:ey, :] = A[sy:ey, :] * B[:,:]
void matmul(int sy, int ey) {
  for (int y = sy; y < ey; y += BLOCK_Y) {
    for (int x = 0; x < N; x += BLOCK * BLOCK_X) {
      // Initialize accumulators for each subblock.
      float32x4_t acc[BLOCK_Y][BLOCK_X];
      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int ix = 0; ix < BLOCK_X; ix++) {
          acc[iy][ix] = vdupq_n_f32(0.0f);
        }
      }

      // Loop over k (columns of A, rows of B)
      // Note: we mimic the AVX version’s indexing.
      for (int k = 0; k < N; k++) {
        for (int iy = 0; iy < BLOCK_Y; iy++) {
          // Broadcast a scalar from A into a NEON vector.
          float32x4_t ta = vdupq_n_f32(A[(y+iy)*N + k]);
          for (int ix = 0; ix < BLOCK_X; ix++) {
            // Compute the index into the preswizzled B.
            // In the AVX code, the index was computed as:
            //    ((x+ix*BLOCK)*N + k*8)/8
            // Here we use 4 instead of 8.
            int index = ((x + ix * BLOCK) * N + k * 4) / 4;
            float32x4_t tb = Bfm[index];
            acc[iy][ix] = vmlaq_f32(acc[iy][ix], ta, tb);
          }
        }
      }
      // Store the accumulated block into C.
      for (int iy = 0; iy < BLOCK_Y; iy++) {
        for (int ix = 0; ix < BLOCK_X; ix++) {
          int index = ((y + iy) * N + x + ix * BLOCK) / 4;
          Cm[index] = acc[iy][ix];
        }
      }
    }
  }
}

// Global variables for thread synchronization.
pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
atomic_int nready = 0;
atomic_int ndone = 0;

// Worker thread routine.
void *matmul_thread(void *n) {
  int k = (int)(intptr_t)n;
  int sy = (N / NTHREADS) * k;
  int ey = (N / NTHREADS) * (k + 1);

  cpu_set_t set;
  CPU_ZERO(&set);
  CPU_SET(k, &set);
  pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &set);

  nready++;

  // Wait for the main thread to release the lock.
  pthread_mutex_lock(&lock);
  pthread_mutex_unlock(&lock);

  matmul(sy, ey);

  ndone++;
  return NULL;
}

int main() {
  printf("hello with %d threads\n", NTHREADS);

#ifdef DEBUG
  // For debug mode, fill A and B with test values.
  for (int i = 0; i < N*N; i++) {
    A[i] = i;
    B[i] = i;
  }
#else
  // In non-debug mode, read matrices from a file.
  FILE *f = fopen("/tmp/matmul", "rb");
  if (f == NULL) {
    printf("please pregenerate python /tmp/matmul file\n");
    return -1;
  }
  fread(A, sizeof(float), N*N, f);
  fread(B, sizeof(float), N*N, f);
  fread(val, sizeof(float), N*N, f);
  fclose(f);
#endif

  // Preswizzle matrix B into Bf.
  // Rearrange B into blocks of 4 rows for efficient NEON loads.
  for (int y = 0; y < N; y += 4) {
    for (int x = 0; x < N; x++) {
      for (int iy = 0; iy < 4; iy++) {
        Bf[y * N + x * 4 + iy] = B[(y + iy) * N + x];
      }
    }
  }

  // Run the multiplication multiple times.
  for (int i = 0; i < 10; i++) {
    memset(C, 0, sizeof(float)*N*N);

#if NTHREADS != 1
    nready = 0;
    ndone = 0;
    pthread_mutex_lock(&lock);
    pthread_t threads[NTHREADS];
    for (int j = 0; j < NTHREADS; j++) {
      pthread_create(&threads[j], NULL, matmul_thread, (void *)(intptr_t)j);
    }
    while (nready != NTHREADS) usleep(1);
#endif

    uint64_t start = nanos();
#if NTHREADS == 1
    matmul(0, N);
#else
    pthread_mutex_unlock(&lock);
    while (ndone != NTHREADS) usleep(1);
#endif
    uint64_t end = nanos();

#if NTHREADS != 1
    for (int j = 0; j < NTHREADS; j++) {
      pthread_join(threads[j], NULL);
    }
#endif

    double gflop = (2.0 * N * N * N) * 1e-9;
    double s = (end - start) * 1e-9;
    printf("%f GFLOP/S -- %.2f ms\n", gflop/s, s*1e3);
  }

#ifndef DEBUG
  // Check the computed C against the reference value.
  for (int k = 0; k < N * N; k++) {
    if (fabsf(C[k] - val[k]) > 1e-3) {
      printf("MISMATCH AT %d, %f != %f\n", k, C[k], val[k]);
      return -1;
    }
  }
  printf("match\n");
#endif

  return 0;
}
