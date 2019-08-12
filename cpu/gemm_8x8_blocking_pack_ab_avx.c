/* Routine for computing C = A[m,k] * B[k,n] + C[m,n] */
#include <stdlib.h>
#include <string.h>
#include <immintrin.h>  // AVX


#include "util.h"

#define bsize_rows 224 // blocking over B, rows & cols
#define bsize_cols 224

#define min( i, j ) ( (i)<(j) ? (i): (j) )


void block_kernel(int m, 
                  int b_i, int b_j, int b_rows, int b_cols,
                  float *a, float *b, float *c[]);
void pack_b_block(int bi, int bj, int brows, int bcols, int unroll_size,
                  float *b[],
                  float *packed_mem);

void pack_a(int m,
            int bi, int b_rows, float *a[], float *packed_a);

void gemm_my(int m, int n, int k, float *a[], float *b[], float *c[]) {
  int b_i, b_j, b_rows, b_cols;
  float *packed_b = get_cached_packed_b(n*k);
  float *packed_a = get_cached_packed_a(m*k);

  for (b_i = 0; b_i < k; b_i += bsize_rows) {
    b_rows = min(k-b_i, bsize_rows);
    pack_a(m, b_i, b_rows, a, packed_a);
    for (b_j = 0; b_j < n; b_j += bsize_cols) {
      b_cols = min(n-b_j, bsize_cols);
      /* kernel block @ B[b_i, b_j], with block size (b_rows, b_cols) */
      /* optimization: we loop block @ B[b_i, b_j] by column-major order for m times. 
         we could pack it as a continuous memory to make it faster(hit l1 cache & mem prefetching) */
      pack_b_block(b_i, b_j, b_rows, b_cols, 8 /*unrolling piece size*/, b, packed_b);
      block_kernel(m, b_i, b_j, b_rows, b_cols, packed_a, packed_b, c);
      packed_b += (b_rows * b_cols);
    }
    packed_a += (m*b_rows);
  }
}

/*
 * pack a[:,bi:bi+b_rows] -> packed_a, column-major order
 */
void pack_a(int m,
            int bi, int b_rows, float *a[], float *packed_a){
  for (int ai = 0; ai < m; ai++) {
    memcpy(packed_a, &a[ai][bi], b_rows*sizeof(float));
    packed_a += b_rows;
  }
}

/*
 * pack b into kernel-column-major order.
 */
void pack_b_block(int bi, int bj, int brows, int bcols, int unroll_size,
                  float *b[],
                  float *packed_mem) {
  int i, j, ii;
  for (i = bi; i < bi + brows; i += unroll_size){
    for (j = bj; j < bj + bcols; j += unroll_size) {
      for (ii = i; ii < i+unroll_size; ii++){
        memcpy(packed_mem, &b[ii][j], unroll_size * sizeof(float));
        packed_mem += unroll_size;
      }
    }
  }
}

/*
 * a was packed by column-major order
 * b was packed by kernel-column-major order.
 */
void block_kernel(int m,
                  int b_i, int b_j,
                  int b_rows, int b_cols,
                  float *a, float *b, float *c[]){
  // tiling to 8x8

  // intel x64 should have 32 XMM registers
  __m256 c_f8; // b[aj:aj+8,:]

  int bj_shifted;
  float *bp = b, *ap = a;
  for (int ai = 0; ai < m; ai++) {
    // init block packsize.
    bp = b;
    for (int aj = b_i; aj < b_i + b_rows; aj += 8, ap += 8) {
      for (int bj = 0; bj < b_cols; bj += 8) {
        // initialize c, we don't load value from mem for further parallizing.
        bj_shifted = bj + b_j;
        
        c_f8 = _mm256_load_ps(&c[ai][bj_shifted]); 
        
        
        // vectorized multmul
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+0),_mm256_load_ps(bp)); 
        bp += 8;
        
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+1), _mm256_load_ps(bp)); 
        bp += 8;
        
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+2), _mm256_load_ps(bp)); 
        bp += 8;
        
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+3), _mm256_load_ps(bp)); 
        bp += 8;
        
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+4), _mm256_load_ps(bp)); 
        bp += 8;
        
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+5), _mm256_load_ps(bp)); 
        bp += 8;
        
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+6), _mm256_load_ps(bp)); 
        bp += 8;
        
        c_f8 += _mm256_mul_ps(_mm256_broadcast_ss(ap+7), _mm256_load_ps(bp));
        bp += 8;

        // sum up
        _mm256_store_ps(&c[ai][bj_shifted], c_f8);

      }
    }
  }
}
