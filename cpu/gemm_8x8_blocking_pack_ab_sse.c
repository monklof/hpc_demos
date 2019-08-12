/* Routine for computing C = A[m,k] * B[k,n] + C[m,n] */
#include <stdlib.h>
#include <string.h>
#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

#include "util.h"

#define bsize_rows 224 // blocking over B, rows & cols
#define bsize_cols 224

#define min( i, j ) ( (i)<(j) ? (i): (j) )

typedef union {
  __m128 f4;
  float f[4];
} float4;

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
      for (ii = i; ii < i+unroll_size; ii++) {
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
  float4 c0_0_f4, c0_1_f4,
    a_f4, 
    b0_0_f4, b0_1_f4, b1_0_f4, b1_1_f4, b2_0_f4, b2_1_f4, b3_0_f4, b3_1_f4,
    b4_0_f4, b4_1_f4, b5_0_f4, b5_1_f4, b6_0_f4, b6_1_f4, b7_0_f4, b7_1_f4; // b[aj:aj+8,:]

  /* calc everything that related to block @B[b_i, b_j], with shape [b_rows, b_cols] */
  /* loop without tiling */
  /* for (int ai = 0; ai < m; ai++) {  */
  /*   for (int aj = b_i; aj < b_i + b_rows; aj ++ ) { */
  /*     for (int bj = b_j; bj < b_j + b_cols; bj ++ ) { */
  /*       c[ai][bj] += a[ai][aj] * b[aj][bj]; */
  /*     } */
  /*   } */
  /* } */

  
  /* /\* 8x8 version*\/ */
  /* for (int ai = 0; ai < m; ai++) { */
  /*   for (int aj = b_i; aj < b_i + b_rows; aj += 8 ) { */
  /*     for (int bj = b_j; bj < b_j + b_cols; bj += 8) { */
  /*       c[ai][bj] += a[ai][aj] * b[aj][bj]; */
  /*       c[ai][bj+1] += a[ai][aj] * b[aj][bj+1]; */
  /*       /\* ... *\/ */
  /*       c[ai][bj+7] += a[ai][aj] * b[aj][bj+7]; */


  /*       c[ai][bj] += a[ai][aj+1] * b[aj+1][bj]; */
  /*       c[ai][bj+1] += a[ai][aj+1] * b[aj+1][bj+1]; */
  /*       /\* ... *\/ */
  /*       c[ai][bj+7] += a[ai][aj+1] * b[aj+1][bj+7]; */

  /*       /\* ..... */
  /*        *\/ */
        
  /*       c[ai][bj] += a[ai][aj+7] * b[aj+7][bj]; */
  /*       c[ai][bj+1] += a[ai][aj+7] * b[aj+7][bj+1]; */
  /*       /\* ... *\/ */
  /*       c[ai][bj+7] += a[ai][aj+7] * b[aj+7][bj+7]; */
  /*     } */
  /*   } */
  /* } */

  /* tiling with sse */

  int b_packed_index, bj_shifted, a_packed_index;
  a_packed_index = 0;
  for (int ai = 0; ai < m; ai++) {
    // init block packsize.
    b_packed_index = 0;
    for (int aj = b_i; aj < b_i + b_rows; aj += 8, a_packed_index += 8) {
      for (int bj = 0; bj < b_cols; bj += 8) {
        // initialize c, we don't load value from mem for further parallizing.
        c0_0_f4.f4 = _mm_setzero_ps(); 
        c0_1_f4.f4 = _mm_setzero_ps(); 
        
        // init b xmm registers
        b0_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b0_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b1_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b1_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b2_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b2_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b3_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b3_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b4_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b4_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b5_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b5_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b6_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b6_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b7_0_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;
        b7_1_f4.f4 = _mm_load_ps(&b[b_packed_index]); b_packed_index += 4;


        // vectorized multmul
        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+0]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b0_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b0_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+1]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b1_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b1_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+2]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b2_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b2_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+3]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b3_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b3_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+4]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b4_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b4_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+5]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b5_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b5_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+6]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b6_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b6_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[a_packed_index+7]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b7_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b7_1_f4.f4); 


        // sum up
        // shift to align matrix c 
        bj_shifted = bj + b_j;
        c[ai][bj_shifted+0] += c0_0_f4.f[0];
        c[ai][bj_shifted+1] += c0_0_f4.f[1];
        c[ai][bj_shifted+2] += c0_0_f4.f[2];
        c[ai][bj_shifted+3] += c0_0_f4.f[3];
        c[ai][bj_shifted+4] += c0_1_f4.f[0];
        c[ai][bj_shifted+5] += c0_1_f4.f[1];
        c[ai][bj_shifted+6] += c0_1_f4.f[2];
        c[ai][bj_shifted+7] += c0_1_f4.f[3];
        
      }
    }
  }
}
