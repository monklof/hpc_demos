/* Routine for computing C = A[m,k] * B[k,n] + C[m,n] */

#include <mmintrin.h>
#include <xmmintrin.h>  // SSE
#include <pmmintrin.h>  // SSE2
#include <emmintrin.h>  // SSE3

#define bsize_rows 224 // blocking over B, rows & cols
#define bsize_cols 224

#define min( i, j ) ( (i)<(j) ? (i): (j) )

typedef union {
  __m128 f4;
  float f[4];
} float4;

void block_kernel(int m, 
                  int b_i, int b_j, int b_rows, int b_cols,
                  float *a[], float *b[], float *c[]);

void gemm_my(int m, int n, int k, float *a[], float *b[], float *c[]) {
  int b_i, b_j, b_row_len, b_col_len;
  for (b_i = 0; b_i < k; b_i += bsize_rows) {
    b_row_len = min(k-b_i, bsize_rows);
    for (b_j = 0; b_j < n; b_j += bsize_cols) {
      b_col_len = min(n-b_j, bsize_cols);
      /* kernel block @ B[b_i, b_j], with block size (b_row_len, b_col_len) */
      block_kernel(m, b_i, b_j, b_row_len, b_col_len, a, b, c); 
    }
  }
}

void block_kernel(int m, 
                  int b_i, int b_j, int b_rows, int b_cols,
                  float *a[], float *b[], float *c[]){
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
  for (int ai = 0; ai < m; ai++) {
    for (int aj = b_i; aj < b_i + b_rows; aj += 8 ) {

      for (int bj = b_j; bj < b_j + b_cols; bj += 8) {
        // initialize c, we don't load value from mem for further parallizing.
        c0_0_f4.f4 = _mm_setzero_ps(); 
        c0_1_f4.f4 = _mm_setzero_ps(); 
        
        // init b xmm registers
        b0_0_f4.f4 = _mm_load_ps(&b[aj+0][bj+0]);
        b0_1_f4.f4 = _mm_load_ps(&b[aj+0][bj+4]);
        b1_0_f4.f4 = _mm_load_ps(&b[aj+1][bj+0]);
        b1_1_f4.f4 = _mm_load_ps(&b[aj+1][bj+4]);
        b2_0_f4.f4 = _mm_load_ps(&b[aj+2][bj+0]);
        b2_1_f4.f4 = _mm_load_ps(&b[aj+2][bj+4]);
        b3_0_f4.f4 = _mm_load_ps(&b[aj+3][bj+0]);
        b3_1_f4.f4 = _mm_load_ps(&b[aj+3][bj+4]);
        b4_0_f4.f4 = _mm_load_ps(&b[aj+4][bj+0]);
        b4_1_f4.f4 = _mm_load_ps(&b[aj+4][bj+4]);
        b5_0_f4.f4 = _mm_load_ps(&b[aj+5][bj+0]);
        b5_1_f4.f4 = _mm_load_ps(&b[aj+5][bj+4]);
        b6_0_f4.f4 = _mm_load_ps(&b[aj+6][bj+0]);
        b6_1_f4.f4 = _mm_load_ps(&b[aj+6][bj+4]);
        b7_0_f4.f4 = _mm_load_ps(&b[aj+7][bj+0]);
        b7_1_f4.f4 = _mm_load_ps(&b[aj+7][bj+4]);


        // multmul
        a_f4.f4 = _mm_load_ps1(&a[ai][aj+0]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b0_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b0_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[ai][aj+1]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b1_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b1_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[ai][aj+2]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b2_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b2_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[ai][aj+3]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b3_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b3_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[ai][aj+4]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b4_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b4_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[ai][aj+5]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b5_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b5_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[ai][aj+6]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b6_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b6_1_f4.f4); 

        a_f4.f4 = _mm_load_ps1(&a[ai][aj+7]);
        c0_0_f4.f4 += _mm_mul_ps(a_f4.f4, b7_0_f4.f4);
        c0_1_f4.f4 += _mm_mul_ps(a_f4.f4, b7_1_f4.f4);


        // sum up
        
        c[ai][bj+0] += c0_0_f4.f[0];
        c[ai][bj+1] += c0_0_f4.f[1];
        c[ai][bj+2] += c0_0_f4.f[2];
        c[ai][bj+3] += c0_0_f4.f[3];
        c[ai][bj+4] += c0_1_f4.f[0];
        c[ai][bj+5] += c0_1_f4.f[1];
        c[ai][bj+6] += c0_1_f4.f[2];
        c[ai][bj+7] += c0_1_f4.f[3];
        
      }
    }
  }
}
