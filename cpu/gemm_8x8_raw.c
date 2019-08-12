/* Routine for computing C = A[m,k] * B[k,n] + C[m,n] */

void gemm_my(int m, int n, int k, float *a[], float *b[], float *c[]) {
  int i, j, p;
  for (i = 0; i < m; i += 8) { 
    for (j = 0; j < n; j += 8) { // reuse data in cache line (64bytes cache line, 16 floats)
      for (p = 0; p < k; p++) {
        // calc
        c[i][j] += a[i][p] * b[p][j];
        c[i][j+1] += a[i][p] * b[p][j+1];
        c[i][j+2] += a[i][p] * b[p][j+2];
        c[i][j+3] += a[i][p] * b[p][j+3];
        c[i][j+4] += a[i][p] * b[p][j+4];
        c[i][j+5] += a[i][p] * b[p][j+5];
        c[i][j+6] += a[i][p] * b[p][j+6];
        c[i][j+7] += a[i][p] * b[p][j+7];

        c[i+1][j] += a[i+1][p] * b[p][j];
        c[i+1][j+1] += a[i+1][p] * b[p][j+1];
        c[i+1][j+2] += a[i+1][p] * b[p][j+2];
        c[i+1][j+3] += a[i+1][p] * b[p][j+3];
        c[i+1][j+4] += a[i+1][p] * b[p][j+4];
        c[i+1][j+5] += a[i+1][p] * b[p][j+5];
        c[i+1][j+6] += a[i+1][p] * b[p][j+6];
        c[i+1][j+7] += a[i+1][p] * b[p][j+7];

        c[i+2][j] += a[i+2][p] * b[p][j];
        c[i+2][j+1] += a[i+2][p] * b[p][j+1];
        c[i+2][j+2] += a[i+2][p] * b[p][j+2];
        c[i+2][j+3] += a[i+2][p] * b[p][j+3];
        c[i+2][j+4] += a[i+2][p] * b[p][j+4];
        c[i+2][j+5] += a[i+2][p] * b[p][j+5];
        c[i+2][j+6] += a[i+2][p] * b[p][j+6];
        c[i+2][j+7] += a[i+2][p] * b[p][j+7];


        c[i+3][j] += a[i+3][p] * b[p][j];
        c[i+3][j+1] += a[i+3][p] * b[p][j+1];
        c[i+3][j+2] += a[i+3][p] * b[p][j+2];
        c[i+3][j+3] += a[i+3][p] * b[p][j+3];
        c[i+3][j+4] += a[i+3][p] * b[p][j+4];
        c[i+3][j+5] += a[i+3][p] * b[p][j+5];
        c[i+3][j+6] += a[i+3][p] * b[p][j+6];
        c[i+3][j+7] += a[i+3][p] * b[p][j+7];


        c[i+4][j] += a[i+4][p] * b[p][j];
        c[i+4][j+1] += a[i+4][p] * b[p][j+1];
        c[i+4][j+2] += a[i+4][p] * b[p][j+2];
        c[i+4][j+3] += a[i+4][p] * b[p][j+3];
        c[i+4][j+4] += a[i+4][p] * b[p][j+4];
        c[i+4][j+5] += a[i+4][p] * b[p][j+5];
        c[i+4][j+6] += a[i+4][p] * b[p][j+6];
        c[i+4][j+7] += a[i+4][p] * b[p][j+7];

        c[i+5][j] += a[i+5][p] * b[p][j];
        c[i+5][j+1] += a[i+5][p] * b[p][j+1];
        c[i+5][j+2] += a[i+5][p] * b[p][j+2];
        c[i+5][j+3] += a[i+5][p] * b[p][j+3];
        c[i+5][j+4] += a[i+5][p] * b[p][j+4];
        c[i+5][j+5] += a[i+5][p] * b[p][j+5];
        c[i+5][j+6] += a[i+5][p] * b[p][j+6];
        c[i+5][j+7] += a[i+5][p] * b[p][j+7];

        c[i+6][j] += a[i+6][p] * b[p][j];
        c[i+6][j+1] += a[i+6][p] * b[p][j+1];
        c[i+6][j+2] += a[i+6][p] * b[p][j+2];
        c[i+6][j+3] += a[i+6][p] * b[p][j+3];
        c[i+6][j+4] += a[i+6][p] * b[p][j+4];
        c[i+6][j+5] += a[i+6][p] * b[p][j+5];
        c[i+6][j+6] += a[i+6][p] * b[p][j+6];
        c[i+6][j+7] += a[i+6][p] * b[p][j+7];

        c[i+7][j] += a[i+7][p] * b[p][j];
        c[i+7][j+1] += a[i+7][p] * b[p][j+1];
        c[i+7][j+2] += a[i+7][p] * b[p][j+2];
        c[i+7][j+3] += a[i+7][p] * b[p][j+3];
        c[i+7][j+4] += a[i+7][p] * b[p][j+4];
        c[i+7][j+5] += a[i+7][p] * b[p][j+5];
        c[i+7][j+6] += a[i+7][p] * b[p][j+6];
        c[i+7][j+7] += a[i+7][p] * b[p][j+7];

      }
    }
  }
}
