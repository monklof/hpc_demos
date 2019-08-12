/* Routine for computing C = A[m,k] * B[k,n] + C[m,n] */

void gemm_my(int m, int n, int k, float *a[], float *b[], float *c[]) {
  int i, j, p;
  for (i = 0; i < m; i ++) { 
    for (j = 0; j < n; j += 8) { // reuse data in cache line (64bytes cache line, 16 floats)
      for (p = 0; p < k; p++) {
        c[i][j] += a[i][p] * b[p][j];
        c[i][j+1] += a[i][p] * b[p][j+1];
        c[i][j+2] += a[i][p] * b[p][j+2];
        c[i][j+3] += a[i][p] * b[p][j+3];
        c[i][j+4] += a[i][p] * b[p][j+4];
        c[i][j+5] += a[i][p] * b[p][j+5];
        c[i][j+6] += a[i][p] * b[p][j+6];
        c[i][j+7] += a[i][p] * b[p][j+7];
      }
    }
  }
}

