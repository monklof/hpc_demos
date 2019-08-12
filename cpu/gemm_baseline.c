/* Baseline Routine for computing C = A[m,k] * B[k,n] + C[m,n] */
void gemm_my( int m, int n, int k, float *a[], float *b[], float *c[])
{
  int i, j, p;

  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ ){
      for ( p=0; p<k; p++ ){
        c[i][j] += a[i][p]*b[p][j];
      }
    }
  }
}
