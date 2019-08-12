#include <stdio.h>
// #include <malloc.h>
#include <stdlib.h>

#include "parameters.h"
#include "util.h"

void gemm_baseline(int, int, int, float **, float **, float **);
void gemm_my(int, int, int, float **, float **, float **);

int main()
{
  int 
    p, 
    m, n, k,
    rep;

  double
    dtime, dtime_best,        
    gflops, 
    diff;

  float 
    **a, **b, **c, **c_ref, **c_old;    
  
  printf( "gemm_my = [\n" );
  /* Pre-allocated packed_a & packed_b data for packing algorithms */
  get_cached_packed_a(PLAST*PLAST);
  get_cached_packed_b(PLAST*PLAST);

  /* WARNING: we assume that the matrix is at aliginment of 8x8,
   * all logic is based on this.
   */
  for ( p=PFIRST; p<=PLAST; p+=PINC ){
    m = p;
    n = p;
    k = p;

    gflops = 2.0 * m * n * k * 1.0e-09;

    /* Note: I create an extra column in A to make sure that
       prefetching beyond the matrix does not cause a segfault ?*/

    float *a_data = allocate_matrix(&a, m, k);
    float *b_data = allocate_matrix(&b, k, n);
    float *c_data = allocate_matrix(&c, m, n);
    float *c_old_data = allocate_matrix(&c_old, m, n);
    float *c_ref_data = allocate_matrix(&c_ref, m, n);

    /* Generate random matrices A, B, Cold */
    random_matrix( m, k, a);
    random_matrix( k, n, b);
    random_matrix( m, n, c_old);
    copy_matrix( m, n, c_old_data, c_ref_data);

    /* Run the reference implementation so the answers can be compared */

    gemm_baseline( m, n, k, a, b, c_ref);

    /* Time the "optimized" implementation */
    for ( rep=0; rep<NREPEATS; rep++ ){
      copy_matrix( m, n, c_old_data, c_data);
      /* Time your implementation */
      dtime = dclock();
      gemm_my( m, n, k, a, b, c);
      dtime = dclock() - dtime;
      if ( rep==0 )
        dtime_best = dtime;
      else
        dtime_best = ( dtime < dtime_best ? dtime : dtime_best );
    }
    diff = compare_matrices( m, n, c, c_ref);

    printf( "%d %le %le \n", p, gflops / dtime_best, diff );
    fflush( stdout );
    deallocate_matrix(&a, a_data);
    deallocate_matrix(&b, b_data);
    deallocate_matrix(&c, c_data);
    deallocate_matrix(&c_ref, c_ref_data);
    deallocate_matrix(&c_old, c_old_data);

  }
  free_cached_packed_data();
  printf( "];\n" );

  exit( 0 );
}


/* Baseline Routine for computing C = A[m,k] * B[k,n] + C[m,n] */
void gemm_baseline( int m, int n, int k, float *a[], float *b[], float *c[])
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

