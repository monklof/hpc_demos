#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>

#define abs( x ) ( (x) < 0.0 ? -(x) : (x) )

#define ALIGNMENT 32 // avx 256 bit register required.

float *packed_a = NULL;
float *packed_b = NULL;

/**
 * allocate a 2D array in contiguous memory (row major)
 */
float* allocate_matrix(float*** arr, int n, int m)
{
  *arr = (float**) malloc(n * sizeof(float*));
  float *arr_data;
   posix_memalign((void **) &arr_data, ALIGNMENT, n * m * sizeof(float));
  for(int i=0; i<n; i++)
     (*arr)[i] = arr_data + i * m ;
  return arr_data; //free point
}


void deallocate_matrix(float*** arr, float* arr_data){
    free(arr_data);
    free(*arr);
}


void random_matrix( int m, int n, float *a[]){
  double drand48();
  int i,j;
  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ )
      a[i][j] = 2.0 * (float) drand48( ) - 1.0;
}

void copy_matrix( int m, int n, float *a_data, float *b_data){
  memcpy(b_data, a_data, n*m*sizeof(float));
}



double compare_matrices( int m, int n, float *a[], float *b[]){
  int i, j;
  float max_diff = 0.0, diff;

  for ( j=0; j<n; j++ )
    for ( i=0; i<m; i++ ){
      diff = abs( a[i][j] - b[i][j] );
      max_diff = ( diff > max_diff ? diff : max_diff );
    }

  return max_diff;
}

void print_matrix( int m, int n, float *a[]){
  int i, j;
  for ( i=0; i<m; i++ ){
    for ( j=0; j<n; j++ )
      printf("%le ", a[i][j]);
    printf("\n");
  }
  printf("\n");
}

float* get_cached_packed_a(int size){
  if (NULL == packed_a) {
     posix_memalign((void **) &packed_a, ALIGNMENT, size * sizeof(float));
  }
  return packed_a;
}
float* get_cached_packed_b(int size){
  if (NULL == packed_b) {
     posix_memalign((void **) &packed_b, ALIGNMENT, size * sizeof(float));
  }
  return packed_b;
}
void free_cached_packed_data(){
  if (packed_a != NULL){
    free(packed_a);
    packed_a = NULL;
  }

  if (packed_b != NULL){
    free(packed_b);
    packed_b = NULL;
  }
}

static double gtod_ref_time_sec = 0.0;

/* Adapted from the bl2_clock() routine in the BLIS library */

double dclock()
{
        double         the_time, norm_sec;
        struct timeval tv;

        gettimeofday( &tv, NULL );

        if ( gtod_ref_time_sec == 0.0 )
                gtod_ref_time_sec = ( double ) tv.tv_sec;

        norm_sec = ( double ) tv.tv_sec - gtod_ref_time_sec;

        the_time = norm_sec + tv.tv_usec * 1.0e-6;

        return the_time;
}

