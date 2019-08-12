#ifndef __UTIL
#define __UTIL

double dclock();
float* allocate_matrix(float*** arr, int n, int m);
void deallocate_matrix(float*** arr, float* arr_data);
void copy_matrix(int, int, float *, float *);
void random_matrix(int, int, float **);
void print_matrix(int, int, float **);
float* get_cached_packed_a(int size);
float* get_cached_packed_b(int size);
void free_cached_packed_data();
double compare_matrices( int, int, float **, float **);

#endif
