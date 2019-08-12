%
% Clear all variables and close all graphs
%

clear all
close all

%
% Get max_gflops from /proc/cpuinfo by reading the parameters
% set in file proc_parameters.m
%

proc_parameters

max_gflops = nflops_per_cycle * nprocessors * GHz_of_processor;

%
% Read in the first data set and plot it.
%

output_old

version_old = version;

plot( gemm_my( :,1 ), gemm_my( :,2 ), 'bo-.;OLD;' );
last = size( gemm_my, 1 );

hold on

axis( [ 0 gemm_my( last,1 ) 0 max_gflops ] );

xlabel( 'm = n = k' );
ylabel( 'GFLOPS/sec.' );

%
% Read in second data set and plot it.
%

output_new

version_new = version

title_string = sprintf("OLD = %s, NEW = %s", version_old, version_new);

plot( gemm_my( :,1 ), gemm_my( :,2 ), 'r-*;NEW;' );

title( title_string );

filename = sprintf( "compare_%s_%s", version_old, version_new );

print( filename, '-dpng' );
