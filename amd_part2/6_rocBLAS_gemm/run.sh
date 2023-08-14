#!/bin/bash


#srun -n 1 ./6_rocBLAS_gemm --K 16384 --N 16384 --M 16384

#profile with tracing
srun -n 1 rocprof --hip-trace --hsa-trace ./6_rocBLAS_gemm --K 16384 --N 16384 --M 16384

#srun -n 1 rocprof -i my_counters.txt ./6_rocBLAS_gemm --K 16384 --N 16384 --M 16384
