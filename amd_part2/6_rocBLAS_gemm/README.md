# rocBLAS-Examples gemm
Example showing moving matrix and vector data to the GPU device and calling the rocblas gemm (general matrix matrix product) function. Results are fetched from GPU and compared against a CPU implementation.

## Documentation
Run the example without any command line arguments to use default values.
Running with --help will show the options:

    Usage: ./gemm
      --K <value>              Matrix/vector dimension
      --M <value>              Matrix/vector dimension
      --N <value>              Matrix/vector dimension
      --alpha <value>          Alpha scalar
      --beta <value>           Beta scalar

## Building

    make
 
