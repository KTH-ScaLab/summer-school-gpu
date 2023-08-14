#include <sys/time.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h> //Include standard math library containing sqrt.

// Timer
double get_cpu_millisecond() {
   struct timeval tp;
   gettimeofday(&tp,NULL);
   return ((double)tp.tv_sec*1.e6 + (double)tp.tv_usec);
}

// A scaling function to convert integers 0,1,...,N-1 to evenly spaced floats
float scale(int i, int n)
{
  return ((float)i) / (n - 1);
}

// Compute the distance between 2 points on a line.
float distance(float x1, float x2)
{
  return sqrt((x2 - x1)*(x2 - x1));
}

int main(int argc, char** argv)
{
  // Specify a constant value for array length.
  long N = atol(argv[1]);
  
  // Choose a reference value from which distances are measured.
  const float ref = 0.5; 

  // Initialize the output array
  float *out = (float *) calloc(sizeof(float), N);

  double iStart = get_cpu_millisecond();
  for (int i = 0; i < N; ++i)
  {
    float x = scale(i, N);
    out[i] = distance(x, ref);
  }
  double iElaps = get_cpu_millisecond() - iStart;
  //printf("N %ld, Elapsed %.2f ms out[0]=%f\n", N, iElaps, out[0]);
  printf("%.2f, ", iElaps);

  free(out);
  return 0;
}
