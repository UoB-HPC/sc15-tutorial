
#define TYPE double

#if (TYPE == double)
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

kernel void jacobi(
  const unsigned Ndim,
  global const TYPE * restrict A,
  global const TYPE * restrict b,
  global const TYPE * restrict xold,
  global TYPE * restrict xnew)
{
  size_t i = get_global_id(0);

  xnew[i] = (TYPE) 0.0;
  for (int j = 0; j < Ndim; j++)
  {
    if (i != j)
      xnew[i] += A[i*Ndim + j] * xold[j];
  }
  xnew[i] = (b[i] - xnew[i]) / A[i*Ndim + i];
}
