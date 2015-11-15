
#ifdef DOUBLE
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#endif

kernel void jacobi(
  const unsigned Ndim,
  global const TYPE * restrict A,
  global const TYPE * restrict b,
  global const TYPE * restrict xold,
  global       TYPE * restrict xnew,
  local        TYPE * restrict scratch)
{
  size_t row = get_group_id(0);
  size_t lid = get_local_id(0);
  size_t lsz = get_local_size(0);

  // Compute partial sums within each work-item
  TYPE tmp = (TYPE) 0.0;
  for (int col = lid; col < Ndim; )
  {
    tmp += A[row*Ndim + col] * xold[col] * (TYPE)(row != col);
    col+=lsz;
    tmp += A[row*Ndim + col] * xold[col] * (TYPE)(row != col);
    col+=lsz;
    tmp += A[row*Ndim + col] * xold[col] * (TYPE)(row != col);
    col+=lsz;
    tmp += A[row*Ndim + col] * xold[col] * (TYPE)(row != col);
    col+=lsz;
  }

  // Perform work-group reduction to produce final result for this row
  scratch[lid] = tmp;
  barrier(CLK_LOCAL_MEM_FENCE);
  for (int offset = lsz/2; offset > 0; offset/=2)
  {
    if (lid < offset)
      scratch[lid] += scratch[lid + offset];
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (lid == 0)
    xnew[row] = (b[row] - scratch[0]) / A[row*Ndim + row];
}


kernel void convergence(
  global const TYPE * restrict xold,
  global const TYPE * restrict xnew,
  local TYPE * restrict conv_loc,
  global TYPE * restrict conv
  )
{
  size_t i = get_global_id(0);
  TYPE tmp;

  tmp = xnew[i] - xold[i];
  conv_loc[get_local_id(0)] = tmp * tmp;

  barrier(CLK_LOCAL_MEM_FENCE);

  for (int offset = get_local_size(0) / 2; offset > 0; offset /= 2)
  {
    if (get_local_id(0) < offset)
    {
      conv_loc[get_local_id(0)] += conv_loc[get_local_id(0) + offset];
    }
    barrier(CLK_LOCAL_MEM_FENCE);
  }
  if (get_local_id(0) == 0)
  {
    conv[get_group_id(0)] = conv_loc[0];
  }
}
