/*
**  PROGRAM: jacobi Solver
**
**  PURPOSE: This program will explore use of a jacobi iterative
**           method to solve a system of linear equations (Ax= b).
**
**           Here is the basic idea behind the method.   Rewrite
**           the matrix A as a Lower Triangular (L), upper triangular
**           (U) and diagonal matrix (D)
**
**                Ax = (L + D + U)x = b
**
**            Carry out the multiplication and rearrange:
**
**                Dx = b - (L+U)x  -->   x = (b-(L+U)x)/D
**
**           We can do this iteratively
**
**                x_new = (b-(L+U)x_old)/D
**
**  USAGE:   Run wtihout arguments to use default SIZE.
**
**              ./jac_solv_ocl_basic
**
**           Run with a single argument for the order of the A
**           matrix ... for example
**
**              ./jac_solv_ocl_basic 2500
**
**  HISTORY: Written by Tim Mattson, Oct 2015
**           Ported to OpenCL by Tom Deakin and James Price, Oct 2015
*/

#include <omp.h>
#include <math.h>
#include <string.h>
#include "ocl_utils.h"
#include "mm_utils.h"   //a library of basic matrix utilities functions
                        //and some key constants used in this program
                        //(such as TYPE)

#define TOLERANCE 0.001
#define DEF_SIZE  1024
#define MAX_ITERS 5000
#define LARGE     1000000.0

//#define DEBUG    1     // output a small subset of intermediate values
//#define VERBOSE  1

int main(int argc, char **argv)
{
  int i,j, iters;
  double start_time, elapsed_time;
  TYPE conv, tmp, err, chksum;
  TYPE *A, *b, *x1, *x2, *xnew, *xold, *conv_tmp;

  unsigned num_devices;
  char device_name[MAX_INFO_STRING];
  cl_device_id devices[MAX_DEVICES];

  cl_int           clerr;
  cl_device_id     device;                        // compute device id
  cl_context       context;                       // compute context
  cl_command_queue commands;                      // compute command queue
  cl_program       program;                       // compute program
  cl_kernel        ko_jacobi;                     // compute kernel
  cl_kernel        ko_convergence;                // convergence kernel
  cl_mem           d_A, d_b, d_x1, d_x2, d_conv;  // device memory objects
  cl_mem           d_xnew;
  cl_mem           d_xold;

  char *kernel_string = NULL;
  char *build_options = NULL;

  cl_uint Ndim;
  Arguments args;


  args.n            = DEF_SIZE;
  args.device_index = 0;
  args.wgsize       = 64;
  parse_arguments(argc, argv, &args);

  Ndim = args.n;

  // Check Ndim is divisible by workgroup size
  if (Ndim % args.wgsize != 0)
  {
    printf("Problem size must be divisible by work-group size of %d\n", args.wgsize);
    exit(EXIT_FAILURE);
  }

  // set matrix dimensions and allocate memory for matrices
  printf(" ndim = %d\n",Ndim);

  A    = (TYPE *) malloc(Ndim*Ndim*sizeof(TYPE));
  b    = (TYPE *) malloc(Ndim*sizeof(TYPE));
  x1   = (TYPE *) malloc(Ndim*sizeof(TYPE));
  x2   = (TYPE *) malloc(Ndim*sizeof(TYPE));
  conv_tmp   = (TYPE *) malloc(Ndim/args.wgsize*sizeof(TYPE));

  if (!A || !b || !x1 || !x2)
  {
    printf("\n memory allocation error\n");
    exit(-1);
  }

  // generate our diagonally dominant matrix, A
  init_diag_dom_near_identity_matrix(Ndim, A);

#ifdef VERBOSE
  mm_print(Ndim, Ndim, A);
#endif

//
// Initialize x and just give b some non-zero random values
//
  for (i = 0; i < Ndim; i++)
  {
    x1[i] = (TYPE)0.0;
    x2[i] = (TYPE)0.0;
    b[i]  = (TYPE)(rand()%51)/100.0;
  }

  // Check device index in range
  num_devices = get_device_list(devices);
  if (args.device_index >= num_devices)
  {
    printf("Invalid device index (try '--list')\n");
    return 1;
  }

  device = devices[args.device_index];

  // Print device name
  clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_INFO_STRING, device_name, NULL);
  printf("\nUsing OpenCL device: %s\n", device_name);

  // Create a compute context
  context = clCreateContext(0, 1, &device, NULL, NULL, &clerr);
  check_error(clerr, "Creating context");

  // Create a command queue
  commands = clCreateCommandQueue(context, device, 0, &clerr);
  check_error(clerr, "Creating command queue");

  // Create the compute program from the source buffer
  kernel_string = get_kernel_string("jac_ocl_basic.cl");
  program = clCreateProgramWithSource(context, 1, (const char **)&kernel_string, NULL, &clerr);
  check_error(clerr, "Creating program");

  // Build the program
  if (sizeof(TYPE) == sizeof(float))
    build_options = "-DTYPE=float";
  else if (sizeof(TYPE) == sizeof(double))
    build_options = "-DTYPE=double -DDOUBLE";
  clerr = clBuildProgram(program, 0, NULL, build_options, NULL, NULL);
  if (clerr == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t len;
    char buffer[2048];

    printf("OpenCL build log:\n");
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
  }
  check_error(clerr, "Building program");

  // Create the compute kernel from the program
  ko_jacobi = clCreateKernel(program, "jacobi", &clerr);
  check_error(clerr, "Creating compute kernel");
  ko_convergence = clCreateKernel(program, "convergence", &clerr);
  check_error(clerr, "Creating convergence kernel");

  // Create the input buffers in device memory
  d_A  = clCreateBuffer(context, CL_MEM_READ_ONLY, Ndim*Ndim*sizeof(TYPE), NULL, &clerr);
  check_error(clerr, "Creating buffer d_A");

  d_b  = clCreateBuffer(context, CL_MEM_READ_ONLY, Ndim*sizeof(TYPE), NULL, &clerr);
  check_error(clerr, "Creating buffer d_b");

  d_x1 = clCreateBuffer(context, CL_MEM_READ_WRITE, Ndim*sizeof(TYPE), NULL, &clerr);
  check_error(clerr, "Creating buffer d_x1");

  d_x2 = clCreateBuffer(context, CL_MEM_READ_WRITE, Ndim*sizeof(TYPE), NULL, &clerr);
  check_error(clerr, "Creating buffer d_x2");

  d_conv = clCreateBuffer(context, CL_MEM_WRITE_ONLY, Ndim/args.wgsize*sizeof(TYPE), NULL, &clerr);
  check_error(clerr, "Creating buffer d_conv");

  // Write initial values to buffers
  clerr = clEnqueueWriteBuffer(commands, d_A, CL_TRUE, 0, Ndim*Ndim*sizeof(TYPE), A, 0, NULL, NULL);
  check_error(clerr, "Copying A to device at d_A");

  clerr = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, Ndim*sizeof(TYPE), b, 0, NULL, NULL);
  check_error(clerr, "Copying b to device at d_b");

  clerr = clEnqueueWriteBuffer(commands, d_x1, CL_TRUE, 0, Ndim*sizeof(TYPE), x1, 0, NULL, NULL);
  check_error(clerr, "Copying x1 to device at d_x1");

  clerr = clEnqueueWriteBuffer(commands, d_x2, CL_TRUE, 0, Ndim*sizeof(TYPE), x2, 0, NULL, NULL);
  check_error(clerr, "Copying x2 to device at d_x2");

  // Set the arguments to our compute kernel
  clerr  = clSetKernelArg(ko_jacobi, 0, sizeof(cl_uint), &Ndim);
  clerr |= clSetKernelArg(ko_jacobi, 1, sizeof(cl_mem), &d_A);
  clerr |= clSetKernelArg(ko_jacobi, 2, sizeof(cl_mem), &d_b);
  check_error(clerr, "Setting compute kernel arguments");

  // Set the arguments to our convergence kernel
  clerr  = clSetKernelArg(ko_convergence, 0, sizeof(cl_mem), &d_x1);
  clerr |= clSetKernelArg(ko_convergence, 1, sizeof(cl_mem), &d_x2);
  clerr |= clSetKernelArg(ko_convergence, 2, args.wgsize*sizeof(TYPE), NULL);
  clerr |= clSetKernelArg(ko_convergence, 3, sizeof(cl_mem), &d_conv);
  check_error(clerr, "Setting converence kernel arguments");

  start_time = omp_get_wtime();
//
// jacobi iterative solver
//
  conv  = LARGE;
  iters = 0;
  xnew  = x1;
  xold  = x2;
  d_xnew = d_x1;
  d_xold = d_x2;
  while ((conv > TOLERANCE) && (iters<MAX_ITERS))
  {
    int ll;
    size_t global[] = {Ndim};
    size_t local[] = {args.wgsize};

    cl_mem d_xtmp = d_xnew;
    d_xnew = d_xold; // don't copy arrays.
    d_xold = d_xtmp; // just swap pointers.
    iters++;

    // Update the xold/xnew kernel arguments
    clerr  = clSetKernelArg(ko_jacobi, 3, sizeof(cl_mem), &d_xold);
    clerr |= clSetKernelArg(ko_jacobi, 4, sizeof(cl_mem), &d_xnew);
    check_error(clerr, "Updating xold/xnew kernel arguments");

    // Enqueue the kernel
    clerr = clEnqueueNDRangeKernel(commands, ko_jacobi, 1, NULL, global, local, 0, NULL, NULL);
    check_error(clerr, "Enqueueing compute kernel");

    // Test convergence
    clerr = clEnqueueNDRangeKernel(commands, ko_convergence, 1, NULL, global, local, 0, NULL, NULL);
    check_error(clerr, "Enqueueing convergence kernel");
    clerr = clEnqueueReadBuffer(commands, d_conv, CL_TRUE, 0, Ndim/args.wgsize*sizeof(TYPE), conv_tmp, 0, NULL, NULL);
    check_error(clerr, "Copying back partial convergence array");
    conv = (TYPE) 0.0;
    for (ll = 0 ; ll < Ndim/args.wgsize; ll++)
      conv += conv_tmp[ll];
    conv = sqrt((double)conv);

#ifdef DEBUG
    printf(" conv = %f \n",(float)conv);
#endif
  }

  clerr = clFinish(commands);
  check_error(clerr, "Running kernels");

  // Read final results
  clerr = clEnqueueReadBuffer(commands, d_xold, CL_TRUE, 0, Ndim*sizeof(TYPE), xold, 0, NULL, NULL);
  check_error(clerr, "Reading final d_xold values");

  clerr = clEnqueueReadBuffer(commands, d_xnew, CL_TRUE, 0, Ndim*sizeof(TYPE), xnew, 0, NULL, NULL);
  check_error(clerr, "Reading final d_xnew values");


  elapsed_time = omp_get_wtime() - start_time;
  printf(" Convergence = %g with %d iterations and %f seconds\n",
          (float)conv, iters, (float)elapsed_time);

  //
  // test answer by multiplying my computed value of x by
  // the input A matrix and comparing the result with the
  // input b vector.
  //
  err    = (TYPE) 0.0;
  chksum = (TYPE) 0.0;

  for (i = 0; i < Ndim; i++)
  {
    xold[i] = (TYPE) 0.0;
      for (j = 0; j < Ndim; j++)
        xold[i] += A[i*Ndim+j]*xnew[j];
    tmp = xold[i] - b[i];
#ifdef DEBUG
    printf(" i=%d, diff = %f,  computed b = %f, input b= %f \n",
            i, (float)tmp, (float)xold[i], (float)b[i]);
#endif
    chksum += xnew[i];
    err += tmp*tmp;
  }

  err = sqrt((double)err);
  printf("jacobi solver: err = %f, solution checksum = %f \n",
           (float)err, (float)chksum);
  if (err > TOLERANCE)
    printf("\nWARNING: solution failed to converge\n\n");

  free(A);
  free(b);
  free(x1);
  free(x2);

  // Release OpenCL objects
  clReleaseMemObject(d_A);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_x1);
  clReleaseMemObject(d_x2);
  clReleaseProgram(program);
  clReleaseKernel(ko_jacobi);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);
}
