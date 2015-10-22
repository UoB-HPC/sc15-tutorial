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
#include "mm_utils.h"   //a library of basic matrix utilities functions
                        //and some key constants used in this program
                        //(such as TYPE)

#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define TOLERANCE 0.001
#define DEF_SIZE  1024
#define MAX_ITERS 5000
#define LARGE     1000000.0

//#define DEBUG    1     // output a small subset of intermediate values
//#define VERBOSE  1

static cl_uint Ndim = DEF_SIZE;           // A[Ndim][Ndim]

static cl_uint device_index = 0;

void parse_arguments(int argc, char *argv[]);
void check_error(const cl_int err, const char *msg);

int main(int argc, char **argv)
{

  int i,j, iters;
  double start_time, elapsed_time;
  TYPE conv, tmp, err, chksum;
  TYPE *A, *b, *x1, *x2, *xnew, *xold, *xtmp;


  parse_arguments(argc, argv);

  // set matrix dimensions and allocate memory for matrices
  printf(" ndim = %d\n",Ndim);

  A    = (TYPE *) malloc(Ndim*Ndim*sizeof(TYPE));
  b    = (TYPE *) malloc(Ndim*sizeof(TYPE));
  x1   = (TYPE *) malloc(Ndim*sizeof(TYPE));
  x2   = (TYPE *) malloc(Ndim*sizeof(TYPE));

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

  start_time = omp_get_wtime();
//
// jacobi iterative solver
//
  conv  = LARGE;
  iters = 0;
  xnew  = x1;
  xold  = x2;
  while ((conv > TOLERANCE) && (iters<MAX_ITERS))
  {
    iters++;
    xtmp  = xnew;   // don't copy arrays.
    xnew  = xold;   // just swap pointers.
    xold  = xtmp;

    for (i=0; i<Ndim; i++)
    {
      xnew[i] = (TYPE) 0.0;
      for (j=0; j<Ndim;j++)
      {
        if (i != j)
          xnew[i]+= A[i*Ndim + j]*xold[j];
      }
        xnew[i] = (b[i]-xnew[i])/A[i*Ndim+i];

    }

    //
    // test convergence
    //
    conv = 0.0;
    for (i = 0; i < Ndim; i++)
    {
      tmp  = xnew[i]-xold[i];
      conv += tmp*tmp;
    }
    conv = sqrt((double)conv);

#ifdef DEBUG
    printf(" conv = %f \n",(float)conv);
#endif

  }

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
           (float)sqrt(err), (float)chksum);

  free(A);
  free(b);
  free(x1);
  free(x2);
}

void check_error(const cl_int err, const char *msg)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Error %d: %s\n", err, msg);
    exit(EXIT_FAILURE);
  }
}


#define MAX_PLATFORMS     8
#define MAX_DEVICES      16
#define MAX_INFO_STRING 256


unsigned getDeviceList(cl_device_id devices[MAX_DEVICES])
{
  cl_int err;

  // Get list of platforms
  cl_uint numPlatforms = 0;
  cl_platform_id platforms[MAX_PLATFORMS];
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &numPlatforms);
  check_error(err, "getting platforms");

  // Enumerate devices
  unsigned numDevices = 0;
  for (int i = 0; i < numPlatforms; i++)
  {
    cl_uint num = 0;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-numDevices, devices+numDevices, &num);
    check_error(err, "getting deviceS");
    numDevices += num;
  }

  return numDevices;
}

void getDeviceName(cl_device_id device, char name[MAX_INFO_STRING])
{
  cl_device_info info = CL_DEVICE_NAME;
  clGetDeviceInfo(device, info, MAX_INFO_STRING, name, NULL);
}


int parseUInt(const char *str, cl_uint *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}



void parse_arguments(int argc, char *argv[])
{
  for (int i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--list"))
    {
      // Get list of devices
      cl_device_id devices[MAX_DEVICES];
      unsigned numDevices = getDeviceList(devices);

      // Print device names
      if (numDevices == 0)
      {
        printf("No devices found.\n");
      }
      else
      {
        printf("\n");
        printf("Devices:\n");
        for (int i = 0; i < numDevices; i++)
        {
          char name[MAX_INFO_STRING];
          getDeviceName(devices[i], name);
          printf("%2d: %s\n", i, name);
        }
        printf("\n");
      }
      exit(EXIT_SUCCESS);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parseUInt(argv[i], &device_index))
      {
        fprintf(stderr, "Invalid device index\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./jac_solv_ocl_basic [OPTIONS]\n\n");
      printf("Options:\n");
      printf("  -h    --help               Print the message\n");
      printf("        --list               List available devices\n");
      printf("        --device     INDEX   Select device at INDEX\n");
      printf("  NDIM                       Set matrix dimensions to NDIM\n");
      printf("\n");
      exit(EXIT_SUCCESS);
    }
    else
    {
      // Try to parse NDIM
      if (!parseUInt(argv[i], &Ndim))
      {
        printf("Invalid Ndim value\n");
      }
    }
  }
}
