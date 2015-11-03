//------------------------------------------------------------------------------
//
// Name:       vadd.c
//
// Purpose:    Elementwise addition of two vectors (c = a + b)
//
// HISTORY:    Written by Tim Mattson, December 2009
//             Updated by Tom Deakin and Simon McIntosh-Smith, October 2012
//             Updated by Tom Deakin, July 2013
//             Updated by Tom Deakin, October 2014
//             Updated by Tom Deakin and James Price, October 2015
//
//------------------------------------------------------------------------------

#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#include "ocl_utils.h"

#define TOL    (0.001)   // tolerance used in floating point comparisons

//------------------------------------------------------------------------------
//
// kernel:  vadd
//
// Purpose: Compute the elementwise sum c = a+b
//
// input: a and b float vectors of length count
//
// output: c float vector of length count holding the sum a + b
//

const char *KernelSource = "\n" \
"kernel void vadd(                                                      \n" \
"  global float* a,                                                     \n" \
"  global float* b,                                                     \n" \
"  global float* c,                                                     \n" \
"  const unsigned int count)                                            \n" \
"{                                                                      \n" \
"  int i = get_global_id(0);                                            \n" \
"  if(i < count)                                                        \n" \
"    c[i] = a[i] + b[i];                                                \n" \
"}                                                                      \n" \
"\n";

//------------------------------------------------------------------------------


int main(int argc, char** argv)
{
  int          err;               // error code returned from OpenCL calls

  Arguments args = {1024, 0, 0};
  parse_arguments(argc, argv, &args);

  unsigned count = args.n;
  float*       h_a = (float*) calloc(count, sizeof(float));       // a vector
  float*       h_b = (float*) calloc(count, sizeof(float));       // b vector
  float*       h_c = (float*) calloc(count, sizeof(float));       // c vector (a+b) returned from the compute device

  unsigned int correct;           // number of correct results

  size_t global;                  // global domain size

  cl_device_id     device;     // compute device id
  cl_context       context;       // compute context
  cl_command_queue commands;      // compute command queue
  cl_program       program;       // compute program
  cl_kernel        ko_vadd;       // compute kernel

  cl_mem d_a;                     // device memory used for the input  a vector
  cl_mem d_b;                     // device memory used for the input  b vector
  cl_mem d_c;                     // device memory used for the output c vector

  // Fill vectors a and b with random float values
  unsigned i = 0;
  for(i = 0; i < count; i++)
  {
    h_a[i] = rand() / (float)RAND_MAX;
    h_b[i] = rand() / (float)RAND_MAX;
  }

  // Get list of OpenCL devices
  cl_device_id devices[MAX_DEVICES];
  unsigned num_devices = get_device_list(devices);

  // Check device index in range
  if (args.device_index >= num_devices)
  {
    printf("Invalid device index (try '--list')\n");
    return 1;
  }

  device = devices[args.device_index];

  // Print device name
  char name[MAX_INFO_STRING];
  clGetDeviceInfo(device, CL_DEVICE_NAME, MAX_INFO_STRING, name, NULL);
  printf("\nUsing OpenCL device: %s\n", name);


  // Create a compute context
  context = clCreateContext(0, 1, &device, NULL, NULL, &err);
  check_error(err, "Creating context");

  // Create a command queue
  commands = clCreateCommandQueue(context, device, 0, &err);
  check_error(err, "Creating command queue");

  // Create the compute program from the source buffer
  program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
  check_error(err, "Creating program");

  // Build the program
  err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
  if (err == CL_BUILD_PROGRAM_FAILURE)
  {
    size_t len;
    char buffer[2048];

    printf("OpenCL build log:\n");
    clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
    printf("%s\n", buffer);
  }
  check_error(err, "Building program");

  // Create the compute kernel from the program
  ko_vadd = clCreateKernel(program, "vadd", &err);
  check_error(err, "Creating kernel");

  // Create the input (a, b) and output (c) arrays in device memory
  d_a  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
  check_error(err, "Creating buffer d_a");

  d_b  = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(float) * count, NULL, &err);
  check_error(err, "Creating buffer d_b");

  d_c  = clCreateBuffer(context,  CL_MEM_WRITE_ONLY, sizeof(float) * count, NULL, &err);
  check_error(err, "Creating buffer d_c");

  // Write a and b vectors into compute device memory
  err = clEnqueueWriteBuffer(commands, d_a, CL_TRUE, 0, sizeof(float) * count, h_a, 0, NULL, NULL);
  check_error(err, "Copying h_a to device at d_a");

  err = clEnqueueWriteBuffer(commands, d_b, CL_TRUE, 0, sizeof(float) * count, h_b, 0, NULL, NULL);
  check_error(err, "Copying h_b to device at d_b");

  // Set the arguments to our compute kernel
  err  = clSetKernelArg(ko_vadd, 0, sizeof(cl_mem), &d_a);
  err |= clSetKernelArg(ko_vadd, 1, sizeof(cl_mem), &d_b);
  err |= clSetKernelArg(ko_vadd, 2, sizeof(cl_mem), &d_c);
  err |= clSetKernelArg(ko_vadd, 3, sizeof(unsigned int), &count);
  check_error(err, "Setting kernel arguments");

  double rtime = omp_get_wtime();

  // Execute the kernel over the entire range of our 1d input data set
  // letting the OpenCL runtime choose the work-group size
  global = count;
  err = clEnqueueNDRangeKernel(commands, ko_vadd, 1, NULL, &global, NULL, 0, NULL, NULL);
  check_error(err, "Enqueueing kernel");

  // Wait for the commands to complete before stopping the timer
  err = clFinish(commands);
  check_error(err, "Waiting for kernel to finish");

  rtime = omp_get_wtime() - rtime;
  printf("\nThe kernel ran in %lf seconds\n",rtime);

  // Read back the results from the compute device
  err = clEnqueueReadBuffer( commands, d_c, CL_TRUE, 0, sizeof(float) * count, h_c, 0, NULL, NULL );
  check_error(err, "Reading results");

  // Test the results
  correct = 0;
  float tmp;

  for(i = 0; i < count; i++)
  {
    tmp = h_a[i] + h_b[i];     // assign element i of a+b to tmp
    tmp -= h_c[i];             // compute deviation of expected and output result
    if(tmp*tmp < TOL*TOL)        // correct if square deviation is less than tolerance squared
        correct++;
    else {
        printf(" tmp %f h_a %f h_b %f h_c %f \n",tmp, h_a[i], h_b[i], h_c[i]);
    }
  }

  // summarise results
  printf("C = A+B:  %d out of %d results were correct.\n", correct, count);

  // cleanup then shutdown
  clReleaseMemObject(d_a);
  clReleaseMemObject(d_b);
  clReleaseMemObject(d_c);
  clReleaseProgram(program);
  clReleaseKernel(ko_vadd);
  clReleaseCommandQueue(commands);
  clReleaseContext(context);

  free(h_a);
  free(h_b);
  free(h_c);

#if defined(_WIN32) && !defined(__MINGW32__)
  system("pause");
#endif

  return 0;
}
