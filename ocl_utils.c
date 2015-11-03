#include "ocl_utils.h"

#include <stdio.h>
#include <string.h>

void check_error(const cl_int err, const char *msg)
{
  if (err != CL_SUCCESS)
  {
    fprintf(stderr, "Error %d: %s\n", err, msg);
    exit(EXIT_FAILURE);
  }
}

unsigned get_device_list(cl_device_id devices[MAX_DEVICES])
{
  cl_int err;

  // Get list of platforms
  cl_uint num_platforms = 0;
  cl_platform_id platforms[MAX_PLATFORMS];
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
  check_error(err, "getting platforms");

  // Enumerate devices
  unsigned num_devices = 0;
  for (int i = 0; i < num_platforms; i++)
  {
    cl_uint num = 0;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-num_devices, devices+num_devices, &num);
    check_error(err, "getting deviceS");
    num_devices += num;
  }

  return num_devices;
}

char *get_kernel_string(const char *file_name)
{
  FILE *file = fopen(file_name, "r");
  if (file == NULL)
  {
    fprintf(stderr, "Error: kernel file not found\n");
    exit(EXIT_FAILURE);
  }
  fseek(file, 0, SEEK_END);
  size_t len = ftell(file);
  fseek(file, 0, SEEK_SET);
  char *result = (char *)calloc(len+1, sizeof(char));
  size_t read = fread(result, sizeof(char), len, file);
  if (read != len)
  {
    fprintf(stderr, "Error reading file\n");
    exit(EXIT_FAILURE);
  }
  return result;
}

int parse_uint(const char *str, cl_uint *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}
