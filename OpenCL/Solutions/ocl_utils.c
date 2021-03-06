#include "ocl_utils.h"

#include <stdio.h>
#include <stdlib.h>
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
  cl_int         err;

  cl_uint        num_platforms = 0;
  cl_uint        num_devices = 0;
  cl_platform_id platforms[MAX_PLATFORMS];

  int i;

  // Get list of platforms
  err = clGetPlatformIDs(MAX_PLATFORMS, platforms, &num_platforms);
  check_error(err, "getting platforms");

  // Enumerate devices
  for (i = 0; i < num_platforms; i++)
  {
    cl_uint num = 0;
    err = clGetDeviceIDs(platforms[i], CL_DEVICE_TYPE_ALL,
                         MAX_DEVICES-num_devices, devices+num_devices, &num);
    check_error(err, "getting deviceS");
    num_devices += num;
  }

  return num_devices;
}

char* get_kernel_string(const char *file_name)
{
  FILE *file;
  size_t len;
  size_t read;
  char *result;

  file = fopen(file_name, "r");
  if (file == NULL)
  {
    fprintf(stderr, "Error: kernel file not found\n");
    exit(EXIT_FAILURE);
  }

  // Get length of file
  fseek(file, 0, SEEK_END);
  len = ftell(file);
  fseek(file, 0, SEEK_SET);

  // Read data
  result = (char *)calloc(len+1, sizeof(char));
  read = fread(result, sizeof(char), len, file);
  if (read != len)
  {
    fprintf(stderr, "Error reading file\n");
    exit(EXIT_FAILURE);
  }
  return result;
}

int parse_sizet(const char *str, size_t *output)
{
  char *next;
  *output = strtoul(str, &next, 10);
  return !strlen(next);
}

void parse_arguments(int argc, char *argv[], Arguments *args)
{
  int i;
  for (i = 1; i < argc; i++)
  {
    if (!strcmp(argv[i], "--list"))
    {
      // Get list of devices
      cl_device_id devices[MAX_DEVICES];
      unsigned num_devices = get_device_list(devices);

      // Print device names
      if (num_devices == 0)
      {
        printf("No devices found.\n");
      }
      else
      {
        int d;
        printf("\n");
        printf("Devices:\n");
        for (d = 0; d < num_devices; d++)
        {
          char name[MAX_INFO_STRING];
          clGetDeviceInfo(devices[d], CL_DEVICE_NAME, MAX_INFO_STRING, name, NULL);
          printf("%2d: %s\n", d, name);
        }
        printf("\n");
      }
      exit(EXIT_SUCCESS);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parse_sizet(argv[i], &args->device_index))
      {
        fprintf(stderr, "Invalid device index\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[i], "--wgsize"))
    {
      if (++i >= argc || !parse_sizet(argv[i], &args->wgsize))
      {
        fprintf(stderr, "Invalid WGSIZE\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: %s [OPTIONS]\n\n", argv[0]);
      printf("Options:\n");
      printf("  -h    --help               Print the message\n");
      printf("        --list               List available devices\n");
      printf("        --device     INDEX   Select device at INDEX\n");
      printf("        --wgsize     WGSIZE  Set workgroup size to WGSIZE\n");
      printf("  N                          Set problem size to N\n");
      printf("\n");
      exit(EXIT_SUCCESS);
    }
    else
    {
      // Try to parse positional argument
      if (!parse_sizet(argv[i], &args->n))
      {
        printf("Invalid value for problem size '%s'\n", argv[i]);
        exit(EXIT_FAILURE);
      }
    }
  }
}
