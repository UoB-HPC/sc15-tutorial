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

void parse_arguments(int argc, char *argv[], const char *exe_name,
                     const char *pos_name, const char *pos_help,
                     Arguments *args)
{
  for (int i = 1; i < argc; i++)
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
        printf("\n");
        printf("Devices:\n");
        for (int i = 0; i < num_devices; i++)
        {
          char name[MAX_INFO_STRING];
          clGetDeviceInfo(devices[i], CL_DEVICE_NAME, MAX_INFO_STRING, name, NULL);
          printf("%2d: %s\n", i, name);
        }
        printf("\n");
      }
      exit(EXIT_SUCCESS);
    }
    else if (!strcmp(argv[i], "--device"))
    {
      if (++i >= argc || !parse_uint(argv[i], &args->device_index))
      {
        fprintf(stderr, "Invalid device index\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[i], "--wgsize"))
    {
      if (++i >= argc || !parse_uint(argv[i], &args->wgsize))
      {
        fprintf(stderr, "Invalid WGSIZE\n");
        exit(EXIT_FAILURE);
      }
    }
    else if (!strcmp(argv[i], "--help") || !strcmp(argv[i], "-h"))
    {
      printf("\n");
      printf("Usage: ./%s [OPTIONS]\n\n", exe_name);
      printf("Options:\n");
      printf("  -h    --help               Print the message\n");
      printf("        --list               List available devices\n");
      printf("        --device     INDEX   Select device at INDEX\n");
      printf("        --wgsize     WGSIZE  Set workgroup size to WGSIZE\n");
      printf("  %-12s               %s\n", pos_name, pos_help);
      printf("\n");
      exit(EXIT_SUCCESS);
    }
    else
    {
      // Try to parse positional argument
      if (!parse_uint(argv[i], &args->positional))
      {
        printf("Invalid %s value\n", pos_name);
      }
    }
  }
}
