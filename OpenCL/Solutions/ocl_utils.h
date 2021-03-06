#ifdef __APPLE__
  #include <OpenCL/cl.h>
#else
  #include <CL/cl.h>
#endif

#define MAX_PLATFORMS     8
#define MAX_DEVICES      16
#define MAX_INFO_STRING 256

typedef struct
{
  size_t n;
  size_t device_index;
  size_t wgsize;
} Arguments;

// Check an OpenCL error code.
// If err != CL_SUCCESS, print out error along with 'msg', and exit program.
void check_error(const cl_int err, const char *msg);

// Get list of OpenCL devices across all platforms.
// Return number of devices found.
unsigned get_device_list(cl_device_id devices[MAX_DEVICES]);

// Load a file into a string.
// Exits program on failure.
char* get_kernel_string(const char *file_name);

// Parse command-line arguments.
// args provides default values for each argument, and will receive the final parsed values.
void parse_arguments(int argc, char *argv[], Arguments *args);
