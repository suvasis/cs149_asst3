#include <stdio.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <driver_functions.h>

#include <thrust/scan.h>
#include <thrust/device_ptr.h>
#include <thrust/device_malloc.h>
#include <thrust/device_free.h>

#include "CycleTimer.h"

#define THREADS_PER_BLOCK 256
#define BLOCK_SIZE 512


static void HandleError( cudaError_t err,
                         const char *file,
                         int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
                file, line );
        exit( EXIT_FAILURE );
    }
    printf( "SUCCESS " );
}
#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))



// helper function to round an integer up to the next power of 2
static inline int nextPow2(int n) {
    n--;
    n |= n >> 1;
    n |= n >> 2;
    n |= n >> 4;
    n |= n >> 8;
    n |= n >> 16;
    n++;
    return n;
}

//forward helper functions
__global__
void printBlkAndThreada();

//forward declaration
// downsweep phase
__global__
void downsweep(int * output, int offset);
//// upsweep phase
__global__
void upsweep(int * output, int offset);

//output[N-1] = 0;
__global__
void clear_last(int * array, int N);

// exclusive_scan --
//
// Implementation of an exclusive scan on global memory array `input`,
// with results placed in global memory `result`.
//
// N is the logical size of the input and output arrays, however
// students can assume that both the start and result arrays we
// allocated with next power-of-two sizes as described by the comments
// in cudaScan().  This is helpful, since your parallel segmented scan
// will likely write to memory locations beyond N, but of course not
// greater than N rounded up to the next power of 2.
//
// Also, as per the comments in cudaScan(), you can implement an
// "in-place" scan, since the timing harness makes a copy of input and
// places it in result
void exclusive_scan(int* input, int N, int* result)
{

    // CS149 TODO:
    //
    // Implement your exclusive scan implementation here.  Keep input
    // mind that although the arguments to this function are device
    // allocated arrays, this is a function that is running in a thread
    // on the CPU.  Your implementation will need to make multiple calls
    // to CUDA kernel functions (that you must write) to implement the
    // scan.


    // up sweep
    int offset = 1;
    int d = N / 2;


    // upsweep phase
    //for (int two_d = 1; two_d <= N/2; two_d*=2) {
    //    int two_dplus1 = 2*two_d;
    //    parallel_for (int i = 0; i < N; i += two_dplus1) {
    //        output[i+two_dplus1-1] += output[i+two_d-1];
    //    }
    //}
    //for (int two_d = 1; two_d <= N/2; two_d*=2)
    for (d; d > BLOCK_SIZE; d /= 2)
    {
        int gridSize = d / BLOCK_SIZE;
        upsweep<<<gridSize, BLOCK_SIZE>>>(result, offset);
        //printBlkAndThreada<<<gridSize, BLOCK_SIZE>>>();
        offset *= 2;
    }
    for (d; d > 0; d /= 2)
    {
        upsweep<<<1, d>>>(result, offset);
        //printBlkAndThreada<<<1, BLOCK_SIZE>>>();
        offset *= 2;
    }

    // clear the last element
    clear_last<<<1, 1>>>(result, N);


    // downsweep phase
    //for (int two_d = N/2; two_d >= 1; two_d /= 2) {
    //    int two_dplus1 = 2*two_d;
    //    parallel_for (int i = 0; i < N; i += two_dplus1) {
    //        int t = output[i+two_d-1];
    //        output[i+two_d-1] = output[i+two_dplus1-1];
    //        output[i+two_dplus1-1] += t;
    //    }
    //}

    // down sweep
    d = 1;
    for (d; d <= BLOCK_SIZE; d *= 2)
    {
            offset /= 2;
            int gridSize = 1;
            downsweep<<<gridSize, d>>>(result, offset);
    }
    for (d; d < N; d *= 2)
    {
            offset /= 2;
            int gridSize = d / BLOCK_SIZE;
            downsweep<<<gridSize, BLOCK_SIZE>>>(result, offset);
    }
    cudaDeviceSynchronize();
}
// downsweep phase
__global__
void downsweep(int * output, int offset) {
    int _idx= threadIdx.x + blockDim.x * blockIdx.x;
    //parallel_for (int i = 0; i < N; i += two_dplus1) {
    //        int t = output[i+two_d-1];
    //        output[i+two_d-1] = output[i+two_dplus1-1];
    //        output[i+two_dplus1-1] += t;
    //    }
    int t = output[offset * (2 * _idx + 1) - 1];
    output[offset * (2 * _idx + 1) - 1] = output[offset * (2 * _idx + 2) - 1];
    output[offset * (2 * _idx + 2) - 1] += t;
}

//// upsweep phase
__global__
void upsweep(int * output, int offset) {
    int _idx= threadIdx.x + blockDim.x * blockIdx.x;
    //parallel_for (int i = 0; i < N; i += two_dplus1) {
    //        output[i+two_dplus1-1] += output[i+two_d-1];
    //    }
    output[offset * (2 * _idx + 2) - 1] += output[offset * (2 * _idx + 1) - 1];
}

//output[N-1] = 0;
__global__
void clear_last(int * array, int N) {
    array[N - 1] = 0;
}

//helper
__global__
void printBlkAndThreada() {
      int myThreadID =
              printf(" ThreadID %d: I am thread %d from block %d \n", myThreadID, threadIdx.x, blockIdx.x);
}
//
// cudaScan --
//
// This function is a timing wrapper around the student's
// implementation of segmented scan - it copies the input to the GPU
// and times the invocation of the exclusive_scan() function
// above. Students should not modify it.
double cudaScan(int* inarray, int* end, int* resultarray)
{
    int* device_result;
    int* device_input;

    // This code rounds the arrays provided to exclusive_scan up
    // to a power of 2, but elements after the end of the original
    // input are left uninitialized and not checked for correctness.
    //
    // Student implementations of exclusive_scan may assume an array's
    // allocated length is a power of 2 for simplicity. This will
    // result in extra work on non-power-of-2 inputs, but it's worth
    // the simplicity of a power of two only solution.

    int rounded_length = nextPow2(end - inarray);

    cudaMalloc((void **)&device_result, sizeof(int) * rounded_length);
    cudaMalloc((void **)&device_input, sizeof(int) * rounded_length);

    // For convenience, both the input and output vectors on the
    // device are initialized to the input values. This means that
    // students are free to implement an in-place scan on the result
        // vector if desired.  If you do this, you will need to keep this
    // in mind when calling exclusive_scan from find_repeats.
    //cudaMemcpy(device_input, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_result, inarray, (end - inarray) * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    exclusive_scan(device_input, rounded_length, device_result);

    // Wait for completion
    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, device_result, (end - inarray) * sizeof(int), cudaMemcpyDeviceToHost);

    // for (int i = 0; i < rounded_length; i++) {
    //     printf("%d ", resultarray[i]);
    // }
    cudaFree(device_result);
    cudaFree(device_input);
    double overallDuration = endTime - startTime;
    return overallDuration;
}
// cudaScanThrust --
//
// Wrapper around the Thrust library's exclusive scan function
// As above in cudaScan(), this function copies the input to the GPU
// and times only the execution of the scan itself.
//
// Students are not expected to produce implementations that achieve
// performance that is competition to the Thrust version, but it is fun to try.
double cudaScanThrust(int* inarray, int* end, int* resultarray) {

    int length = end - inarray;
    thrust::device_ptr<int> d_input = thrust::device_malloc<int>(length);
    thrust::device_ptr<int> d_output = thrust::device_malloc<int>(length);

    cudaMemcpy(d_input.get(), inarray, length * sizeof(int), cudaMemcpyHostToDevice);

    double startTime = CycleTimer::currentSeconds();

    thrust::exclusive_scan(d_input, d_input + length, d_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    cudaMemcpy(resultarray, d_output.get(), length * sizeof(int), cudaMemcpyDeviceToHost);
        thrust::device_free(d_input);
    thrust::device_free(d_output);

    double overallDuration = endTime - startTime;
    return overallDuration;
}

//forward declaration
__global__
void cmp_mark(int *input, int *output, int length);

__global__
void copy(int *input, int *output, int length);

__global__
void accu_index_1(int *prefix_sum, int *repeats_indicate, int *output, int length);

// find_repeats --
//
// Given an array of integers `device_input`, returns an array of all
// indices `i` for which `device_input[i] == device_input[i+1]`.
//
// Returns the total number of pairs found
int find_repeats(int *device_input, int length, int *device_output) {

    // CS149 TODO:
    //
    // Implement this function. You will probably want to
    // make use of one or more calls to exclusive_scan(), as well as
    // additional CUDA kernel launches.
    //
    // Note: As in the scan code, the calling code ensures that
    // allocated arrays are a power of 2 in size, so you can use your
    // exclusive_scan function with them. However, your implementation
    // must ensure that the results of find_repeats are correct given
    // the actual array length.

    //length doesn't work for exclusive_scan needs rounding
    int len = nextPow2(length);
    int *temp;
    cudaMalloc((void **)&temp, len * sizeof(int));

    if (length <= BLOCK_SIZE) {

        //copy the output to temp
        cmp_mark<<<1, length - 1>>>(device_input, temp, length);
        cudaDeviceSynchronize();
        //call exclusive scan on temp
        exclusive_scan(temp, len, temp);
        //copy the temp to device input
        copy<<<1, length - 1>>>(temp, device_input, length);
        cudaDeviceSynchronize();
        //aggregate the indexes.
        accu_index_1<<<1, length - 1>>>(temp, device_input, device_output, length);
    } else {

        int gridSize = (length + BLOCK_SIZE ) / BLOCK_SIZE;
        cmp_mark<<<gridSize, BLOCK_SIZE>>>(device_input, temp, length);
        cudaDeviceSynchronize();
        exclusive_scan(temp, len, temp);
        copy<<<gridSize, BLOCK_SIZE>>>(temp, device_input, length);
        cudaDeviceSynchronize();
        accu_index_1<<<gridSize, BLOCK_SIZE>>>(temp, device_input, device_output, length);
    }
    int  result;
    cudaMemcpy(&result, temp + length - 1, sizeof(int), cudaMemcpyDeviceToHost);
     cudaFree(temp);
    return result;

}
//
// cudaFindRepeats --
//
// Timing wrapper around find_repeats. You should not modify this function.
double cudaFindRepeats(int *input, int length, int *output, int *output_length) {
    int *device_input;
    int *device_output;
    int rounded_length = nextPow2(length);


    cudaMalloc((void **)&device_input, rounded_length * sizeof(int));
    cudaMalloc((void **)&device_output, rounded_length * sizeof(int));
    cudaMemcpy(device_input, input, length * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(device_output, input, length * sizeof(int), cudaMemcpyHostToDevice);

    cudaDeviceSynchronize();
    double startTime = CycleTimer::currentSeconds();

    int result = find_repeats(device_input, length, device_output);

    cudaDeviceSynchronize();
    double endTime = CycleTimer::currentSeconds();

    // set output count and results array
    *output_length = result;
    cudaMemcpy(output, device_output, length * sizeof(int), cudaMemcpyDeviceToHost);

    cudaFree(device_input);
    cudaFree(device_output);

    float duration = endTime - startTime;
    return duration;
}

__global__
void cmp_mark(int *input, int *output, int length) {
    int _idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (_idx < length - 1) {
        output[_idx] = input[_idx] == input[_idx + 1] ? 1 : 0;
    }
}

__global__
void copy(int *input, int *output, int length) {
    int _idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (_idx < length - 1) {
        output[_idx] = input[_idx + 1] - input[_idx];
    }
}
__global__
void accu_index_1(int *prefix_sum, int *repeats_indicate, int *output, int length) {
    int _idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (_idx < length - 1) {
        if (repeats_indicate[_idx] == 1) {
            output[prefix_sum[_idx]] = _idx;
        }
    }
}



void printCudaInfo()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    printf("---------------------------------------------------------\n");
    printf("Found %d CUDA devices\n", deviceCount);

    for (int i=0; i<deviceCount; i++)
    {
        cudaDeviceProp deviceProps;
        cudaGetDeviceProperties(&deviceProps, i);
        printf("Device %d: %s\n", i, deviceProps.name);
        printf("   SMs:        %d\n", deviceProps.multiProcessorCount);
        printf("   Global mem: %.0f MB\n",
               static_cast<float>(deviceProps.totalGlobalMem) / (1024 * 1024));
        printf("   CUDA Cap:   %d.%d\n", deviceProps.major, deviceProps.minor);
    }
    printf("---------------------------------------------------------\n");
}
