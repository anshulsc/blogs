---
layout: post
title: How to Program Your GPUs? Learning Cuda...
date:   2024-01-07 12:57:49 +0000
categories: jekyll update
excerpt: how to operate the parallel chips if i ever become GPU rich...
mathjax: true
---


I've been thinking about diving into parallel programming. No matter what new AI technology I learn, whether it's the latest LLM models or techniques like flash attention to speed up inference, I keep circling back to the same thought: How do I train and implement this efficiently using CUDA code? Do I have the necessary technology to train it on multiple GPUs, whether on the cloud or a local machine? So, I've made up my mind to delve into the internals of GPU workings and finally understand this black box. In this series of blogs, I will document my journey of learning how to program this parallel processor and effectively utilize its capabilities.

### Why Do We Need GPUs? A Brief History of Processors

Let's start from the beginning. Why do we need GPUs in the first place? Weren't CPUs enough? To answer this, we need to talk about the history of processors, which begins with Moore's Law: the number of transistors per unit area doubles every 18-24 months, doubling computation. This trend has continued to this day. As transistors became smaller we were able fit a chunk of them into a single chip, hence it became easier to switch them on and off faster, allowing us to increase the clock frequency of processors. Processor frequency increased until we hit a limit: transistors became so small that we could no longer switch them on and off reliably. The processor frequency stagnated due to power limitations—switching transistors faster generates heat, which can melt the CPU, and our cooling technology was not capable of handling this.

However, we still had a large number of transistors that could fit into small chips. We needed a way to utilize this power, leading to the development of multi-core processors, which made parallel computing mainstream. This brought about new design approaches for these chips:

- **Latency-oriented design**: Minimizes the time to perform a single task. Think of how fast you can transfer data.
- **Throughput-oriented design**: Maximizes the number of tasks that can be performed in a given time frame. Think of how much data you can send at once.

CPUs and GPUs are built using these design approaches. The CPU is an example of a latency-oriented design, whereas the GPU is an example of a throughput-oriented design. If you want to accomplish a task as quickly as possible, you use a CPU. If you want to perform many small tasks within a given timeframe, a GPU is the best option.

### Design of CPUs and GPUs

Processors have various components: the Control Unit (CU), Arithmetic Logic Unit (ALU), and cache. If you look at a diagram, you'll see that CPUs have very few powerful ALUs optimized to reduce the latency of single operations, whether multiplication or something else. In contrast, GPUs have many ALUs that are not as powerful and take more time than CPUs to perform operations, but they are heavily pipelined for high throughput. The same goes for cache: GPUs have smaller caches because more area is dedicated to computation.

GPUs were originally built for graphics and gaming, hence the name Graphics Processing Unit. Over time, people started using GPUs for tasks requiring high throughput, like guessing the random seed in bitcoin mining. This led to the development of general-purpose GPU programming, with NVIDIA releasing CUDA as a programming interface for using GPUs in a general-purpose way. Now, GPUs are the go-to for machine learning, and this is why demand for GPUs has surged.

### Getting Started with CUDA: A Simple Example

Let's dive into data parallel programming using CUDA. We'll start with a simple example: vector addition.

Below is the traditional, sequential way of adding two vectors using simple C code:

```c
void add_cpu(float *x_h, float *y_h, float *res_h, int n) {
    for (int i = 0; i < n; ++i) {
        res_h[i] = x_h[i] + y_h[i];
    }
}
```

#### System Organization and Naming Conventions

In the context of CUDA and GPU programming, the CPU is referred to as the **host**, and its memory is called **host memory**. The GPU is called the **device**, and its memory is known as **global memory**. CPUs and GPUs have separate memories and cannot directly access each other's memory without transferring data via interconnects like PCIe or NVLink.

A typical process follows these steps:
1. Allocate memory on the GPU.
2. Copy data from host memory to GPU memory.
3. Perform computation on the GPU.
4. Copy data from GPU memory back to host memory.
5. Deallocate GPU memory.

Let's go through these steps in detail:

### Step 1: Allocate GPU Memory

We need to allocate memory on the GPU for our vectors. CUDA provides a function `cudaMalloc` to allocate memory on the device.

```c
cudaMalloc((void**)&x_d, N * sizeof(float));
cudaMalloc((void**)&y_d, N * sizeof(float));
cudaMalloc((void**)&z_d, N * sizeof(float));
```

### Step 2: Copy Data to the GPU

Next, we need to copy the input vectors from host memory to GPU memory.This step is essential for ensuring that the data needed for computation is available on the GPU and that the results are brought back to the CPU for further processing or analysis. 
In CUDA programming, we use the `cudaMemcpy` function to handle memory transfers. This function takes several parameters:

- `dst`: The destination memory address where the data will be copied.
- `src`: The source memory address from which the data will be copied.
- `count`: The size in bytes of the data to be copied.
- `kind`: The type of transfer, which can be one of the following:
  - `cudaMemcpyHostToHost`: Copy data from host to host.
  - `cudaMemcpyHostToDevice`: Copy data from host to device.
  - `cudaMemcpyDeviceToHost`: Copy data from device to host.
  - `cudaMemcpyDeviceToDevice`: Copy data from device to device.

Here's how we incorporate memory copying into our vector addition function:


```c
void vecadd_gpu(float *x, float *y, float *z, int N) {
    int size = N * sizeof(float);
    float *x_d, *y_d, *z_d; // _d conventionally represents pointers to device memory (GPU)

    // Allocate Memory on the GPU
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    // Copy Data from Host to Device
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Run the GPU Code

    // Copy Results from Device to Host
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate Memory on the GPU
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}
```

In this function, we first allocate memory on the GPU using `cudaMalloc` for arrays `x_d`, `y_d`, and `z_d`, which will store the input and output data for our vector addition operation.

Next, we use `cudaMemcpy` to copy the input arrays `x` and `y` from the CPU to the GPU.

After performing the computation on the GPU (which we'll cover in Step 3), we copy the result array `z_d` from the GPU back to the CPU.

Finally, we free the allocated memory on the GPU using `cudaFree` to release resources once they are no longer needed.

By efficiently managing memory transfers between the host and the device, we ensure smooth execution of GPU-accelerated computations, optimizing performance and resource utilization.

### Step 3: Perform Computation on the GPU


Now, let's dive into the heart of our task: performing vector addition in parallel on the GPU. To achieve this, we need to understand how to parallelize the addition effectively.

Firstly, let's explore how threads are organized in GPUs. Threads in a GPU are arranged in an array known as a grid. Within this grid, threads are grouped into blocks. Threads within the same block can collaborate in ways that threads in different blocks cannot.

To launch a grid of threads, we need to specify how many blocks we want and how many threads per block. All threads within the same grid execute the same function, which we call a kernel. We launch a grid by calling this kernel and configuring it with appropriate grid and block sizes.

Here's a snippet of code showcasing how we perform vector addition on the GPU:

```c
void vecadd_gpu(float *x, float *y, float *z, int N) {
    int size = N * sizeof(float);
    float *x_d, *y_d, *z_d; // _d conventionally represents pointers to device memory (GPU)

    // Allocate Memory
    cudaMalloc((void**)&x_d, N*sizeof(float));
    cudaMalloc((void**)&y_d, N*sizeof(float));
    cudaMalloc((void**)&z_d, N*sizeof(float));

    // Copy to the GPU
    cudaMemcpy(x_d, x, N*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, N*sizeof(float), cudaMemcpyHostToDevice);

    // Run the GPU Code
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = N / 512;
    vecadd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d, y_d, z_d, N);

    // Copy from the GPU to CPU
    cudaMemcpy(z, z_d, N*sizeof(float), cudaMemcpyDeviceToHost);

    // Deallocate Memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}
```

In this function, `vecadd_kernel<<< numBlocks, numThreadsPerBlock >>>(x_d, y_d, z_d, N)` creates a grid on the GPU with `numBlocks` blocks and `numThreadsPerBlock` threads per block. Each block calls the `vecadd_kernel` function.

Now, let's discuss the implementation of the kernel itself. A kernel is similar to a C/C++ function but is preceded by the `__global__` keyword to indicate that it is a GPU kernel. It utilizes special keywords to distinguish different threads from each other:

- `gridDim.x` tells us the number of blocks in the grid.
- `blockIdx.x` tells us the position of the block in the grid.
- `blockDim.x` gives the number of threads in a block.
- `threadIdx.x` gives the position of the thread in the block.

To parallelize vector addition, we map each thread to an element in the array. Therefore, we need to calculate the global index or index from the beginning of the array. This is done using the formula `unsigned int i = blockIdx.x * blockDim.x + threadIdx.x`, giving us the global index from the start of the grid.

With this index, each thread accesses the corresponding element in the arrays `x` and `y` and computes `z[i] = x[i] + y[i]` concurrently for all threads, thereby achieving parallel vector addition.

```c
__global__ void vecadd_kernel(float* x, float* y, float* z, int N){
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if(i < N){
        z[i] = x[i] + y[i];
    }
}
```

In summary, by harnessing the parallel processing capabilities of the GPU and carefully orchestrating thread organization, we efficiently perform vector addition, unlocking significant computational power for tasks requiring massive data processing.

### Step 4: Copy the Result Back to the Host

Once the computation is complete, we copy the result vector from GPU memory back to host memory.

```c
cudaMemcpy(z, z_d, N * sizeof(float), cudaMemcpyDeviceToHost);
```

### Step 5: Deallocate GPU Memory

Finally, we free the memory allocated on the GPU.

```c
cudaFree(x_d);
cudaFree(y_d);
cudaFree(z_d);
```

### Full Example with Main Function

Below is the full example, including the main function to initialize data and run the vector addition on the GPU.

```c
#include <cuda_runtime.h>
#include <stdlib.h>
#include <stdio.h>

__global__ void vecadd_kernel(float* x, float* y, float* z, int N) {
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < N) {
        z[i] = x[i] + y[i];
    }
}

void vecadd_gpu(float* x, float* y, float* z, int N) {
    int size = N * sizeof(float);
    float *x_d, *y_d, *z_d;

    // Allocate GPU memory
    cudaMalloc((void**)&x_d, size);
    cudaMalloc((void**)&y_d, size);
    cudaMalloc((void**)&z_d, size);

    // Copy data to the GPU
    cudaMemcpy(x_d, x, size, cudaMemcpyHostToDevice);
    cudaMemcpy(y_d, y, size, cudaMemcpyHostToDevice);

    // Perform computation on the GPU
    const unsigned int numThreadsPerBlock = 512;
    const unsigned int numBlocks = (N + numThreadsPerBlock - 1) / numThreadsPerBlock;
    vecadd_kernel<<<numBlocks, numThreadsPerBlock>>>(x_d, y_d, z_d, N);
    cudaDeviceSynchronize();

    // Copy the result back to the host
    cudaMemcpy(z, z_d, size, cudaMemcpyDeviceToHost);

    // Deallocate GPU memory
    cudaFree(x_d);
    cudaFree(y_d);
    cudaFree(z_d);
}

int main(int argc, char** argv) {
    unsigned int N = (argc > 1) ? atoi(argv[1]) : (1 << 25);
    float* x = (float*) malloc(N * sizeof(float));
    float* y = (float*) malloc(N * sizeof(float));
    float* z = (float*) malloc(N * sizeof(float));

    for (unsigned int i = 0; i < N; ++i) {
        x[i] = rand() / (float)RAND_MAX;
        y[i] = rand() / (float)RAND_MAX;
    }

    // Perform vector addition on the GPU
    vecadd_gpu(x, y, z, N);

    // Free the memory
    free(x);
    free(y);
    free(z);

    return 0;
}
```

### Compilation and Execution

To compile and run the CUDA program, use the NVIDIA C Compiler (NVCC):

```sh
nvcc vecadd.cu -o vecadd
./vecadd
```

NVCC compiles the host code using the standard C/C++ compilers and the device code into PTX files, which are then compiled into object files for execution on NVIDIA GPUs.

---

This blog introduces you to the basics of CUDA programming and sets the stage for more advanced topics in parallel programming. Stay tuned for more insights and examples...