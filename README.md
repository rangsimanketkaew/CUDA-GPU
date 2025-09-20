# CUDA-GPU Fortran : Friendly Resources :-)

CUDA GPU resources for beginners with focus on scientific computing

### Terminology

- **The Host:** The CPU and its memory is called the host. 
- **Device:** GPU and its memory is called the device. 
- **Communication** CPU and GPU are connected with PCI bus which have much slower data bandwidth compared to the each processing unit and their memory and moving data between them is time consuming. Thus, frequent exchange of data between the two memory is highly discourage.
  
- **Kernels:** A function that is executed on the GPU.
- Threads Hierarchy
  - **Thread:** At the lowest level of CUDA threads hierarchy are the individual threads. Each thread execute the kernel on a single piece of data and each gets mapped to a single CUDA core.
  - **Blocks:** A group of thread.
  - **Grid:** The collection of blocks that gets mapped on the entire GPU 
  - Blocks and Grids can be 1D, 2D or 3D and the program has to written in such way to control over multidimensional Blocks/Grids.

- **Flow of Program:** The main code execution is started on the CPU (the host). Separate memory are allocated for host and device to hold the data for each of their computation. When needed the data is copied to the device from host and back. Host can launch a group of kernels on the device. When the kernels are launched, the host does not wait for the kernels execution to finish and can proceed with its own flow. The memory copy between the host and  device can be synchronous or asynchronous. Usually they are done in synchronous manner. The assignment operator (`=`) in CUDA Fortran is overloaded with synchronous memory copy i.e. the copy operation will wait for the kernels to finish their execution

---

1. **CUDA Fortran**
CUDA kernels, which is similar to OpenACC parallel loops, which allows parallel code on the GPU without writing explicit kernels. This method does not work for everything, but it works really well when it is the right tool.

1. **OpenMP**
OpenMP is a directive-based model. 
See [OpenMP for GPU: an introduction](http://www.idris.fr/media/formations/openacc/openmp_gpu_idris_c.pdf)

1. **OpenACC**
OpenACC is also a directive-based model and supported by NVIDIA, Cray Fortran and GCC.
[OpenACC Getting Started Guide](https://docs.nvidia.com/hpc-sdk/compilers/openacc-gs/#fortran-derived-types-in-openacc)

1. **StdPar (Fortran Standard Parallel)** 
StdPar allows the use of `DO CONCURRENT` on GPUs, along with many data-parallel Fortran intrinsics.

https://www.jefflarkin.com/posts/using-fortran-standard-parallel-programming-for-gpu-acceleration

---

### CUDA Fortran Installation:

1. Install Nvidia drivers for your system. https://www.nvidia.com/en-us/drivers/
2. Install Nvidia CUDA toolkit. https://developer.nvidia.com/cuda-downloads
3. Install Nvidia HPC SDK. https://developer.nvidia.com/hpc-sdk-downloads. 
    ```bash
    wget https://developer.download.nvidia.com/hpc-sdk/25.7/nvhpc_2025_257_Linux_x86_64_cuda_12.9.tar.gz
    tar xpzf nvhpc_2025_257_Linux_x86_64_cuda_12.9.tar.gz
    nvhpc_2025_257_Linux_x86_64_cuda_12.9/install
    ```
    The installation path is `/opt/nvidia/hpc_sdk/Linux_x86_64/*/compilers/bin` (sudo is needed).

--- 

### Compilation and Execution

Earlier the CUDA Fortran compiler was developed by PGI. From 2020 the PGI compiler tools was replaced with the Nvidia HPC Toolkit. You can use compilers like `nvc`, `nvc++` and `nvfortan` to compile `C`, `C++` and `Fortran`, respectively.

- CUDA Fortran codes have suffixed `.cuf`.

- Compile CUDA Fortran with `nvfortran` and just run the executable.
    ```bash
    nvfortran test_code.cuf -o test
    ./test
    ```

---

### CUDA Fortran Code:

**SAXPY** (Scalar A*X Plus Y) program is like *Hello World* for CUDA programming to how to go from CPU to GPU code.

- The serial CPU code [saxpy-gpu.F90](saxpy-cpu.F90)
- The same code as above but offloaded to GPU [saxpy-gpu.cuf]()

### Profiling:

Profiling can be done with the `nvprof` utility. 

<details>	

  <summary>A profiling on the `saxpy-gpu.cuf` code</summary>

```bash
$ nvfortran saxpy-gpu.cuf -o saxpy-gpu
$ nvprof ./saxpy-gpu
==30449== NVPROF is profiling process 30449, command: ./saxpy-gpu
 Max error:     0.000000
==30449== Profiling application: ./saxpy-gpu
==30449== Warning: 7 API trace records have same start and end timestamps.
This can happen because of short execution duration of CUDA APIs and low timer resolution on the underlying operating system.
==30449== Profiling result:
            Type  Time(%)      Time     Calls       Avg       Min       Max  Name
 GPU activities:   62.34%  30.400us         4  7.6000us     896ns  14.400us  [CUDA memcpy HtoD]
                   27.17%  13.248us         1  13.248us  13.248us  13.248us  [CUDA memcpy DtoH]
                   10.50%  5.1200us         1  5.1200us  5.1200us  5.1200us  mathops_saxpy_
      API calls:   98.58%  235.08ms         4  58.770ms  2.5000us  235.06ms  cudaMalloc
                    0.67%  1.5930ms         1  1.5930ms  1.5930ms  1.5930ms  cudaLaunchKernel
                    0.41%  977.80us         5  195.56us  52.400us  501.30us  cudaMemcpy
                    0.20%  474.30us       114  4.1600us       0ns  454.60us  cuDeviceGetAttribute
                    0.13%  321.00us         1  321.00us  321.00us  321.00us  cuDeviceGetPCIBusId
                    0.00%  10.300us         2  5.1500us  2.6000us  7.7000us  cudaFree
                    0.00%  6.2000us         2  3.1000us     200ns  6.0000us  cuDeviceGet
                    0.00%  1.3000us         1  1.3000us  1.3000us  1.3000us  cuDeviceGetName
                    0.00%  1.1000us         3     366ns     200ns     500ns  cuDeviceGetCount
                    0.00%  1.0000us         1  1.0000us  1.0000us  1.0000us  cuDeviceTotalMem
                    0.00%     500ns         1     500ns     500ns     500ns  cuDeviceGetUuid
                    0.00%     200ns         1     200ns     200ns     200ns  cuModuleGetLoadingMode
```

</details>

---

## Source Codes examples

#### Loop parallelization:

`!$cuf kernel do` directive can be used to simplify parallelizing loops. These directives instruct the compiler to generate kernels from a region of host code consisting of tightly nested loops. Essentially, kernel loop directives allow us to inline kernels in host code.  
1. [Loop parallelization](./cufKernel.cuf)
2. [Loop parallelization 2](./cufILP.cuf)
3. [Nested Loop parallelization](./cufKernel2D.cuf)
4. [Reduction operation](./cufReduction.cuf)

---

### References

1. [An Easy Introduction to CUDA Fortran
](https://developer.nvidia.com/blog/easy-introduction-cuda-fortran/)
2. [Accelerating Fortran DO CONCURRENT with GPUs and the NVIDIA HPC SDK](https://developer.nvidia.com/blog/)accelerating-fortran-do-concurrent-with-gpus-and-the-nvidia-hpc-sdk/
3. https://docs.nvidia.com/hpc-sdk/compilers/cuda-fortran-prog-guide/
4. [Using Fortran Standard Parallel Programming for GPU Acceleration - Jeff Larkin](https://www.jefflarkin.com/posts/using-fortran-standard-parallel-programming-for-gpu-acceleration)
5. Coding or GPUs Using Standard Fortran - Jeff Larkin ([Slide](https://www.olcf.ornl.gov/wp-content/uploads/20220513_OLCF_Fortran.pdf) | [Recording](https://vimeo.com/manage/videos/711784748))
6. Simple Fortran program test set https://github.com/ParRes/Kernels/tree/main/FORTRAN 

### Tutorial videos

1. Programming GPUs with Fortran https://www.youtube.com/watch?v=COjvWNpxnxc
2. Offloading to GPU using Fortran https://www.youtube.com/watch?v=K_-Tr3FZuDs
