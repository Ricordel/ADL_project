#ifndef __CUDA_MACROS_H__
#define __CUDA_MACROS_H__


/* Define a function as both on host and device */
#ifdef __CUDA
#define CUDA_BOTH __host__ __device__
#else
#define CUDA_BOTH
#endif


#endif /* end of include guard: __CUDA_MACROS_H__ */
