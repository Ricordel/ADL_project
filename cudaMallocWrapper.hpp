#ifndef __CUDAMALLOCWRAPPER_H__
#define __CUDAMALLOCWRAPPER_H__


#include <stdexcept>
#include <cuda.h>


class CudaMemoryException : public std::runtime_error
{
        public:
                CudaMemoryException(std::string msg) : runtime_error(msg) {}
};


template<typename T>
class CudaMallocWrapper
{
        public:
                CudaMallocWrapper(size_t nmemb)
                {
                        cudaError_t ret;
                        ret = cudaMalloc((void **) &mem, nmemb * sizeof(T));
                        if (ret != cudaSuccess) {
                                throw CudaMemoryException("No more memory on device");
                        }
                }

                ~CudaMallocWrapper()
                {
                        cudaFree(mem);
                }

                T *mem;
};


/* A function that memcpys and throw exceptions if failed */
template <typename T>
void cudaMemcpyWrapped(void *to, const void *from, size_t nmemb, enum cudaMemcpyKind kind)
{
        cudaError_t ret;
        ret = cudaMemcpy(to, from, nmemb * sizeof(T), kind);
        if (ret != cudaSuccess) {
                throw CudaMemoryException("cudaMemcpy failed with message: " +
                                           std::string(cudaGetErrorString(ret)));
        }
}



#endif /* end of include guard: __CUDAMALLOCWRAPPER_H__ */
