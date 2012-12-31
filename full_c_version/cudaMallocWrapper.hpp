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
                switch (kind) {
                        case cudaMemcpyHostToDevice:
                                throw CudaMemoryException("Memcpy from host to device failed");
                        case cudaMemcpyDeviceToHost:
                                throw CudaMemoryException("Memcpy from device to host failed");
                        case cudaMemcpyDeviceToDevice:
                                throw CudaMemoryException("Memcpy from device to device failed");
                        case cudaMemcpyHostToHost:
                                throw CudaMemoryException("Memcpy from host to host failed");
                        default:
                                throw CudaMemoryException("Memcpy: unknown kind");
                }
        }
}



#endif /* end of include guard: __CUDAMALLOCWRAPPER_H__ */
