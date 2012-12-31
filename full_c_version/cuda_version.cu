#include <stdexcept>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <stdint.h>

#include "cudaMallocWrapper.hpp"
#include "dbg.h"


#define FUNCS_PER_KERNEL 8192



// Some static variables for the CUDA version. That's not beautiful, I know.
struct GPUProps {
    bool initialized;
    int maxThreadsPerBlock;
    int maxConcurrentBlocks;
    int maxConcurrentThreads;
};

static struct GPUProps GPUProps = {false, 0, 0, 0};

void getGPUProperties()
{
    if (GPUProps.initialized) {
        return;
    }

    cudaError_t ret;
    struct cudaDeviceProp deviceProp;

    ret = cudaGetDeviceProperties(&deviceProp, 0);
    if (ret != cudaSuccess) {
        throw std::runtime_error("Failed to get device properties: " + std::string(cudaGetErrorString(ret)));
    }

    GPUProps.maxThreadsPerBlock = deviceProp.maxThreadsDim[0];
    GPUProps.maxConcurrentBlocks = deviceProp.maxGridSize[0];
    GPUProps.maxConcurrentThreads = GPUProps.maxThreadsPerBlock * GPUProps.maxConcurrentBlocks;

    GPUProps.initialized = true;
}






/**********************************************************************************************
 ********************************* Form x0 + a + b + cd ***************************************
 **********************************************************************************************/


struct Function_0_a_b_cd {
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t d;
    uint8_t nVariables;
};


bool smaller_or_equal_0_a_b_cd(struct Function_0_a_b_cd& one, struct Function_0_a_b_cd& other)
{
    if (one.a < other.a)
        return true;

    if (one.a == other.a && one.b < other.b)
        return true;

    if (one.a == other.a && one.b == other.b && one.c < other.c)
        return true;

    if (one.a == other.a && one.b == other.b && one.c == other.c && one.d <= other.d)
        return true;

    return false;
}

bool canonical_0_a_b_cd(struct Function_0_a_b_cd& func)
{
    int32_t ar = func.nVariables - func.a;
    int32_t br = func.nVariables - func.b;
    int32_t cr = func.nVariables - func.c;
    int32_t dr = func.nVariables - func.d;

    struct Function_0_a_b_cd other = {br, ar, dr, cr, func.nVariables};
    return smaller_or_equal_0_a_b_cd(func, other);
}


#define bit(nBit, val) (((val) >> (nBit)) & 1)


__global__ void kernel_0_a_b_cd(struct Function_0_a_b_cd *d_funcArray, uint32_t *d_funcLength,
                                uint32_t nQueued, uint32_t maxPossibleLength)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        // Copy things into registers
        uint8_t a = d_funcArray[me].a;
        uint8_t b = d_funcArray[me].b;
        uint8_t c = d_funcArray[me].c;
        uint8_t d = d_funcArray[me].d;
        uint8_t nVariables = d_funcArray[me].nVariables;

        uint32_t curVal = 1;
        uint32_t length = 0;
        uint32_t newBit = 0;

        do {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                    (bit(c, curVal) & bit(d, curVal));
            curVal = (curVal >> 1) | (newBit << (nVariables - 1));

            length++;
        } while (curVal != 1);

        d_funcLength[me] = length;
    }
}







void sendAndReport_0_a_b_cd(struct Function_0_a_b_cd *h_funcArray, uint32_t *h_funcLength,
                            struct Function_0_a_b_cd *d_funcArray, uint32_t *d_funcLength,
                            uint32_t maxPossibleLength, uint32_t nQueued)
{
    uint32_t nBlocks, nThreadsPerBlock;

    /*log_info("Sending %u functions to device\n", nQueued);*/

    cudaMemcpyWrapped<Function_0_a_b_cd>(d_funcArray, h_funcArray, nQueued, cudaMemcpyHostToDevice);

    nBlocks = nQueued / GPUProps.maxThreadsPerBlock + (nQueued % GPUProps.maxThreadsPerBlock == 0 ? 0 : 1);

    nThreadsPerBlock = (nBlocks == 1) ? nQueued : GPUProps.maxThreadsPerBlock;

    /*log_info("Launching kernel");*/

    kernel_0_a_b_cd <<< nBlocks, nThreadsPerBlock >>> (d_funcArray, d_funcLength, nQueued, maxPossibleLength);

    cudaMemcpyWrapped<uint32_t>(h_funcLength, d_funcLength, nQueued, cudaMemcpyDeviceToHost);

    /*log_info("Kernel returned");*/

    for (int i = 0; i < nQueued; i++) {
        std::cout << h_funcLength[i] << std::endl;
#if 0
        if (h_funcLength[i] == maxPossibleLength) {
            std::cout << "0," << (uint32_t)h_funcArray[i].a << "," << (uint32_t)h_funcArray[i].b
                      << ",(" << (uint32_t)h_funcArray[i].c << "," << (uint32_t)h_funcArray[i].d << ")"
                      << std::endl;
        }
#endif
    }

    std::cout << maxPossibleLength << std::endl;
}




/************************************ Functions generation **************************************/
void report_0_a_b_cd(uint32_t nVariables)
{
    getGPUProperties();

    //XXX Carefull with nVariables 32.
    uint32_t maxPossibleLength = (1 << nVariables) - 1;

    std::vector<Function_0_a_b_cd> h_funcArray;
    std::vector<uint32_t> h_funcLength(FUNCS_PER_KERNEL);

    /* Allocate memory for arrays on device */
    CudaMallocWrapper<Function_0_a_b_cd> d_funcArray(FUNCS_PER_KERNEL);
    CudaMallocWrapper<uint32_t> d_funcLength(FUNCS_PER_KERNEL);


    /* Generate the functions */
    for (int32_t a = 1; a <= (nVariables + 1) / 2; a++) {
        for (int32_t b = a + 1; b <= nVariables - 1; b++) {

            for (int32_t c = 1; c <= nVariables - 2; c++) {
                for (int32_t d = c + 1; d <= nVariables - 1; d++) {

                    // Keep the function for later evaluation
                    struct Function_0_a_b_cd func = {a, b, c, d, nVariables};
                    if (!canonical_0_a_b_cd(func)) {
                        continue;
                    }

                    h_funcArray.push_back(func);

                    if (h_funcArray.size() == FUNCS_PER_KERNEL) {
                        // Warning with nvcc for taking adress of a stack variable, but no problem here
                        sendAndReport_0_a_b_cd((struct Function_0_a_b_cd*)&h_funcArray[0], (uint32_t*)&h_funcLength[0],
                                               (struct Function_0_a_b_cd*)d_funcArray.mem, (uint32_t*)d_funcLength.mem,
                                                maxPossibleLength, h_funcArray.size());

                        h_funcArray.clear();
                    }
                }
            }
        }
    }

    if (h_funcArray.size() != 0) {
        sendAndReport_0_a_b_cd((struct Function_0_a_b_cd*)&h_funcArray[0], (uint32_t *)&h_funcLength[0],
                               (struct Function_0_a_b_cd*)d_funcArray.mem, (uint32_t *)d_funcLength.mem,
                                maxPossibleLength, h_funcArray.size());
    }
}







#if 0






    template <class FunctionType>
__global__ void kernel(FunctionType *d_funcArray, bool *d_isMaxLength, uint32_t nFunctions, uint32_t maxPossibleLength)
{
    // Get my position in the grid
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nFunctions) {
        if (d_funcArray[me].getCycleLength_device() == maxPossibleLength) {
            d_isMaxLength[me] = true;
        } else {
            d_isMaxLength[me] = false;
        }
    }

}







    template <class FunctionType>
static void sendAndReport(FunctionType *h_funcArray, FunctionType *d_funcArray,
        bool *h_isMaxLength, bool *d_isMaxLength, uint32_t enqueued, uint32_t maxPossibleLength)
{
    cudaError_t ret;
    uint32_t nBlocks, nThreadsPerBlock;

    std::cerr << "Sending " << enqueued << " functions to device" << std::endl;

    ret = cudaMemcpy(d_funcArray, h_funcArray, enqueued * sizeof(FunctionType), cudaMemcpyHostToDevice);
    if (ret != cudaSuccess) {
        throw std::runtime_error("Failed to memcpy to device: " + std::string(cudaGetErrorString(ret)));
    }

    nBlocks = enqueued / GPUProps.maxThreadsPerBlock + (enqueued % GPUProps.maxThreadsPerBlock == 0 ? 0 : 1);

    nThreadsPerBlock = (nBlocks == 1) ? enqueued : GPUProps.maxThreadsPerBlock;

    std::cerr << "Launching kernel" << std::endl;
    // Launch kernel
    kernel<FunctionType> <<< nBlocks, nThreadsPerBlock >>> (d_funcArray, d_isMaxLength, enqueued, maxPossibleLength);

    ret = cudaMemcpy(h_isMaxLength, d_isMaxLength, enqueued * sizeof(bool), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
        throw std::runtime_error("Kernel execution failed: " + std::string(cudaGetErrorString(ret)));
    }

    std::cerr << "Kernel has returned successfully" << std::endl;

    for (int i = 0; i < enqueued; i++) {
        if (h_isMaxLength[i]) {
            std::cout << h_funcArray[i].toString() << std::endl;
        }
    }
}




/***********************************************************************
 ********************* For x0 + xa + xb + xc.xd ************************
 ***********************************************************************/

    FuncGenerator_0_a_b_cd::FuncGenerator_0_a_b_cd(uint32_t nVariables)
: m_nVariables(nVariables), m_maxPossibleLength((1 << nVariables) - 1)
{}

FuncGenerator_0_a_b_cd::~FuncGenerator_0_a_b_cd() {}







/***********************************************************************
 ******************** For x0 + xa + xb.xc + xd.xe **********************
 ***********************************************************************/

    FuncGenerator_0_a_bc_de::FuncGenerator_0_a_bc_de(uint32_t nVariables)
: m_nVariables(nVariables), m_maxPossibleLength((1 << m_nVariables) - 1)
{}

FuncGenerator_0_a_bc_de::~FuncGenerator_0_a_bc_de() {}




void FuncGenerator_0_a_bc_de::reportMaxFunctions()
{
    getGPUProperties();

    // The lexicographical order is difficult to handle in the generation
    // for b,c and d,e. So this will be handled in isCanonicalForm()

    // d can start from b, because if d < b, then a commutatively equivalent function
    // will have been tested (as b.c and d.e can commute around +), and that variant
    // would be smaller by lexicographical order.

    // We don't want b = c AND d = e either, which gives us kind of a "degenerated" function.
    // This is also handled in isCanonicalForm()

    cudaError_t ret;

    std::vector<Function_0_a_bc_de> h_funcVector;
    Function_0_a_bc_de *h_funcArray;
    Function_0_a_bc_de *d_funcArray;
    bool *d_isMaxLength;
    bool *h_isMaxLength;

    h_isMaxLength = new bool[GPUProps.actualConcurrentThreads];
    h_funcArray = new Function_0_a_bc_de[GPUProps.actualConcurrentThreads];

    ret = cudaMalloc((void **) &d_isMaxLength, GPUProps.actualConcurrentThreads * sizeof(bool));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for bools");
    }

    ret = cudaMalloc((void **) &d_funcArray, GPUProps.actualConcurrentThreads * sizeof(Function_0_a_bc_de));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for Function_0_a_bc_de");
    }

    uint32_t enqueued = 0;

    for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {

        for (int32_t b = 1; b <= m_nVariables - 2; b++) {
            for (int32_t c = b + 1; c <= m_nVariables - 1; c++) {

                for (int32_t d = b; d <= m_nVariables - 2; d++) {
                    for (int32_t e = d + 1; e <= m_nVariables - 1; e++) {

                        // Keep the function for later evaluation
                        Function_0_a_bc_de func(a, b, c, d, e, m_nVariables);
                        if (!func.isCanonicalForm()) {
                            continue;
                        }

                        h_funcArray[enqueued++] = func;

                        if (enqueued == GPUProps.actualConcurrentThreads) {
                            sendAndReport<Function_0_a_bc_de>(h_funcArray, d_funcArray, h_isMaxLength,
                                    d_isMaxLength, enqueued, m_maxPossibleLength);
                            enqueued = 0;
                        }
                    }
                }
            }
        }
    }

    if (enqueued != 0) {
        sendAndReport<Function_0_a_bc_de>(h_funcArray, d_funcArray, h_isMaxLength,
                d_isMaxLength, enqueued, m_maxPossibleLength);
    }

    cudaFree(d_funcArray);
    cudaFree(d_isMaxLength);
    delete [] h_isMaxLength;
    delete [] h_funcArray;
}




/***********************************************************************
 **************** For x0 + xa + xb + xc + xd + xe.xf *******************
 ***********************************************************************/

    FuncGenerator_0_a_b_c_d_ef::FuncGenerator_0_a_b_c_d_ef(uint32_t nVariables)
: m_nVariables(nVariables), m_maxPossibleLength((1 << m_nVariables) - 1)
{}


FuncGenerator_0_a_b_c_d_ef::~FuncGenerator_0_a_b_c_d_ef()
{}


void FuncGenerator_0_a_b_c_d_ef::reportMaxFunctions()
{
    getGPUProperties();

    cudaError_t ret;

    Function_0_a_b_c_d_ef *h_funcArray;
    Function_0_a_b_c_d_ef *d_funcArray;
    bool *d_isMaxLength;
    bool *h_isMaxLength;

    h_isMaxLength = new bool[GPUProps.actualConcurrentThreads];
    h_funcArray = new Function_0_a_b_c_d_ef[GPUProps.actualConcurrentThreads];

    ret = cudaMalloc((void **) &d_isMaxLength, GPUProps.actualConcurrentThreads * sizeof(bool));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for bools");
    }

    ret = cudaMalloc((void **) &d_funcArray, GPUProps.actualConcurrentThreads * sizeof(Function_0_a_b_c_d_ef));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for Function_0_a_b_c_d_ef");
    }

    uint32_t enqueued = 0;

    for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
        for (int32_t b = a + 1; b <= m_nVariables - 3; b++) { /* -3 to leave room for c and d */
            for (int32_t c = b + 1; c <= m_nVariables - 2; c++) { /* -2 to leave room for d */
                for (int32_t d = c + 1; d <= m_nVariables - 1; d++) {

                    for (int32_t e = 1; e <= m_nVariables - 2; e++) {
                        for (int32_t f = e + 1; f <= m_nVariables - 1; f++) {

                            // Keep the function for later evaluation
                            Function_0_a_b_c_d_ef func(a, b, c, d, e, f, m_nVariables);
                            if (!func.isCanonicalForm()) {
                                continue;
                            }

                            h_funcArray[enqueued++] = func;

                            if (enqueued == GPUProps.actualConcurrentThreads) {
                                sendAndReport<Function_0_a_b_c_d_ef>(h_funcArray, d_funcArray, h_isMaxLength,
                                        d_isMaxLength, enqueued, m_maxPossibleLength);
                                enqueued = 0;
                            }

                        }
                    }
                }
            }
        }
    }

    if (enqueued != 0) {
        sendAndReport<Function_0_a_b_c_d_ef>(h_funcArray, d_funcArray, h_isMaxLength,
                d_isMaxLength, enqueued, m_maxPossibleLength);
    }

    cudaFree(d_funcArray);
    cudaFree(d_isMaxLength);
    delete [] h_isMaxLength;
    delete [] h_funcArray;
}






/***********************************************************************
 **************** For x0 + xa + xb + xc.xd.xe *******************
 ***********************************************************************/

    FuncGenerator_0_a_b_cde::FuncGenerator_0_a_b_cde(uint32_t nVariables)
: m_nVariables(nVariables), m_maxPossibleLength((1 << m_nVariables) - 1)
{}


FuncGenerator_0_a_b_cde::~FuncGenerator_0_a_b_cde()
{}


void FuncGenerator_0_a_b_cde::reportMaxFunctions()
{
    getGPUProperties();

    cudaError_t ret;

    Function_0_a_b_cde *h_funcArray;
    Function_0_a_b_cde *d_funcArray;
    bool *d_isMaxLength;
    bool *h_isMaxLength;

    h_isMaxLength = new bool[GPUProps.actualConcurrentThreads];
    h_funcArray = new Function_0_a_b_cde[GPUProps.actualConcurrentThreads];

    ret = cudaMalloc((void **) &d_isMaxLength, GPUProps.actualConcurrentThreads * sizeof(bool));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for bools");
    }

    ret = cudaMalloc((void **) &d_funcArray, GPUProps.actualConcurrentThreads * sizeof(Function_0_a_b_cde));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for Function_0_a_b_cde");
    }

    uint32_t enqueued = 0;

    for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
        for (int32_t b = a + 1; b <= m_nVariables - 1; b++) {


            for (int32_t c = 1; c <= m_nVariables - 3; c++) { /* -3 to leave room for d and e */
                for (int32_t d = c + 1; d <= m_nVariables - 2; d++) {
                    for (int32_t e = d + 1; e <= m_nVariables - 1; e++) {

                        // Keep the function for later evaluation
                        Function_0_a_b_cde func(a, b, c, d, e, m_nVariables);
                        if (!func.isCanonicalForm()) {
                            continue;
                        }

                        h_funcArray[enqueued++] = func;

                        if (enqueued == GPUProps.actualConcurrentThreads) {
                            sendAndReport<Function_0_a_b_cde>(h_funcArray, d_funcArray, h_isMaxLength,
                                    d_isMaxLength, enqueued, m_maxPossibleLength);
                            enqueued = 0;
                        }

                    }
                }
            }
        }
    }

    if (enqueued != 0) {
        sendAndReport<Function_0_a_b_cde>(h_funcArray, d_funcArray, h_isMaxLength,
                d_isMaxLength, enqueued, m_maxPossibleLength);
    }

    cudaFree(d_funcArray);
    cudaFree(d_isMaxLength);
    delete [] h_isMaxLength;
    delete [] h_funcArray;
}
#endif


int main(int argc, char *argv[])
{
    report_0_a_b_cd(atoi(argv[1]));
    return 0;
}
