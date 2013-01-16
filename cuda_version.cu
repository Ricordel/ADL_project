#include <stdexcept>
#include <vector>
#include <iostream>

#include <cuda.h>
#include <stdint.h>

#include "cudaMallocWrapper.hpp"
#include "dbg.h"


#define FUNCS_PER_KERNEL (1 << 16)



///////////////////////////////// Some CUDA administrivia ///////////////////////////////////////

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

    /*GPUProps.maxThreadsPerBlock = deviceProp.maxThreadsDim[0];*/
    //XXX tentative
    GPUProps.maxThreadsPerBlock = 256;
    GPUProps.maxConcurrentBlocks = deviceProp.maxGridSize[0];
    GPUProps.maxConcurrentThreads = GPUProps.maxThreadsPerBlock * GPUProps.maxConcurrentBlocks;

    GPUProps.initialized = true;
}


#define bit(nBit, val) (((val) >> (nBit)) & 1)


//////////////////////////////////////////////////////////////////////////////////////////////
//////////////////// General template declarations and generic definitions ///////////////////
//////////////////////////////////////////////////////////////////////////////////////////////

////// The following templates will need to be specialized for each type of functions ///////

template <typename FuncType>
bool smaller_or_equal(FuncType& one, FuncType& other);

template <typename FuncType>
bool canonical(FuncType& func);


template <typename FuncType>
__global__ void kernel_to_the_end(FuncType *d_funcArray,
                                  uint32_t nQueued, uint32_t maxPossibleLength);

template <typename FuncType>
__global__ void kernel_filter(FuncType *d_funcArray,
                              uint32_t nQueued, uint32_t filterValue);


template <typename FuncType> void print_func(FuncType& func);

template <typename FuncType>
void generate_functions(uint32_t nVariables, std::vector<FuncType>& funcArray);

template <typename FuncType> void report(uint32_t nVariables);



////// The following templates are generic ///////

template <typename FuncType>
void send_and_filter_to(std::vector<FuncType>& h_funcArray,
                     std::vector<FuncType>& h_remainingFuncArray,
                     CudaMallocWrapper<FuncType>& d_funcArray,
                     uint32_t filterValue)
{
    uint32_t nBlocks, nThreadsPerBlock, nQueued;
    nQueued = h_funcArray.size();

    cudaMemcpyWrapped<FuncType>(d_funcArray.mem, &h_funcArray[0], nQueued, cudaMemcpyHostToDevice);

    nBlocks = nQueued / GPUProps.maxThreadsPerBlock + (nQueued % GPUProps.maxThreadsPerBlock == 0 ? 0 : 1);
    nThreadsPerBlock = (nBlocks == 1) ? nQueued : GPUProps.maxThreadsPerBlock;

    kernel_filter<FuncType> <<< nBlocks, nThreadsPerBlock >>> (d_funcArray.mem, nQueued, filterValue);

    cudaMemcpyWrapped<FuncType>(&h_funcArray[0], d_funcArray.mem, nQueued, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nQueued; i++) {
        if (!h_funcArray[i].done) {
            h_remainingFuncArray.push_back(h_funcArray[i]);
        }
    }
}


template <typename FuncType>
void send_and_report(std::vector<FuncType>& h_funcArray,
                   CudaMallocWrapper<FuncType>& d_funcArray,
                   uint32_t maxPossibleLength)
{
    uint32_t nBlocks, nThreadsPerBlock, nQueued;
    nQueued = h_funcArray.size();

    cudaMemcpyWrapped<FuncType>(d_funcArray.mem, &h_funcArray[0], nQueued, cudaMemcpyHostToDevice);

    nBlocks = nQueued / GPUProps.maxThreadsPerBlock + (nQueued % GPUProps.maxThreadsPerBlock == 0 ? 0 : 1);
    nThreadsPerBlock = (nBlocks == 1) ? nQueued : GPUProps.maxThreadsPerBlock;

    kernel_to_the_end<FuncType> <<< nBlocks, nThreadsPerBlock >>>(d_funcArray.mem, nQueued, maxPossibleLength);

    cudaMemcpyWrapped<FuncType>(&h_funcArray[0], d_funcArray.mem, nQueued, cudaMemcpyDeviceToHost);

    for (int i = 0; i < nQueued; i++) {
        if (h_funcArray[i].lengthSoFar == maxPossibleLength) {
            print_func<FuncType>(h_funcArray[i]);
        }
    }
}


template <typename FuncType>
inline void filter_remaining(std::vector<FuncType>& in, std::vector<FuncType>& out, 
                             CudaMallocWrapper<FuncType>& d_funcArray,
                             uint32_t filterValue)
{
    uint32_t total_sent = 0;
    std::vector<FuncType> toSend;
    for (int i = 0; i < in.size(); i++) {
        toSend.push_back(in[i]);
        if (toSend.size() == FUNCS_PER_KERNEL) {
            total_sent += FUNCS_PER_KERNEL;
            std::cerr << "Filter to " << filterValue << ": " << total_sent << "/" << in.size() << std::endl;
            send_and_filter_to<FuncType>(toSend, out, d_funcArray, filterValue);
            toSend.clear();
        }
    }

    if (toSend.size() != 0) {
            total_sent += toSend.size();
            std::cerr << "Filter to " << filterValue << ": " << total_sent << "/" << in.size() << std::endl;
            send_and_filter_to<FuncType>(toSend, out, d_funcArray, filterValue);
    }
}


template <typename FuncType>
static inline void to_the_end(std::vector<FuncType>& h_funcArray,
                              CudaMallocWrapper<FuncType>& d_funcArray,
                              uint32_t maxPossibleLength)
{
    uint32_t total_sent = 0;
    std::vector<FuncType> toSend;
    // And now go to the end
    for (int i = 0; i < h_funcArray.size(); i++) {
        toSend.push_back(h_funcArray[i]);
        if (i == FUNCS_PER_KERNEL) {
            total_sent += FUNCS_PER_KERNEL;
            std::cerr << "To the end: " << total_sent << "/" << h_funcArray.size() << std::endl;
            send_and_report<FuncType>(toSend, d_funcArray, maxPossibleLength);
            toSend.clear();
        }
    }

    if (toSend.size() != 0) {
            total_sent += toSend.size();
            std::cerr << "To the end: " << total_sent << "/" << h_funcArray.size() << std::endl;
            send_and_report<FuncType>(toSend, d_funcArray, maxPossibleLength);
    }
}



template <typename FuncType>
void report(uint32_t nVariables)
{
    getGPUProperties();

    //XXX Carefull with nVariables 32.
    uint32_t maxPossibleLength = (1 << nVariables) - 1;

    std::vector<FuncType> h_funcArray_a;
    std::vector<FuncType> h_funcArray_b;

    /* Allocate memory for arrays on device */
    CudaMallocWrapper<FuncType> d_funcArray(FUNCS_PER_KERNEL);

    generate_functions<FuncType>(nVariables, h_funcArray_a);

    // The following filtering pattern has been empirically found as being not too far from
    // optimal, it seems.

    filter_remaining<FuncType>(h_funcArray_a, h_funcArray_b, d_funcArray, maxPossibleLength/1000);

    h_funcArray_a.clear();
    filter_remaining<FuncType>(h_funcArray_b, h_funcArray_a, d_funcArray, maxPossibleLength/10);

    h_funcArray_b.clear();
    filter_remaining<FuncType>(h_funcArray_a, h_funcArray_b, d_funcArray, 2 * maxPossibleLength/10);

    h_funcArray_a.clear();
    filter_remaining<FuncType>(h_funcArray_b, h_funcArray_a, d_funcArray, 3 * maxPossibleLength/10);

    h_funcArray_b.clear();
    filter_remaining<FuncType>(h_funcArray_a, h_funcArray_b, d_funcArray, 4 * maxPossibleLength/10);

    h_funcArray_a.clear();
    filter_remaining<FuncType>(h_funcArray_b, h_funcArray_a, d_funcArray, 5 * maxPossibleLength/10);

    h_funcArray_b.clear();
    filter_remaining<FuncType>(h_funcArray_a, h_funcArray_b, d_funcArray, 6 * maxPossibleLength/10);

    h_funcArray_a.clear();
    filter_remaining<FuncType>(h_funcArray_b, h_funcArray_a, d_funcArray, 7 * maxPossibleLength/10);

    h_funcArray_b.clear();
    filter_remaining<FuncType>(h_funcArray_a, h_funcArray_b, d_funcArray, 8 * maxPossibleLength/10);

    h_funcArray_a.clear();
    filter_remaining(h_funcArray_b, h_funcArray_a, d_funcArray, 9 * maxPossibleLength/10);

    to_the_end<FuncType>(h_funcArray_a, d_funcArray, maxPossibleLength);
}






///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Form x0 + a + b + cd ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////


struct __align__(16) Function_0_a_b_cd {
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t d;
    uint8_t nVariables;

    uint32_t curVal;
    uint32_t lengthSoFar;
    bool done;
};


template <>
void print_func<Function_0_a_b_cd> (Function_0_a_b_cd& func)
{
    std::cout << "0," << (uint32_t)func.a << "," << (uint32_t)func.b
              << ",(" << (uint32_t)func.c << "," << (uint32_t)func.d << ")"
              << std::endl;
}



template <>
bool smaller_or_equal<Function_0_a_b_cd>(Function_0_a_b_cd& one, Function_0_a_b_cd& other)
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


template <>
bool canonical<Function_0_a_b_cd>(Function_0_a_b_cd& func)
{
    int32_t ar = func.nVariables - func.a;
    int32_t br = func.nVariables - func.b;
    int32_t cr = func.nVariables - func.c;
    int32_t dr = func.nVariables - func.d;

    struct Function_0_a_b_cd other = {br, ar, dr, cr, func.nVariables};
    return smaller_or_equal<Function_0_a_b_cd>(func, other);
}


template <>
__global__ void kernel_to_the_end<Function_0_a_b_cd> (struct Function_0_a_b_cd *d_funcArray,
                                                      uint32_t nQueued, uint32_t maxPossibleLength)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        struct Function_0_a_b_cd func = d_funcArray[me];

        uint8_t a = func.a;
        uint8_t b = func.b;
        uint8_t c = func.c;
        uint8_t d = func.d;
        uint8_t nVariables = func.nVariables;

        uint32_t curVal = func.curVal;
        uint32_t length = func.lengthSoFar;
        uint32_t newBit = 0;

        do {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                    (bit(c, curVal) & bit(d, curVal));
            newBit &= 0x1;
            curVal = (curVal >> 1) | (newBit << (nVariables - 1));

            length++;
        } while (curVal != 1);

        d_funcArray[me].lengthSoFar = length;
        d_funcArray[me].done = true;
    }
}




template <>
__global__ void kernel_filter<Function_0_a_b_cd>(Function_0_a_b_cd *d_funcArray,
                                                 uint32_t nQueued, uint32_t filterValue)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        // Copy things into registers
        struct Function_0_a_b_cd func = d_funcArray[me];

        uint8_t a = func.a;
        uint8_t b = func.b;
        uint8_t c = func.c;
        uint8_t d = func.d;
        uint8_t nVariables = func.nVariables;

        uint32_t curVal = func.curVal;
        uint32_t length = func.lengthSoFar;

        uint32_t newBit = 0;

        // Unroll the first iteration to put curVal != 1 in the for condition
        newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                (bit(c, curVal) & bit(d, curVal));
        newBit &= 0x1;
        curVal = (curVal >> 1) | (newBit << (nVariables - 1));
        length++;

        for (/* done */; curVal != 1 && length < filterValue; length++) {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                    (bit(c, curVal) & bit(d, curVal));
            newBit &= 0x1;
            curVal = (curVal >> 1) | (newBit << (nVariables - 1));
        }

        if (curVal == 1) { /* We are done */
            d_funcArray[me].done = true;
            d_funcArray[me].lengthSoFar = length;
        } else { /* Some more to go */
            d_funcArray[me].done = false;
            d_funcArray[me].lengthSoFar = length;
            d_funcArray[me].curVal = curVal;
        }

    }
}




/************************************ Functions generation **************************************/
template <>
void generate_functions<Function_0_a_b_cd>(uint32_t nVariables, std::vector<Function_0_a_b_cd>& funcArray)
{

    /* Generate the functions */
    for (int32_t a = 1; a <= (nVariables + 1) / 2; a++) {
        for (int32_t b = a + 1; b <= nVariables - 1; b++) {

            for (int32_t c = 1; c <= nVariables - 2; c++) {
                for (int32_t d = c + 1; d <= nVariables - 1; d++) {

                    // Keep the function for later evaluation
                    struct Function_0_a_b_cd func = {a, b, c, d, nVariables, 1, 0, false};

                    if (!canonical<Function_0_a_b_cd>(func)) {
                        continue;
                    }
                    funcArray.push_back(func);
                }
            }
        }
    }
}



///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////// Form x0 + a + bc + de //////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////


struct __align__(16) Function_0_a_bc_de {
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t d;
    uint8_t e;
    uint8_t nVariables;

    uint32_t curVal;
    uint32_t lengthSoFar;
    bool done;
};


template <>
void print_func<Function_0_a_bc_de> (Function_0_a_bc_de& func)
{
    std::cout << "0," << (uint32_t)func.a
              << ",(" << (uint32_t)func.b << "," << (uint32_t)func.c << ")"
              << ",(" << (uint32_t)func.d << "," << (uint32_t)func.e << ")"
              << std::endl;
}



template <>
bool smaller_or_equal<Function_0_a_bc_de>(Function_0_a_bc_de& one, Function_0_a_bc_de& other)
{
    if (one.a < other.a)
        return true;

    if (one.a == other.a && one.b < other.b)
        return true;

    if (one.a == other.a && one.b == other.b && one.c < other.c)
        return true;

    if (one.a == other.a && one.b == other.b && one.c == other.c && one.d < other.d)
        return true;

    if (one.a == other.a && one.b == other.b && one.c == other.c && one.d == other.d && one.e <= other.e)
        return true;

    return false;
}


template <>
bool canonical<Function_0_a_bc_de>(Function_0_a_bc_de& func)
{
    int32_t ar = func.nVariables - func.a;
    int32_t br = func.nVariables - func.b;
    int32_t cr = func.nVariables - func.c;
    int32_t dr = func.nVariables - func.d;
    int32_t er = func.nVariables - func.e;

    // Commutativity of the two products, checked here because hard to
    // prevent in the generating loops
    if (func.b == func.d && func.c == func.e) {
        return false;
    }

    Function_0_a_bc_de f1 = {ar, er, dr, cr, br, func.nVariables};
    Function_0_a_bc_de f2 = {ar, cr, br, er, dr, func.nVariables};
    Function_0_a_bc_de f3 = {func.a, func.d, func.e, func.b, func.c, func.nVariables};

    return (smaller_or_equal<Function_0_a_bc_de>(func, f1)
         && smaller_or_equal<Function_0_a_bc_de>(func, f2)
         && smaller_or_equal<Function_0_a_bc_de>(func, f3));
}


template <>
__global__ void kernel_to_the_end<Function_0_a_bc_de> (struct Function_0_a_bc_de *d_funcArray,
                                                      uint32_t nQueued, uint32_t maxPossibleLength)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        struct Function_0_a_bc_de func = d_funcArray[me];

        uint8_t a = func.a;
        uint8_t b = func.b;
        uint8_t c = func.c;
        uint8_t d = func.d;
        uint8_t e = func.e;
        uint8_t nVariables = func.nVariables;

        uint32_t curVal = func.curVal;
        uint32_t length = func.lengthSoFar;
        uint32_t newBit = 0;

        do {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^
                    (bit(b, curVal) & bit(c, curVal)) ^
                    (bit(d, curVal) & bit(e, curVal));
            curVal = (curVal >> 1) | (newBit << (nVariables - 1));

            length++;
        } while (curVal != 1);

        d_funcArray[me].lengthSoFar = length;
        d_funcArray[me].done = true;
    }
}




template <>
__global__ void kernel_filter<Function_0_a_bc_de>(Function_0_a_bc_de *d_funcArray,
                                                 uint32_t nQueued, uint32_t filterValue)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        // Copy things into registers
        struct Function_0_a_bc_de func = d_funcArray[me];

        uint8_t a = func.a;
        uint8_t b = func.b;
        uint8_t c = func.c;
        uint8_t d = func.d;
        uint8_t e = func.e;
        uint8_t nVariables = func.nVariables;

        uint32_t curVal = func.curVal;
        uint32_t length = func.lengthSoFar;

        uint32_t newBit = 0;

        // Unroll the first iteration to put curVal != 1 in the for condition
        newBit = bit(0, curVal) ^ bit(a, curVal) ^
                (bit(b, curVal) & bit(c, curVal)) ^
                (bit(d, curVal) & bit(e, curVal));
        curVal = (curVal >> 1) | (newBit << (nVariables - 1));
        length++;

        for (/* done */; curVal != 1 && length < filterValue; length++) {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^
                    (bit(b, curVal) & bit(c, curVal)) ^
                    (bit(d, curVal) & bit(e, curVal));
            curVal = (curVal >> 1) | (newBit << (nVariables - 1));
        }

        if (curVal == 1) { /* We are done */
            d_funcArray[me].done = true;
            d_funcArray[me].lengthSoFar = length;
        } else { /* Some more to go */
            d_funcArray[me].done = false;
            d_funcArray[me].lengthSoFar = length;
            d_funcArray[me].curVal = curVal;
        }

    }
}




/************************************ Functions generation **************************************/
template <>
void generate_functions<Function_0_a_bc_de>(uint32_t nVariables, std::vector<Function_0_a_bc_de>& funcArray)
{

    /* Generate the functions */
    for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {

        for (uint8_t b = 1; b <= nVariables - 2; b++) {
            for (uint8_t c = b + 1; c <= nVariables - 1; c++) {

                for (uint8_t d = b; d <= nVariables - 2; d++) {
                    for (uint8_t e = d + 1; e <= nVariables - 1; e++) {

                        // Keep the function for later evaluation
                        struct Function_0_a_bc_de func = {a, b, c, d, e, nVariables, 1, 0, false};

                        if (!canonical<Function_0_a_bc_de>(func)) {
                            continue;
                        }

                        funcArray.push_back(func);
                    }
                }
            }
        }
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
static void send_and_report(FunctionType *h_funcArray, FunctionType *d_funcArray,
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
                            send_and_report<Function_0_a_bc_de>(h_funcArray, d_funcArray, h_isMaxLength,
                                    d_isMaxLength, enqueued, m_maxPossibleLength);
                            enqueued = 0;
                        }
                    }
                }
            }
        }
    }

    if (enqueued != 0) {
        send_and_report<Function_0_a_bc_de>(h_funcArray, d_funcArray, h_isMaxLength,
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
                                send_and_report<Function_0_a_b_c_d_ef>(h_funcArray, d_funcArray, h_isMaxLength,
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
        send_and_report<Function_0_a_b_c_d_ef>(h_funcArray, d_funcArray, h_isMaxLength,
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
                            send_and_report<Function_0_a_b_cde>(h_funcArray, d_funcArray, h_isMaxLength,
                                    d_isMaxLength, enqueued, m_maxPossibleLength);
                            enqueued = 0;
                        }

                    }
                }
            }
        }
    }

    if (enqueued != 0) {
        send_and_report<Function_0_a_b_cde>(h_funcArray, d_funcArray, h_isMaxLength,
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
    report<Function_0_a_b_cd>(atoi(argv[1]));
    report<Function_0_a_bc_de>(atoi(argv[1]));
    return 0;
}
