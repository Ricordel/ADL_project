#include <stdexcept>
#include <vector>
#include <iostream>
#include <algorithm>

#include <cuda.h>
#include <stdint.h>
#include <getopt.h>
#include <cstdlib>

#include "cudaMallocWrapper.hpp"
#include "dbg.h"


#define FUNCS_PER_KERNEL (1 << 18)
#define N_FUNCS_TO_REPORT 10U



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

template <class FuncType>
bool smaller_or_equal(FuncType& one, FuncType& other);

template <class FuncType>
bool canonical(FuncType& func);


template <class FuncType>
__global__ void kernel_to_the_end(FuncType *d_funcArray,
                                  uint32_t nQueued, uint32_t maxPossibleLength);

template <class FuncType>
__global__ void kernel_filter(FuncType *d_funcArray,
                              uint32_t nQueued, uint32_t filterValue);


template <class FuncType> void print_func(FuncType& func);

template <class FuncType>
void generate_functions(uint32_t nVariables, std::vector<FuncType>& funcArray, int32_t keepProba);



////// The following templates are generic ///////

// Allow to sort by DECREASING cycle length order
template <class FuncKind>
bool compare_funcs(const FuncKind& one, const FuncKind& other)
{
        return (one.lengthSoFar > other.lengthSoFar);
}


template <class FuncType>
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


template <class FuncType>
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


    std::sort(h_funcArray.begin(), h_funcArray.end(), compare_funcs<FuncType>);

    //FIXME: There would be a problem here if the number of functions remaining
    // after the last 'filter' operation was less than N_FUNCS_TO_REPORT. Then,
    // less functions would be reported.
    // The correction of this problem would require some extra computation
    // in filter_remaining. Considering that with N_FUNCS_TO_REPORT == 10
    // and 10 filtering passes, this will never happen, this code has not
    // been written.
    for (int i = 0; i < N_FUNCS_TO_REPORT && i < h_funcArray.size(); i++) {
        print_func<FuncType>(h_funcArray[i]);
    }

}


template <class FuncType>
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


template <class FuncType>
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



template <class FuncType>
void report(uint32_t nVariables, int32_t keepProba)
{
    getGPUProperties();

    //XXX Carefull with nVariables 32.
    uint32_t maxPossibleLength = (1 << nVariables) - 1;

    std::vector<FuncType> h_funcArray_a;
    std::vector<FuncType> h_funcArray_b;

    /* Allocate memory for arrays on device */
    CudaMallocWrapper<FuncType> d_funcArray(FUNCS_PER_KERNEL);

    generate_functions<FuncType>(nVariables, h_funcArray_a, keepProba);

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
    std::cout << (uint32_t) func.nVariables << " variables: "
              << "0," << (uint32_t)func.a << "," << (uint32_t)func.b
              << ",(" << (uint32_t)func.c << "," << (uint32_t)func.d << ")"
              << ", cycle length: " << func.lengthSoFar << ", max poss length: " << (1 << func.nVariables) - 1
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
void generate_functions<Function_0_a_b_cd>(uint32_t nVariables, std::vector<Function_0_a_b_cd>& funcArray, int32_t keepProba)
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
                    if ( (rand() % 100) < keepProba) {
                        funcArray.push_back(func);
                    }
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
    std::cout << (uint32_t) func.nVariables << " variables: "
              << "0," << (uint32_t)func.a
              << ",(" << (uint32_t)func.b << "," << (uint32_t)func.c << ")"
              << ",(" << (uint32_t)func.d << "," << (uint32_t)func.e << ")"
              << ", cycle length: " << func.lengthSoFar << ", max poss length: " << (1 << func.nVariables) - 1
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
void generate_functions<Function_0_a_bc_de>(uint32_t nVariables, std::vector<Function_0_a_bc_de>& funcArray, int32_t keepProba)
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

                        if ( (rand() % 100) < keepProba) {
                            funcArray.push_back(func);
                        }
                    }
                }
            }
        }
    }
}






///////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////// Form x0 + a + b + c + d + ef ///////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////


struct __align__(16) Function_0_a_b_c_d_ef {
    uint8_t a;
    uint8_t b;
    uint8_t c;
    uint8_t d;
    uint8_t e;
    uint8_t f;
    uint8_t nVariables;

    uint32_t curVal;
    uint32_t lengthSoFar;
    bool done;
};


template <>
void print_func<Function_0_a_b_c_d_ef> (Function_0_a_b_c_d_ef& func)
{
    std::cout << (uint32_t) func.nVariables << " variables: "
              << "0," << (uint32_t)func.a << "," << (uint32_t)func.b
              << ","  << (uint32_t)func.c << "," << (uint32_t)func.d
              << ",(" << (uint32_t)func.e << "," << (uint32_t)func.f << ")"
              << ", cycle length: " << func.lengthSoFar << ", max poss length: " << (1 << func.nVariables) - 1
              << std::endl;
}



template <>
bool smaller_or_equal<Function_0_a_b_c_d_ef>(Function_0_a_b_c_d_ef& one, Function_0_a_b_c_d_ef& other)
{
    if (one.a < other.a)
        return true;

    if (one.a == other.a && one.b < other.b)
        return true;

    if (one.a == other.a && one.b == other.b && one.c < other.c)
        return true;

    if (one.a == other.a && one.b == other.b && one.c == other.c && one.d < other.d)
        return true;

    if (one.a == other.a && one.b == other.b && one.c == other.c && one.d == other.d && one.e < other.e)
        return true;

    if (one.a == other.a && one.b == other.b && one.c == other.c && one.d == other.d && one.e == other.e && one.f <= other.f)
        return true;

    return false;
}


template <>
bool canonical<Function_0_a_b_c_d_ef>(Function_0_a_b_c_d_ef& func)
{
    int32_t ar = func.nVariables - func.a;
    int32_t br = func.nVariables - func.b;
    int32_t cr = func.nVariables - func.c;
    int32_t dr = func.nVariables - func.d;
    int32_t er = func.nVariables - func.e;
    int32_t fr = func.nVariables - func.f;

    Function_0_a_b_c_d_ef other = {dr, cr, br, ar, fr, er, func.nVariables};
    return smaller_or_equal(func, other);
}


template <>
__global__ void kernel_to_the_end<Function_0_a_b_c_d_ef> (struct Function_0_a_b_c_d_ef *d_funcArray,
                                                      uint32_t nQueued, uint32_t maxPossibleLength)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        struct Function_0_a_b_c_d_ef func = d_funcArray[me];

        uint8_t a = func.a;
        uint8_t b = func.b;
        uint8_t c = func.c;
        uint8_t d = func.d;
        uint8_t e = func.e;
        uint8_t f = func.f;
        uint8_t nVariables = func.nVariables;

        uint32_t curVal = func.curVal;
        uint32_t length = func.lengthSoFar;
        uint32_t newBit = 0;

        do {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                     bit(c, curVal) ^ bit(d, curVal) ^
                    (bit(e, curVal) & bit(f, curVal));
            curVal = (curVal >> 1) | (newBit << (nVariables - 1));

            length++;
        } while (curVal != 1);

        d_funcArray[me].lengthSoFar = length;
        d_funcArray[me].done = true;
    }
}




template <>
__global__ void kernel_filter<Function_0_a_b_c_d_ef>(Function_0_a_b_c_d_ef *d_funcArray,
                                                 uint32_t nQueued, uint32_t filterValue)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        // Copy things into registers
        struct Function_0_a_b_c_d_ef func = d_funcArray[me];

        uint8_t a = func.a;
        uint8_t b = func.b;
        uint8_t c = func.c;
        uint8_t d = func.d;
        uint8_t e = func.e;
        uint8_t f = func.f;
        uint8_t nVariables = func.nVariables;

        uint32_t curVal = func.curVal;
        uint32_t length = func.lengthSoFar;

        uint32_t newBit = 0;

        // Unroll the first iteration to put curVal != 1 in the for condition
        newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                 bit(c, curVal) ^ bit(d, curVal) ^
                (bit(e, curVal) & bit(f, curVal));
        curVal = (curVal >> 1) | (newBit << (nVariables - 1));
        length++;

        for (/* done */; curVal != 1 && length < filterValue; length++) {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                     bit(c, curVal) ^ bit(d, curVal) ^
                    (bit(e, curVal) & bit(f, curVal));
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
void generate_functions<Function_0_a_b_c_d_ef>(uint32_t nVariables, std::vector<Function_0_a_b_c_d_ef>& funcArray, int32_t keepProba)
{
    for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
        for (uint8_t b = a + 1; b <= nVariables - 3; b++) { /* -3 to leave room for c and d */
            for (uint8_t c = b + 1; c <= nVariables - 2; c++) { /* -2 to leave room for d */
                for (uint8_t d = c + 1; d <= nVariables - 1; d++) {

                    for (uint8_t e = 1; e <= nVariables - 2; e++) {
                        for (uint8_t f = e + 1; f <= nVariables - 1; f++) {

                            // Keep the function for later evaluation
                            struct Function_0_a_b_c_d_ef func = {a, b, c, d, e, f, nVariables, 1, 0, false};

                            if (!canonical<Function_0_a_b_c_d_ef>(func)) {
                                continue;
                            }

                            if ( (rand() % 100) < keepProba) {
                                funcArray.push_back(func);
                            }
                        }
                    }
                }
            }
        }
    }
}







///////////////////////////////////////////////////////////////////////////////////////////////
///////////////////////////////// Form x0 + a + b + cde ///////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////////////////////


struct __align__(16) Function_0_a_b_cde {
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
void print_func<Function_0_a_b_cde> (Function_0_a_b_cde& func)
{
    std::cout << (uint32_t) func.nVariables << " variables: "
              << 0 << "," << (uint32_t) func.a  << "," << (uint32_t) func.b
              << ",(" << (uint32_t) func.c << "," << (uint32_t) func.d << "," << (uint32_t) func.e << ")"
              << ", cycle length: " << func.lengthSoFar << ", max poss length: " << (1 << func.nVariables) - 1
              << std::endl;
}



template <>
bool smaller_or_equal<Function_0_a_b_cde>(Function_0_a_b_cde& one, Function_0_a_b_cde& other)
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
bool canonical<Function_0_a_b_cde>(Function_0_a_b_cde& func)
{
    int32_t ar = func.nVariables - func.a;
    int32_t br = func.nVariables - func.b;
    int32_t cr = func.nVariables - func.c;
    int32_t dr = func.nVariables - func.d;
    int32_t er = func.nVariables - func.e;

    Function_0_a_b_cde other = {br, ar, er, dr, cr, func.nVariables};
    return smaller_or_equal(func, other);
}


template <>
__global__ void kernel_to_the_end<Function_0_a_b_cde> (struct Function_0_a_b_cde *d_funcArray,
                                                      uint32_t nQueued, uint32_t maxPossibleLength)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        struct Function_0_a_b_cde func = d_funcArray[me];

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
            newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                    (bit(c, curVal) & bit(d, curVal) & bit(e, curVal));
            curVal = (curVal >> 1) | (newBit << (nVariables - 1));

            length++;
        } while (curVal != 1);

        d_funcArray[me].lengthSoFar = length;
        d_funcArray[me].done = true;
    }
}




template <>
__global__ void kernel_filter<Function_0_a_b_cde>(Function_0_a_b_cde *d_funcArray,
                                                 uint32_t nQueued, uint32_t filterValue)
{
    // Get my position
    uint32_t me = blockIdx.x * blockDim.x + threadIdx.x;

    if (me < nQueued) {
        // Copy things into registers
        struct Function_0_a_b_cde func = d_funcArray[me];

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
        newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                (bit(c, curVal) & bit(d, curVal) & bit(e, curVal));
        curVal = (curVal >> 1) | (newBit << (nVariables - 1));
        length++;

        for (/* done */; curVal != 1 && length < filterValue; length++) {
            newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                    (bit(c, curVal) & bit(d, curVal) & bit(e, curVal));
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
void generate_functions<Function_0_a_b_cde>(uint32_t nVariables, std::vector<Function_0_a_b_cde>& funcArray, int32_t keepProba)
{
    for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
        for (uint8_t b = a + 1; b <= nVariables - 1; b++) {

            for (uint8_t c = 1; c <= nVariables - 3; c++) { /* -3 to leave room for d and e */
                for (uint8_t d = c + 1; d <= nVariables - 2; d++) {
                    for (uint8_t e = d + 1; e <= nVariables - 1; e++) {

                        // Keep the function for later evaluation
                        struct Function_0_a_b_cde func = {a, b, c, d, e, nVariables, 1, 0, false};

                        if (!canonical<Function_0_a_b_cde>(func)) {
                            continue;
                        }

                        if ( (rand() % 100) < keepProba) {
                            funcArray.push_back(func);
                        }
                    }
                }
            }
        }
    }
}




/* 
 * Command-line option
 */
static const struct option longOpts[] = {
        {"n-vars", required_argument, NULL, 'n'},
        {"func-kind", required_argument, NULL, 'k'},
        {"keep-proba", required_argument, NULL, 'p'},
};

const char *shortOpts = "nkp";

struct __globalOptions {
        uint32_t nVariables;
        std::string funcKind;
        int32_t keepProba;
} globalOptions;


int main(int argc, char *argv[])
{
    globalOptions.nVariables = 0;
    globalOptions.funcKind = "";
        globalOptions.keepProba = 100;

    int longIndex;

    // Parse program options
    int opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
    while (opt != -1) {
        switch (opt) {
            case 'n':
                globalOptions.nVariables = strtoul(optarg, NULL, 10);
                if (errno == ERANGE) {
                    std::cerr << "Could not parse " << optarg << " as a number" << std::endl;
                    exit(1);
                }
                break;
            case 'k':
                globalOptions.funcKind = std::string(optarg);
                break;
            case 'p':
                globalOptions.keepProba = strtol(optarg, NULL, 10);
                if (errno == ERANGE) {
                    std::cerr << "Could not parse " << optarg << " as a number" << std::endl;
                    exit(1);
                }
                break;
        }

        // Get next option
        opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
    }

    // Filter on the required type of function to test
    if (globalOptions.funcKind == "0_a_b_cd") {
        report<Function_0_a_b_cd>(globalOptions.nVariables, globalOptions.keepProba);
    } else if (globalOptions.funcKind == "0_a_bc_de") {
        report<Function_0_a_bc_de>(globalOptions.nVariables, globalOptions.keepProba);
    } else if (globalOptions.funcKind == "0_a_b_c_d_ef") {
        report<Function_0_a_b_c_d_ef>(globalOptions.nVariables, globalOptions.keepProba);
    } else if (globalOptions.funcKind == "0_a_b_cde") {
        report<Function_0_a_b_cde>(globalOptions.nVariables, globalOptions.keepProba);
    } else {
        std::cerr << "Function kind " << globalOptions.funcKind << " not recognized" << std::endl;
    }

    return 0;
}
