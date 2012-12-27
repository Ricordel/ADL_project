#include <vector>
#include <stdexcept>
#include <iostream>

#include <omp.h>
#include <cuda.h>

#define __CUDA
#include "FuncGenerator.hpp"


// Some static variables for the CUDA version. That's not beautiful, I know.
struct GPUProps {
    bool initialized;
    int maxThreadsPerBlock;
    int maxConcurrentBlocks;
    int maxConcurrentThreads;
    int actualConcurrentThreads;
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

    // BUT: we also need to take memory limits into account to know how many functions
    // we can copy on the device, and hence how many functions we can test at once.
    size_t maxMemory = (size_t) (deviceProp.totalGlobalMem * 0.9); // Take some margin for the rest.
    size_t maxFunctionSize = std::max(sizeof(Function_0_a_b_cd),
                             std::max(sizeof(Function_0_a_bc_de),
                             std::max(sizeof(Function_0_a_b_c_d_ef),
                                      sizeof(Function_0_a_b_cde))));
    GPUProps.actualConcurrentThreads = std::min(GPUProps.maxConcurrentThreads, (int) (maxMemory / maxFunctionSize));

    GPUProps.initialized = true;
}





////////////// Generating functions and checking for canonical forms ////////////////
//
// This explanation applies to all isCanonicalForm() functions below.
// When a function x0 + g(x1, ..., xn-1) is of maximum length, then so
// is x0 + g(xn-1, ..., x1). Hence, one of those functions can be derived
// from the other, so we consider them as "duplicates".
// There are other possible "duplicate functions" due to the commutativity
// of '+' and '.' , for instance x0 + xa + xb = x0 + xb + xa.
// Of course, the reverse function also has variants due to commutativity.
// for each function, we choose to report only the so-called "canonical form"
// to prevent from doing unnecessary computations. All other variants (and
// there can be a lot) can be derived from this canonical form.
//
// We decide to order functions by lexicographical order on a, b, c, ...,
// and we decide that the canonical representant of an equivalence class
// of functions is the smallest element according to this order.
// When generating a function, we must first check that it's in canonical
// form before testing and reporting its cycle length, Function_*::isCanonicalForm()
// is responsible for this test.
// In theory, we need to test the generated function against all possible
// variants by commutativity, reverse, and reverse-and-then-commutativity.
// But in practice, we can skip a lot of those test by generating functions
// in a smart way.
//
// Hence, we generate functions so that they are by construction the smallest
// element of the equivalence class given only commutativity. This then
// allows us to describe unambiguously the smallest representant of the
// "reverse" function, and in the end, we only have to test the generated
// function against that smallest element of the reverse representant.
//
// The existence of reverse functions also means that if the index of the
// first variable is > N/2, then the reverse function has its first index
// <= N/2. So the reverse function has already been tested. This allows us
// to cut half of the outermost loop in the function generation, which
// is a nice optimisation.
//
// More complete and rigorous information can be found in the report.
//
///////////////////////////////////////////////////////////////////////////////////////




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

    // Launch kernel
    kernel<FunctionType> <<< nBlocks, nThreadsPerBlock >>> (d_funcArray, d_isMaxLength, enqueued, maxPossibleLength);
    
    ret = cudaMemcpy(h_isMaxLength, d_isMaxLength, enqueued * sizeof(bool), cudaMemcpyDeviceToHost);
    if (ret != cudaSuccess) {
        throw std::runtime_error("Kernel execution failed: " + std::string(cudaGetErrorString(ret)));
    }

    std::cerr << "Kernel has returned and memcpy is done" << std::endl;

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



void FuncGenerator_0_a_b_cd::reportMaxFunctions()
{
    getGPUProperties();

    cudaError_t ret;

    Function_0_a_b_cd *h_funcArray;
    Function_0_a_b_cd *d_funcArray;
    bool *d_isMaxLength;
    bool *h_isMaxLength;

    h_isMaxLength = new bool[GPUProps.actualConcurrentThreads];
    h_funcArray = new Function_0_a_b_cd[GPUProps.actualConcurrentThreads];

    ret = cudaMalloc((void **) &d_isMaxLength, GPUProps.actualConcurrentThreads * sizeof(bool));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for bools");
    }

    ret = cudaMalloc((void **) &d_funcArray, GPUProps.actualConcurrentThreads * sizeof(Function_0_a_b_cd));
    if (ret != cudaSuccess) {
        throw std::runtime_error("No more memory on device for Function_0_a_b_cd");
    }

    uint32_t enqueued = 0;

    for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
        for (int32_t b = a + 1; b <= m_nVariables - 1; b++) {

            for (int32_t c = 1; c <= m_nVariables - 2; c++) {
                for (int32_t d = c + 1; d <= m_nVariables - 1; d++) {

                    // Keep the function for later evaluation
                    Function_0_a_b_cd func(a, b, c, d, m_nVariables);
                    if (!func.isCanonicalForm()) {
                        continue;
                    }

                    h_funcArray[enqueued++] = func;

                    if (enqueued == GPUProps.actualConcurrentThreads) {

                        sendAndReport<Function_0_a_b_cd>(h_funcArray, d_funcArray, h_isMaxLength,
                                                         d_isMaxLength, enqueued, m_maxPossibleLength);
                        enqueued = 0;
                    }

                }
            }
        }
    }


    if (enqueued != 0) {
        sendAndReport<Function_0_a_b_cd>(h_funcArray, d_funcArray, h_isMaxLength,
                                         d_isMaxLength, enqueued, m_maxPossibleLength);
    }

    cudaFree(d_funcArray);
    cudaFree(d_isMaxLength);
    delete [] h_isMaxLength;
    delete [] h_funcArray;
}





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
