#include <stdexcept>
#include <vector>
#include <iostream>

#include <stdint.h>

#include "dbg.h"


#define FUNCS_PER_KERNEL (1 << 20)



/**********************************************************************************************
 ********************************* Form x0 + a + b + cd ***************************************
 **********************************************************************************************/


class Function_0_a_b_cd {
        public:
                uint8_t a;
                uint8_t b;
                uint8_t c;
                uint8_t d;
                uint8_t nVariables;
};


bool smaller_or_equal_0_a_b_cd( Function_0_a_b_cd& one,  Function_0_a_b_cd& other)
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

bool canonical_0_a_b_cd( Function_0_a_b_cd& func)
{
        uint8_t ar = func.nVariables - func.a;
        uint8_t br = func.nVariables - func.b;
        uint8_t cr = func.nVariables - func.c;
        uint8_t dr = func.nVariables - func.d;

         Function_0_a_b_cd other = {br, ar, dr, cr, func.nVariables};
        return smaller_or_equal_0_a_b_cd(func, other);
}


#define bit(nBit, val) (((val) >> (nBit)) & 1)




uint32_t cycle_length_0_a_b_cd(const Function_0_a_b_cd& func)
{
        uint32_t length = 0;
        uint32_t newBit;
        uint32_t curVal = 1;

        do {
            newBit = bit(0, curVal) ^ bit(func.a, curVal) ^ bit(func.b, curVal) ^
                    (bit(func.c, curVal) & bit(func.d, curVal));
            curVal = (curVal >> 1) | (newBit << (func.nVariables - 1));

            length++;
        } while (curVal != 1);

        return length;
}

void print_function_0_a_b_cd( Function_0_a_b_cd& func)
{
#pragma omp cricital
        {
        std::cout << "0," << (uint32_t) func.a << "," << (uint32_t) func.b
                  << ",(" << (uint32_t) func.c << "," << (uint32_t) func.d << ")"
                  << std::endl;
        }
}





/************************************ Functions generation **************************************/
void report_0_a_b_cd(uint32_t nVariables)
{
        //XXX Carefull with nVariables 32.
        uint32_t maxPossibleLength = (1 << nVariables) - 1;

        /* Generate the functions */
        for (uint32_t a = 1; a <= (nVariables + 1) / 2; a++) {
                for (uint32_t b = a + 1; b <= nVariables - 1; b++) {

                        for (uint32_t c = 1; c <= nVariables - 2; c++) {
                                for (uint32_t d = c + 1; d <= nVariables - 1; d++) {

                                        // Keep the function for later evaluation
                                         Function_0_a_b_cd func = {a, b, c, d, nVariables, 1, 0, false};


                                        #pragma omp task
                                        {
                                                if (canonical_0_a_b_cd(func)) {
                                                        uint32_t length = cycle_length_0_a_b_cd(func);
                                                        if (length == maxPossibleLength) {
                                                                print_function_0_a_b_cd(func);
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

#pragma omp taskwait
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
#pragma omp parallel
#pragma omp single
        report_0_a_b_cd(atoi(argv[1]));

        return 0;
}
