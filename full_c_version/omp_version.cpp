#include <stdexcept>
#include <vector>
#include <iostream>
#include <cstdlib>

#include <stdint.h>


#define FUNCS_PER_KERNEL (1 << 20)


#define bit(nBit, val) (((val) >> (nBit)) & 0x1)


template <typename T> void report(uint8_t nVariables);



/**********************************************************************************************
 ********************************* Form x0 + a + b + cd ***************************************
 **********************************************************************************************/


class Function_0_a_b_cd {
        public:
                // No constructor, it seems to be less efficient than initialazing with 
                // a braces struct, while the same stuff is to be done.
                uint8_t a, b, c, d;
                uint8_t nVariables;

                bool smaller_or_equal(const Function_0_a_b_cd& other) const
                {
                        if (a < other.a)
                                return true;

                        if (a == other.a && b < other.b)
                                return true;

                        if (a == other.a && b == other.b && c < other.c)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d <= other.d)
                                return true;

                        return false;
                }


                bool is_canonical() const
                {
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;

                        Function_0_a_b_cd other = {br, ar, dr, cr, nVariables};
                        return smaller_or_equal(other);
                }


                uint32_t cycle_length()
                {
                        uint32_t length = 0;
                        uint32_t newBit;
                        uint32_t curVal = 1;

                        do {
                                newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                                        (bit(c, curVal) & bit(d, curVal));
                                curVal = (curVal >> 1) | (newBit << (nVariables - 1));

                                length++;
                        } while (curVal != 1);

                        return length;
                }

                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << "0," << (uint32_t) a << "," << (uint32_t) b
                                  << ",(" << (uint32_t) c << "," << (uint32_t) d
                                  << ")" << std::endl;
                }
};



template <>
void report<Function_0_a_b_cd>(uint8_t nVariables)
{
        uint32_t maxPossibleLength = (nVariables == 32) ? 0xffffffff : (1 << nVariables) - 1;

        /* Generate the functions */
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                for (uint8_t b = a + 1; b <= nVariables - 1; b++) {

                        for (uint8_t c = 1; c <= nVariables - 2; c++) {
                                for (uint8_t d = c + 1; d <= nVariables - 1; d++) {

                                        // Keep the function for later evaluation
                                        Function_0_a_b_cd func = {a, b, c, d, nVariables};

                                        //#pragma omp task
                                        {
                                                if (func.is_canonical()) {
                                                        uint32_t length = func.cycle_length();
                                                        if (length == maxPossibleLength) {
                                                        #pragma omp critical
                                                                func.print();
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }

//#pragma omp taskwait
}






/**********************************************************************************************
 ********************************* Form x0 + a + bc + de ***************************************
 **********************************************************************************************/


class Function_0_a_bc_de {
        public:
                // No constructor, it seems to be less efficient than initialazing with 
                // a braces struct, while the same stuff is to be done.
                uint8_t a, b, c, d, e;
                uint8_t nVariables;

                bool smaller_or_equal(const Function_0_a_bc_de& other) const
                {
                        if (a < other.a)
                                return true;

                        if (a == other.a && b < other.b)
                                return true;

                        if (a == other.a && b == other.b && c < other.c)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d < other.d)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d == other.d && e <= other.e)
                                return true;

                        return false;
                }


                bool is_canonical() const
                {
                        int8_t ar = nVariables - a;
                        int8_t br = nVariables - b;
                        int8_t cr = nVariables - c;
                        int8_t dr = nVariables - d;
                        int8_t er = nVariables - e;

                        if (b == d && c == e) {
                                return false;
                        }

                        Function_0_a_bc_de f1 = {ar, er, dr, cr, br, nVariables};
                        Function_0_a_bc_de f2 = {ar, cr, br, er, dr, nVariables};
                        Function_0_a_bc_de f3 = {a, d, e, b, c, nVariables};

                        return (smaller_or_equal(f1) && smaller_or_equal(f2) && smaller_or_equal(f3));
                }


                uint32_t cycle_length()
                {
                        uint32_t length = 0;
                        uint32_t newBit;
                        uint32_t curVal = 1;

                        do {
                                //XXX essayer d'inliner newBit, pour voir... Mais a priori, un 
                                //XXX compilateur est assez bon pour faire ça tout seul.
                                newBit = bit(0, curVal) ^ bit(a, curVal) ^
                                        (bit(b, curVal) & bit(c, curVal)) ^
                                        (bit(d, curVal) & bit(e, curVal));
                                curVal = (curVal >> 1) | (newBit << (nVariables - 1));

                                length++;
                        } while (curVal != 1);

                        return length;
                }


                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << 0 << "," << (uint32_t) a  << ",(" << (uint32_t) b << "," << (uint32_t) c << "),("
                                  << (uint32_t) d << "," << (uint32_t) e << ")" << std::endl;
                }
};



template <>
void report<Function_0_a_bc_de>(uint8_t nVariables)
{
        uint32_t maxPossibleLength = (nVariables == 32) ? 0xffffffff : (1 << nVariables) - 1;

        /* Generate the functions */
        for (int32_t a = 1; a <= (nVariables + 1) / 2; a++) {

                for (int32_t b = 1; b <= nVariables - 2; b++) {
                        for (int32_t c = b + 1; c <= nVariables - 1; c++) {

                                for (int32_t d = b; d <= nVariables - 2; d++) {
                                        for (int32_t e = d + 1; e <= nVariables - 1; e++) {

                                                // Keep the function for later evaluation
                                                Function_0_a_bc_de func = {a, b, c, d, e, nVariables};

                                                #pragma omp task
                                                {
                                                        if (func.is_canonical()) {
                                                                uint32_t length = func.cycle_length();
                                                                if (length == maxPossibleLength) {
                                                                #pragma omp critical
                                                                        func.print();
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }

        }
#pragma omp taskwait
}






/**********************************************************************************************
 ***************************** Form x0 + a + b + c + d + ef ***********************************
 **********************************************************************************************/


class Function_0_a_b_c_d_ef {
        public:
                // No constructor, it seems to be less efficient than initialazing with 
                // a braces struct, while the same stuff is to be done.
                uint8_t a, b, c, d, e, f;
                uint8_t nVariables;

                bool smaller_or_equal(const Function_0_a_b_c_d_ef& other) const
                {
                        if (a < other.a)
                                return true;

                        if (a == other.a && b < other.b)
                                return true;

                        if (a == other.a && b == other.b && c < other.c)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d < other.d)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d == other.d && e < other.e)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d == other.d && e == other.e && f == other.f)
                                return true;

                        return false;
                }


                bool is_canonical() const
                {
                        int8_t ar = nVariables - a;
                        int8_t br = nVariables - b;
                        int8_t cr = nVariables - c;
                        int8_t dr = nVariables - d;
                        int8_t er = nVariables - e;
                        int8_t fr = nVariables - f;

                        Function_0_a_b_c_d_ef other = {dr, cr, br, ar, fr, er, nVariables};

                        return smaller_or_equal(other);
                }


                uint32_t cycle_length()
                {
                        uint32_t length = 0;
                        uint32_t newBit;
                        uint32_t curVal = 1;

                        do {
                                newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                                         bit(c, curVal) ^ bit(d, curVal) ^
                                        (bit(e, curVal) & bit(f, curVal));

                                curVal = (curVal >> 1) | (newBit << (nVariables - 1));

                                length++;
                        } while (curVal != 1);

                        return length;
                }


                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << 0 << "," << (uint32_t) a  << "," << (uint32_t) b << ","
                                  << (uint32_t) c << "," << (uint32_t) d << ",(" << (uint32_t) e << "," << (uint32_t) f << ")"
                                  << std::endl;
                }

};



template <>
void report<Function_0_a_b_c_d_ef>(uint8_t nVariables)
{
        uint32_t maxPossibleLength = (nVariables == 32) ? 0xffffffff : (1 << nVariables) - 1;

        /* Generate the functions */
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                for (uint8_t b = a + 1; b <= nVariables - 3; b++) { /* -3 to leave room for c and d */
                        for (uint8_t c = b + 1; c <= nVariables - 2; c++) { /* -2 to leave room for d */
                                for (uint8_t d = c + 1; d <= nVariables - 1; d++) {

                                        for (uint8_t e = 1; e <= nVariables - 2; e++) {
                                                for (uint8_t f = e + 1; f <= nVariables - 1; f++) {

                                                        // Keep the function for later evaluation
                                                        Function_0_a_b_c_d_ef func = {a, b, c, d, e, f, nVariables};
        
                                                        #pragma omp task
                                                        {
                                                                if (func.is_canonical()) {
                                                                        uint32_t length = func.cycle_length();
                                                                        if (length == maxPossibleLength) {
                                                                        #pragma omp critical
                                                                                func.print();
                                                                        }
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
#pragma omp taskwait
}







/**********************************************************************************************
 ***ù***************************** Form x0 + a + b + cde **************************************
 **********************************************************************************************/


class Function_0_a_b_cde {
        public:
                // No constructor, it seems to be less efficient than initialazing with 
                // a braces struct, while the same stuff is to be done.
                uint8_t a, b, c, d, e;
                uint8_t nVariables;

                bool smaller_or_equal(const Function_0_a_b_cde& other) const
                {
                        if (a < other.a)
                                return true;

                        if (a == other.a && b < other.b)
                                return true;

                        if (a == other.a && b == other.b && c < other.c)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d < other.d)
                                return true;

                        if (a == other.a && b == other.b && c == other.c && d == other.d && e <= other.e)
                                return true;

                        return false;
                }


                bool is_canonical() const
                {
                        int32_t ar = nVariables - a;
                        int32_t br = nVariables - b;
                        int32_t cr = nVariables - c;
                        int32_t dr = nVariables - d;
                        int32_t er = nVariables - e;

                        Function_0_a_b_cde other = {br, ar, er, dr, cr, nVariables};
                        return smaller_or_equal(other);
                }


                uint32_t cycle_length()
                {
                        uint32_t length = 0;
                        uint32_t newBit;
                        uint32_t curVal = 1;

                        do {
                                newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                                        (bit(c, curVal) & bit(d, curVal) & bit(e, curVal));

                                curVal = (curVal >> 1) | (newBit << (nVariables - 1));

                                length++;
                        } while (curVal != 1);

                        return length;
                }


                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << 0 << "," << (uint32_t) a  << "," << (uint32_t) b
                                  << ",(" << (uint32_t) c << "," << (uint32_t) d << "," << (uint32_t) e << ")"
                                  << std::endl;
                }
};



template <>
void report<Function_0_a_b_cde>(uint8_t nVariables)
{
        uint32_t maxPossibleLength = (nVariables == 32) ? 0xffffffff : (1 << nVariables) - 1;

        /* Generate the functions */
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                for (uint8_t b = a + 1; b <= nVariables - 1; b++) {


                        for (uint8_t c = 1; c <= nVariables - 3; c++) { /* -3 to leave room for d and e */
                                for (uint8_t d = c + 1; d <= nVariables - 2; d++) {
                                        for (uint8_t e = d + 1; e <= nVariables - 1; e++) {

                                                // Keep the function for later evaluation
                                                Function_0_a_b_cde func = {a, b, c, d, e, nVariables};

                                                #pragma omp task
                                                {
                                                        if (func.is_canonical()) {
                                                                uint32_t length = func.cycle_length();
                                                                if (length == maxPossibleLength) {
                                                                #pragma omp critical
                                                                        func.print();
                                                                }
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


        /***********************************************************************
         ******************** For x0 + xa + xb.xc + xd.xe **********************
         ***********************************************************************/

        FuncGenerator_0_a_bc_de::FuncGenerator_0_a_bc_de(uint32_t nVariables)
                : nVariables(nVariables), maxPossibleLength((1 << nVariables) - 1)
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

                for (int32_t a = 1; a <= (nVariables + 1) / 2; a++) {

                        for (int32_t b = 1; b <= nVariables - 2; b++) {
                                for (int32_t c = b + 1; c <= nVariables - 1; c++) {

                                        for (int32_t d = b; d <= nVariables - 2; d++) {
                                                for (int32_t e = d + 1; e <= nVariables - 1; e++) {

                                                        // Keep the function for later evaluation
                                                        Function_0_a_bc_de func(a, b, c, d, e, nVariables);
                                                        if (!func.isCanonicalForm()) {
                                                                continue;
                                                        }

                                                        h_funcArray[enqueued++] = func;

                                                        if (enqueued == GPUProps.actualConcurrentThreads) {
                                                                sendAndReport<Function_0_a_bc_de>(h_funcArray, d_funcArray, h_isMaxLength,
                                                                                d_isMaxLength, enqueued, maxPossibleLength);
                                                                enqueued = 0;
                                                        }
                                                }
                                        }
                                }
                        }
                }

                if (enqueued != 0) {
                        sendAndReport<Function_0_a_bc_de>(h_funcArray, d_funcArray, h_isMaxLength,
                                        d_isMaxLength, enqueued, maxPossibleLength);
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
                : nVariables(nVariables), maxPossibleLength((1 << nVariables) - 1)
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

                for (int32_t a = 1; a <= (nVariables + 1) / 2; a++) {
                        for (int32_t b = a + 1; b <= nVariables - 3; b++) { /* -3 to leave room for c and d */
                                for (int32_t c = b + 1; c <= nVariables - 2; c++) { /* -2 to leave room for d */
                                        for (int32_t d = c + 1; d <= nVariables - 1; d++) {

                                                for (int32_t e = 1; e <= nVariables - 2; e++) {
                                                        for (int32_t f = e + 1; f <= nVariables - 1; f++) {

                                                                // Keep the function for later evaluation
                                                                Function_0_a_b_c_d_ef func(a, b, c, d, e, f, nVariables);
                                                                if (!func.isCanonicalForm()) {
                                                                        continue;
                                                                }

                                                                h_funcArray[enqueued++] = func;

                                                                if (enqueued == GPUProps.actualConcurrentThreads) {
                                                                        sendAndReport<Function_0_a_b_c_d_ef>(h_funcArray, d_funcArray, h_isMaxLength,
                                                                                        d_isMaxLength, enqueued, maxPossibleLength);
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
                                        d_isMaxLength, enqueued, maxPossibleLength);
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
                : nVariables(nVariables), maxPossibleLength((1 << nVariables) - 1)
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

                for (int32_t a = 1; a <= (nVariables + 1) / 2; a++) {
                        for (int32_t b = a + 1; b <= nVariables - 1; b++) {


                                for (int32_t c = 1; c <= nVariables - 3; c++) { /* -3 to leave room for d and e */
                                        for (int32_t d = c + 1; d <= nVariables - 2; d++) {
                                                for (int32_t e = d + 1; e <= nVariables - 1; e++) {

                                                        // Keep the function for later evaluation
                                                        Function_0_a_b_cde func(a, b, c, d, e, nVariables);
                                                        if (!func.isCanonicalForm()) {
                                                                continue;
                                                        }

                                                        h_funcArray[enqueued++] = func;

                                                        if (enqueued == GPUProps.actualConcurrentThreads) {
                                                                sendAndReport<Function_0_a_b_cde>(h_funcArray, d_funcArray, h_isMaxLength,
                                                                                d_isMaxLength, enqueued, maxPossibleLength);
                                                                enqueued = 0;
                                                        }

                                                }
                                        }
                                }
                        }
                }

                if (enqueued != 0) {
                        sendAndReport<Function_0_a_b_cde>(h_funcArray, d_funcArray, h_isMaxLength,
                                        d_isMaxLength, enqueued, maxPossibleLength);
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
        {
                report<Function_0_a_b_cd>(atoi(argv[1]));
                report<Function_0_a_bc_de>(atoi(argv[1]));
                report<Function_0_a_b_c_d_ef>(atoi(argv[1]));
                report<Function_0_a_b_cde>(atoi(argv[1]));
        }

        return 0;
}
