#include <stdexcept>
#include <vector>
#include <iostream>
#include <cstdlib>
#include <algorithm>

#include <stdint.h>
#include <getopt.h>
#include <errno.h>



#define N_FUNCS_TO_REPORT 10


#define bit(nBit, val) (((val) >> (nBit)) & 0x1)


template <typename T> void report(uint8_t nVariables);



////////////////////////////////////////////////////////////////////////////////////////////
//
// Function interface:
//      Every function will have the same public interface. Generic comments are gathered
//      on the first Function representant (Function_0_a_b_cd) and are note repetead for
//      other functions. Function kind-specific information can be found in every function.
/////////////////////////////////////////////////////////////////////////////////////////////


/////////// Kepping trace of longest NLFSRs ////////////////////

// This is not the compputationally intensive part, so don't bother with any
// optimization here.

// Allow to sort by DECREASING cycle length order
template <typename FuncKind>
bool compare_funcs(const FuncKind& one, const FuncKind& other)
{
        return (one.m_length > other.m_length);
}


#if 0
// Attention: this is called in a critical section. Do all we can to exit
// as quickly as possible
template <typename FuncKind>
void keep_if_max(std::vector<FuncKind>& maxFunctions, FuncKind candidate)
{
        if (candidate.m_length < maxFunctions[0].m_length) {
                return;
        }



        maxFunctions.push_back(candidate);
        std::sort(maxFunctions.begin(), maxFunctions.end(), compare_funcs<FuncKind>);

        if (maxFunctions.size() == N_FUNCS_TO_REPORT + 1) {
                maxFunctions.erase(maxFunctions.begin());
        } else if (maxFunctions.size() > N_FUNCS_TO_REPORT + 1) {
                std::cerr << "Too much functions, shold not happen !" << std::endl;
        }

        std::cerr << "There are " << maxFunctions.size() << " functions" << std::endl;
}
#endif

/**********************************************************************************************
 ********************************* Form x0 + a + b + cd ***************************************
 **********************************************************************************************/


class Function_0_a_b_cd {
        public:
                uint8_t a, b, c, d;
                uint8_t nVariables;
                uint32_t m_length;

                Function_0_a_b_cd(uint8_t _a, uint8_t _b, uint8_t _c, uint8_t _d, uint8_t _nVariables)
                        : a(_a), b(_b), c(_c), d(_d), nVariables(_nVariables), m_length(0)
                {}


                Function_0_a_b_cd(const Function_0_a_b_cd& other)
                        : a(other.a), b(other.b), c(other.c), d(other.d),
                          nVariables(other.nVariables), m_length(other.m_length)
                {}


                // Compare with another function according do lexicographical order on
                // the variables a, b, c, d.
                //
                // in: other    Function to compare with this
                // return value: true if this is smaller or equal to other
                //               false otherwise
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


                // Tell whether the current function is in canonical form. Each functions has
                // several equivalent, due to commutativity of + and . and to invariance by
                // reversing the arguments. A function is considered in canonical form if it's
                // the smallest member of its equivalence class. Considering the way functions
                // are generated (see the part on function generation), we don't need to compare
                // with all the members of the equivalence class.
                // See the report for more detailed information.
                //
                // return value: true if the function is in canonical form
                //               false otherwise
                bool is_canonical() const
                {
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;

                        Function_0_a_b_cd other(br, ar, dr, cr, nVariables);
                        return smaller_or_equal(other);
                }


                // Compute the length of the NLFSR represented by this function.
                //
                // return value: length of the NLFSR's cycle
                uint32_t compute_cycle_length()
                {
                        if (m_length != 0) {
                                return m_length;
                        }
                        uint32_t length = 0;
                        uint32_t newBit;
                        uint32_t curVal = 1;

                        do {
                                newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                                        (bit(c, curVal) & bit(d, curVal));
                                curVal = (curVal >> 1) | (newBit << (nVariables - 1));

                                length++;
                        } while (curVal != 1);

                        m_length = length;
                        return length;
                }

                // Print the function to stdout, according to the coding already
                // used in the previous paper of maximum length NLFSRs
                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << "0," << (uint32_t) a << "," << (uint32_t) b
                                  << ",(" << (uint32_t) c << "," << (uint32_t) d
                                  << ")"
                                  << ", cycle length: " << m_length << ", max poss length: " << (1 << nVariables) - 1
                                  << std::endl;
                }

};



// Generate all possible functions and compute the length of their cycles.
// If a function is of maximum cycle length, then it's printed to stdout.
// 
// The generation process tries to eliminate as much duplicates as possible
// using simple rules of commutativity of + and . , allowing to reduce the
// number of possible remaining duplicates to test in is_canonical().
//
// The computing of the cycle length of every function is delegated to an
// OpenMP task to be scheduled by the OpenMP runtime.
// Printing to stdout must be done in a critical section to be sure several
// outputs won't interleave. As finding a maximum length NLFSR is a rare
// event, the synchronization overhead is negligible.
//
// in: nVariables       Number of variables in the NLFSR
template <>
void report<Function_0_a_b_cd>(uint8_t nVariables)
{
        std::vector<Function_0_a_b_cd> maxNLFSR;

        /* Generate the functions */
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";
                for (uint8_t b = a + 1; b <= nVariables - 1; b++) {
                        std::cerr << (uint32_t)b << ", ";

                        for (uint8_t c = 1; c <= nVariables - 2; c++) {
                                for (uint8_t d = c + 1; d <= nVariables - 1; d++) {

                                        Function_0_a_b_cd func(a, b, c, d, nVariables);

                                        #pragma omp task shared(maxNLFSR) firstprivate(func)
                                        {
                                                if (func.is_canonical()) {
                                                        func.compute_cycle_length(); // Compute cycle length and keep track in the func.
                                                        #pragma omp critical
                                                                maxNLFSR.push_back(func);
                                                }
                                        }
                                }
                        }
                }
        }

#pragma omp taskwait

        // Sort the functions by cycle length and print the longest ones
        std::sort(maxNLFSR.begin(), maxNLFSR.end(), compare_funcs<Function_0_a_b_cd>);

        std::vector<Function_0_a_b_cd>::iterator it = maxNLFSR.begin();
        for (int i = 0; i < N_FUNCS_TO_REPORT && it != maxNLFSR.end(); i++, it++) {
                it->print();
        }
}






/**********************************************************************************************
 ********************************* Form x0 + a + bc + de ***************************************
 **********************************************************************************************/


class Function_0_a_bc_de {
        public:
                uint8_t a, b, c, d, e;
                uint8_t nVariables;
                uint32_t m_length;

                Function_0_a_bc_de(uint8_t _a, uint8_t _b, uint8_t _c, uint8_t _d, uint8_t _e, uint8_t _nVariables)
                        : a(_a), b(_b), c(_c), d(_d), e(_e), nVariables(_nVariables), m_length(0)
                {}


                Function_0_a_bc_de(const Function_0_a_bc_de& other)
                        : a(other.a), b(other.b), c(other.c), d(other.d), e(other.e),
                          nVariables(other.nVariables), m_length(other.m_length)
                {}

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
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;
                        uint8_t er = nVariables - e;

                        if (b == d && c == e) {
                                return false;
                        }

                        Function_0_a_bc_de f1(ar, er, dr, cr, br, nVariables);
                        Function_0_a_bc_de f2(ar, cr, br, er, dr, nVariables);
                        Function_0_a_bc_de f3(a, d, e, b, c, nVariables);

                        return (smaller_or_equal(f1) && smaller_or_equal(f2) && smaller_or_equal(f3));
                }


                uint32_t compute_cycle_length()
                {
                        if (m_length != 0) {
                                return m_length;
                        }
                        uint32_t length = 0;
                        uint32_t newBit;
                        uint32_t curVal = 1;

                        do {
                                newBit = bit(0, curVal) ^ bit(a, curVal) ^
                                        (bit(b, curVal) & bit(c, curVal)) ^
                                        (bit(d, curVal) & bit(e, curVal));
                                curVal = (curVal >> 1) | (newBit << (nVariables - 1));

                                length++;
                        } while (curVal != 1);

                        m_length = length;
                        return length;
                }


                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << 0 << "," << (uint32_t) a  << ",(" << (uint32_t) b << "," << (uint32_t) c << "),("
                                  << (uint32_t) d << "," << (uint32_t) e << ")"
                                  << ", cycle length: " << m_length << ", max poss length: " << (1 << nVariables) - 1
                                  << std::endl;
                }

};



template <>
void report<Function_0_a_bc_de>(uint8_t nVariables)
{
        std::vector<Function_0_a_bc_de> maxNLFSR;

        /* Generate the functions */
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";

                for (uint8_t b = 1; b <= nVariables - 2; b++) {
                        std::cerr << (uint32_t)b << ", ";
                        for (uint8_t c = b + 1; c <= nVariables - 1; c++) {

                                for (uint8_t d = b; d <= nVariables - 2; d++) {
                                        for (uint8_t e = d + 1; e <= nVariables - 1; e++) {

                                                Function_0_a_bc_de func(a, b, c, d, e, nVariables);

                                                #pragma omp task shared(maxNLFSR)
                                                {
                                                        if (func.is_canonical()) {
                                                                func.compute_cycle_length(); // Compute length and keep track in the func.
                                                                #pragma omp critical
                                                                maxNLFSR.push_back(func);
                                                        }
                                                }
                                        }
                                }
                        }
                }

        }
#pragma omp taskwait

        // Sort the functions by cycle length and print the longest ones
        std::sort(maxNLFSR.begin(), maxNLFSR.end(), compare_funcs<Function_0_a_bc_de>);

        std::vector<Function_0_a_bc_de>::iterator it = maxNLFSR.begin();
        for (int i = 0; i < N_FUNCS_TO_REPORT && it != maxNLFSR.end(); i++, it++) {
                it->print();
        }
}






/**********************************************************************************************
 ***************************** Form x0 + a + b + c + d + ef ***********************************
 **********************************************************************************************/


class Function_0_a_b_c_d_ef {
        public:
                uint8_t a, b, c, d, e, f;
                uint8_t nVariables;
                uint32_t m_length;

                Function_0_a_b_c_d_ef(uint8_t _a, uint8_t _b, uint8_t _c, uint8_t _d,
                                   uint8_t _e, uint8_t _f, uint8_t _nVariables)
                        : a(_a), b(_b), c(_c), d(_d), e(_e), f(_f), nVariables(_nVariables), m_length(0)
                {}


                Function_0_a_b_c_d_ef(const Function_0_a_b_c_d_ef& other)
                        : a(other.a), b(other.b), c(other.c), d(other.d), e(other.e), f(other.f),
                          nVariables(other.nVariables), m_length(other.m_length)
                {}

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
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;
                        uint8_t er = nVariables - e;
                        uint8_t fr = nVariables - f;

                        Function_0_a_b_c_d_ef other(dr, cr, br, ar, fr, er, nVariables);

                        return smaller_or_equal(other);
                }


                uint32_t compute_cycle_length()
                {
                        if (m_length != 0) {
                                return m_length;
                        }
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

                        m_length = length;
                        return length;
                }


                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << 0 << "," << (uint32_t) a  << "," << (uint32_t) b << ","
                                  << (uint32_t) c << "," << (uint32_t) d << ",(" << (uint32_t) e << "," << (uint32_t) f << ")"
                                  << ", cycle length: " << m_length << ", max poss length: " << (1 << nVariables) - 1
                                  << std::endl;
                }


};



template <>
void report<Function_0_a_b_c_d_ef>(uint8_t nVariables)
{
        std::vector<Function_0_a_b_c_d_ef> maxNLFSR;

        /* Generate the functions */
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";
                for (uint8_t b = a + 1; b <= nVariables - 3; b++) { /* -3 to leave room for c and d */
                        std::cerr << (uint32_t)b << ", ";
                        for (uint8_t c = b + 1; c <= nVariables - 2; c++) { /* -2 to leave room for d */
                                for (uint8_t d = c + 1; d <= nVariables - 1; d++) {

                                        for (uint8_t e = 1; e <= nVariables - 2; e++) {
                                                for (uint8_t f = e + 1; f <= nVariables - 1; f++) {

                                                        Function_0_a_b_c_d_ef func(a, b, c, d, e, f, nVariables);

                                                        #pragma omp task shared(maxNLFSR)
                                                        {
                                                                if (func.is_canonical()) {
                                                                        func.compute_cycle_length();
                                                                        #pragma omp critical
                                                                        maxNLFSR.push_back(func);
                                                                }
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
#pragma omp taskwait

        // Sort the functions by cycle length and print the longest ones
        std::sort(maxNLFSR.begin(), maxNLFSR.end(), compare_funcs<Function_0_a_b_c_d_ef>);

        std::vector<Function_0_a_b_c_d_ef>::iterator it = maxNLFSR.begin();
        for (int i = 0; i < N_FUNCS_TO_REPORT && it != maxNLFSR.end(); i++, it++) {
                it->print();
        }
}







/**********************************************************************************************
 ********************************* Form x0 + a + b + cde **************************************
 **********************************************************************************************/


class Function_0_a_b_cde {
        public:
                uint8_t a, b, c, d, e;
                uint8_t nVariables;
                uint32_t m_length;

                Function_0_a_b_cde(uint8_t _a, uint8_t _b, uint8_t _c, uint8_t _d, uint8_t _e, uint8_t _nVariables)
                        : a(_a), b(_b), c(_c), d(_d), e(_e), nVariables(_nVariables), m_length(0)
                {}


                Function_0_a_b_cde(const Function_0_a_b_cde& other)
                        : a(other.a), b(other.b), c(other.c), d(other.d), e(other.e),
                          nVariables(other.nVariables), m_length(other.m_length)
                {}

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
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;
                        uint8_t er = nVariables - e;

                        Function_0_a_b_cde other(br, ar, er, dr, cr, nVariables);
                        return smaller_or_equal(other);
                }


                uint32_t compute_cycle_length()
                {
                        if (m_length != 0) {
                                return m_length;
                        }
                        uint32_t length = 0;
                        uint32_t newBit;
                        uint32_t curVal = 1;

                        do {
                                newBit = bit(0, curVal) ^ bit(a, curVal) ^ bit(b, curVal) ^
                                        (bit(c, curVal) & bit(d, curVal) & bit(e, curVal));

                                curVal = (curVal >> 1) | (newBit << (nVariables - 1));

                                length++;
                        } while (curVal != 1);

                        m_length = length;
                        return length;
                }


                void print() const
                {
                        std::cout << (uint32_t) nVariables << " variables: "
                                  << 0 << "," << (uint32_t) a  << "," << (uint32_t) b
                                  << ",(" << (uint32_t) c << "," << (uint32_t) d << "," << (uint32_t) e << ")"
                                  << ", cycle length: " << m_length << ", max poss length: " << (1 << nVariables) - 1
                                  << std::endl;
                }

};



template <>
void report<Function_0_a_b_cde>(uint8_t nVariables)
{
        std::vector<Function_0_a_b_cde> maxNLFSR;

        /* Generate the functions */
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";
                for (uint8_t b = a + 1; b <= nVariables - 1; b++) {
                        std::cerr << (uint32_t)b << ", ";


                        for (uint8_t c = 1; c <= nVariables - 3; c++) { /* -3 to leave room for d and e */
                                for (uint8_t d = c + 1; d <= nVariables - 2; d++) {
                                        for (uint8_t e = d + 1; e <= nVariables - 1; e++) {

                                                Function_0_a_b_cde func(a, b, c, d, e, nVariables);

                                                #pragma omp task shared(maxNLFSR)
                                                {
                                                        if (func.is_canonical()) {
                                                                func.compute_cycle_length();
                                                                #pragma omp critical
                                                                maxNLFSR.push_back(func);
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
#pragma omp taskwait
        
        // Sort the functions by cycle length and print the longest ones
        std::sort(maxNLFSR.begin(), maxNLFSR.end(), compare_funcs<Function_0_a_b_cde>);

        std::vector<Function_0_a_b_cde>::iterator it = maxNLFSR.begin();
        for (int i = 0; i < N_FUNCS_TO_REPORT && it != maxNLFSR.end(); i++, it++) {
                it->print();
        }
}




/*********************** Main and option-parsing related stuff **************************/


// 
// Long command-line option and their short equivalents
//
static const struct option longOpts[] = {
        {"n-vars", required_argument, NULL, 'n'},
        {"func-kind", required_argument, NULL, 'k'},
};

const char *shortOpts = "nk";

//
// Program global options
//
struct __globalOptions {
        uint32_t nVariables;
        std::string funcKind;
} globalOptions;


int main(int argc, char *argv[])
{
        globalOptions.nVariables = 0;
        globalOptions.funcKind = "";

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
                }

                // Next option
                opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
        }

        // Spawn worker threads, but make them wait for tasks to be generated
#pragma omp parallel
#pragma omp single
        {
                // Filter on the required type of function to test
                if (globalOptions.funcKind == "0_a_b_cd") {
                        report<Function_0_a_b_cd>(globalOptions.nVariables);
                } else if (globalOptions.funcKind == "0_a_bc_de") {
                        report<Function_0_a_bc_de>(globalOptions.nVariables);
                } else if (globalOptions.funcKind == "0_a_b_c_d_ef") {
                        report<Function_0_a_b_c_d_ef>(globalOptions.nVariables);
                } else if (globalOptions.funcKind == "0_a_b_cde") {
                        report<Function_0_a_b_cde>(globalOptions.nVariables);
                } else {
                        std::cerr << "Function kind '" << globalOptions.funcKind << "' not recognized" << std::endl;
                }
        }

        return 0;
}
