#include <stdexcept>
#include <vector>
#include <iostream>
#include <cstdlib>

#include <stdint.h>
#include <getopt.h>
#include <errno.h>


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
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";
                for (uint8_t b = a + 1; b <= nVariables - 1; b++) {
                        std::cerr << (uint32_t)b << ", ";

                        for (uint8_t c = 1; c <= nVariables - 2; c++) {
                                for (uint8_t d = c + 1; d <= nVariables - 1; d++) {

                                        // Keep the function for later evaluation
                                        Function_0_a_b_cd func = {a, b, c, d, nVariables};

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

#pragma omp taskwait
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
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;
                        uint8_t er = nVariables - e;

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
//#pragma omp task untied
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";

                for (uint8_t b = 1; b <= nVariables - 2; b++) {
                        std::cerr << (uint32_t)b << ", ";
                        for (uint8_t c = b + 1; c <= nVariables - 1; c++) {

                                for (uint8_t d = b; d <= nVariables - 2; d++) {
                                        for (uint8_t e = d + 1; e <= nVariables - 1; e++) {

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
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;
                        uint8_t er = nVariables - e;
                        uint8_t fr = nVariables - f;

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
//#pragma omp task untied
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";
                for (uint8_t b = a + 1; b <= nVariables - 3; b++) { /* -3 to leave room for c and d */
                        std::cerr << (uint32_t)b << ", ";
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
                        uint8_t ar = nVariables - a;
                        uint8_t br = nVariables - b;
                        uint8_t cr = nVariables - c;
                        uint8_t dr = nVariables - d;
                        uint8_t er = nVariables - e;

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
//#pragma omp task untied
        for (uint8_t a = 1; a <= (nVariables + 1) / 2; a++) {
                std::cerr << std::endl << "a = " << (uint32_t)a << std::endl << "\t b = ";
                for (uint8_t b = a + 1; b <= nVariables - 1; b++) {
                        std::cerr << (uint32_t)b << ", ";


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




/*********************** Main and option-parsing related stuff **************************/


/* 
 * Command-line option
 */
static const struct option longOpts[] = {
        {"n-vars", required_argument, NULL, 'n'},
        {"func-kind", required_argument, NULL, 'k'},
};

const char *shortOpts = "nk";

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

                // Get next option
                opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
        }

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
                        std::cerr << "Function kind " << globalOptions.funcKind << " not recognized" << std::endl;
                }
        }

        return 0;
}
