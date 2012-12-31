#include <omp.h>

#include "FuncGenerator.hpp"




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



/***********************************************************************
 ********************* For x0 + xa + xb + xc.xd ************************
 ***********************************************************************/

FuncGenerator_0_a_b_cd::FuncGenerator_0_a_b_cd(uint32_t nVariables)
        : m_nVariables(nVariables), m_maxPossibleLength((1 << nVariables) - 1)
{}

FuncGenerator_0_a_b_cd::~FuncGenerator_0_a_b_cd() {}



void FuncGenerator_0_a_b_cd::reportMaxFunctions()
{
        // For the outermost loop, because of reverse functions, no need to go
        // funther than half the number of variables as long as the last variable
        // goes to the end.


        for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
                std::cerr << "a = " << a << std::endl;
                
                for (int32_t b = a + 1; b <= m_nVariables - 1; b++) {
                        std::cerr << "\t b = " << b << std::endl;

                        for (int32_t c = 1; c <= m_nVariables - 2; c++) {
                                for (int32_t d = c + 1; d <= m_nVariables - 1; d++) {

                                #pragma omp task firstprivate(a, b, c, d)
                                {
                                        Function_0_a_b_cd func(a, b, c, d, m_nVariables);

                                        if (func.isCanonicalForm()
                                         && func.getCycleLength() == m_maxPossibleLength) {
                                                #pragma omp critical
                                                std::cout << func.toString() << std::endl;
                                        }
                                } // end of task
                                }
                        }
                }
        }
        #pragma omp taskwait
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
        // The lexicographical order is difficult to handle in the generation
        // for b,c and d,e. So this will be handled in isCanonicalForm()

        // d can start from b, because if d < b, then a commutatively equivalent function
        // will have been tested (as b.c and d.e can commute around +), and that variant
        // would be smaller by lexicographical order.

        // We don't want b = c AND d = e either, which gives us kind of a "degenerated" function.
        // This is also handled in isCanonicalForm()

        for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
                std::cerr << "a = " << a << std::endl;

                for (int32_t b = 1; b <= m_nVariables - 2; b++) {
                        std::cerr << "\t b = " << b << std::endl;
                        for (int32_t c = b + 1; c <= m_nVariables - 1; c++) {

                                for (int32_t d = b; d <= m_nVariables - 2; d++) {
                                        for (int32_t e = d + 1; e <= m_nVariables - 1; e++) {

                                        #pragma omp task firstprivate(a, b, c, d, e)
                                        {
                                                Function_0_a_bc_de func(a, b, c, d, e, m_nVariables);
                                                
                                                if (func.isCanonicalForm()
                                                 && func.getCycleLength() == m_maxPossibleLength) {
                                                        #pragma omp critical
                                                        std::cout << func.toString() << std::endl;
                                                }
                                        } // end of task
                                        }
                                }
                        }
                }
        }
        #pragma omp taskwait
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

        for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
                std::cerr << "a = " << a << std::endl;
                for (int32_t b = a + 1; b <= m_nVariables - 3; b++) { /* -3 to leave room for c and d */
                        std::cerr << "\t b = " << b << std::endl;
                        for (int32_t c = b + 1; c <= m_nVariables - 2; c++) { /* -2 to leave room for d */
                                for (int32_t d = c + 1; d <= m_nVariables - 1; d++) {

                                        for (int32_t e = 1; e <= m_nVariables - 2; e++) {
                                                for (int32_t f = e + 1; f <= m_nVariables - 1; f++) {

                                                #pragma omp task firstprivate(a)
                                                {
                                                        Function_0_a_b_c_d_ef func(a, b, c, d, e, f, m_nVariables);

                                                        
                                                        if (func.isCanonicalForm()
                                                         && func.getCycleLength() == m_maxPossibleLength) {
                                                                #pragma omp critical
                                                                std::cout << func.toString() << std::endl;
                                                        }
                                                } // end of task
                                                }
                                        }
                                }
                        }
                }
        }

        #pragma omp taskwait
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

        for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
                std::cerr << "a = " << a << std::endl;
                for (int32_t b = a + 1; b <= m_nVariables - 1; b++) {
                        std::cerr << "\t b = " << b << std::endl;


                        for (int32_t c = 1; c <= m_nVariables - 3; c++) { /* -3 to leave room for d and e */
                                for (int32_t d = c + 1; d <= m_nVariables - 2; d++) {
                                        for (int32_t e = d + 1; e <= m_nVariables - 1; e++) {

                                        #pragma omp task firstprivate(a, b, c, d, e)
                                        {
                                                Function_0_a_b_cde func(a, b, c, d, e, m_nVariables);

                                                if (func.isCanonicalForm()
                                                 && func.getCycleLength() == m_maxPossibleLength) {
                                                        #pragma omp critical
                                                        std::cout << func.toString() << std::endl;
                                                }
                                        } // end of task
                                        }
                                }
                        }
                }
        }
        #pragma omp taskwait
}
