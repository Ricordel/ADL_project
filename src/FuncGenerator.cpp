#include <omp.h>

#include "FuncGenerator.hpp"



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


#pragma omp parallel for
        for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
                for (int32_t b = a + 1; b <= m_nVariables - 1; b++) {

                        for (int32_t c = 1; c <= m_nVariables - 2; c++) {
                                for (int32_t d = c + 1; d <= m_nVariables - 1; d++) {

                                        Function_0_a_b_cd func(a, b, c, d, m_nVariables);

                                        if (func.isCanonicalForm()
                                         && func.getCycleLength() == this->getMaxPossibleLength()) {
                                                #pragma omp critical
                                                std::cout << func.toString() << std::endl;
                                        }
                                }
                        }
                }
        }
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
        // For the outermost loop, because of reverse functions, no need to go
        // funther than half the number of variables as long as the last variable
        // goes to the end.

        // d can start from b, because if d < b, then a commutatively equivalent function
        // will have been tested (as b.c and d.e can commute around +), and that variant
        // would be smaller by lexicographical order.

        //XXX vérifier que la génération est bien "dans l'ordre"
#pragma omp parallel for
        for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {

                for (int32_t b = 1; b <= m_nVariables - 2; b++) {
                        for (int32_t c = b + 1; c <= m_nVariables - 1; c++) {

                                for (int32_t d = b; d <= m_nVariables - 2; d++) {
                                        for (int32_t e = d + 1; e <= m_nVariables - 1; e++) {

                                                Function_0_a_bc_de func(a, b, c, d, e, m_nVariables);
                                                
                                                if (func.isCanonicalForm()
                                                 && func.getCycleLength() == this->getMaxPossibleLength()) {
                                                        #pragma omp critical
                                                        std::cout << func.toString() << std::endl;
                                                }
                                        }
                                }
                        }
                }
        }
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
        // For the outermost loop, because of reverse functions, no need to go
        // funther than half the number of variables as long as the last variable
        // goes to the end.


#pragma omp parallel for
        for (int32_t a = 1; a <= (m_nVariables + 1) / 2; a++) {
                for (int32_t b = a + 1; b <= m_nVariables - 1; b++) {
                        for (int32_t c = b + 1; c <= m_nVariables - 1; c++) {
                                for (int32_t d = c + 1; d <= m_nVariables - 2; d++) {

                                        for (int32_t e = 1; e <= m_nVariables - 2; e++) {
                                                for (int32_t f = e + 1; e <= m_nVariables - 1; e++) {

                                                        Function_0_a_b_c_d_ef func(a, b, c, d, e, f, m_nVariables);
                                                        
                                                        if (func.isCanonicalForm()
                                                         && func.getCycleLength() == this->getMaxPossibleLength()) {
                                                                #pragma omp critical
                                                                std::cout << func.toString() << std::endl;
                                                        }
                                                }
                                        }
                                }
                        }
                }
        }
}

