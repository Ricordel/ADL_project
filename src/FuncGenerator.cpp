#include "FuncGenerator.hpp"


/***********************************************************************
 ********************* For x0 + xa + xb + xc.xd ************************
 ***********************************************************************/

FuncGenerator_0_a_b_cd::FuncGenerator_0_a_b_cd(uint32_t nVariables)
        : m_nVariables(nVariables)
{
        m_curFunc.m_a = -1;
        m_curFunc.m_b = -1;
        m_curFunc.m_c = -1;
        m_curFunc.m_d = -1;
        m_curFunc.m_nVariables = nVariables;
}

FuncGenerator_0_a_b_cd::~FuncGenerator_0_a_b_cd() {}



/*
 * We must be careful for performance and correctness purposes not to try
 * duplicate functions.
 * For this, we ask the function if it is in canonical form. See Function.cpp
 * for full details on canonical forms.
 */
Function_0_a_b_cd *FuncGenerator_0_a_b_cd::getNextFunction() throw (NoMoreFunctionsException)
{
        do {
                generate_next_function(); /* May throw the NoMoreFunctionsException */
        } while (!m_curFunc.isCanonicalForm());

        Function_0_a_b_cd *func = new Function_0_a_b_cd(m_curFunc);
        return func;
}



/*
 * When generating functions, we try to avoid some easy non-canonical forms
 */
void FuncGenerator_0_a_b_cd::generate_next_function()
{
        /* This will go very far to the right side, so I'll return early instead
         * of using lots of "else" */


        /* If it's the first time we come here */
        if (m_curFunc.m_a == -1) {
                m_curFunc.m_a = 1;
                m_curFunc.m_b = 2;
                m_curFunc.m_c = 1;
                m_curFunc.m_d = 2;
                return;
        }


        /* Try incrementing D */
        if (m_curFunc.m_d < m_nVariables - 1) {
                m_curFunc.m_d++;
                return;
        }

        /* Try to increment C and hence reset D. We must leave some room
         * to D, so start at nVariables - 2 */
        if (m_curFunc.m_c < m_nVariables - 2) {
                m_curFunc.m_c++;
                m_curFunc.m_d = m_curFunc.m_c + 1;
                return;
        }

        /* Try to increment B and reset C and D */
        if (m_curFunc.m_b < m_nVariables - 1) {
                m_curFunc.m_b++;
                m_curFunc.m_c = 1;
                m_curFunc.m_d = 2;
                return;
        }

        /* Try to increment A, and reset the rest. As for C, we must
         * leave room for B, so -2 */
        if (m_curFunc.m_a < m_nVariables - 2) {
                m_curFunc.m_a++;
                m_curFunc.m_b = m_curFunc.m_a + 1;
                m_curFunc.m_c = 1;
                m_curFunc.m_d = 2;
                return;
        }

        /* If we gen here, it means we have exhausted all functions */
        throw NoMoreFunctionsException();
}





/***********************************************************************
 ******************** For x0 + xa + xb.xc + xd.xe **********************
 ***********************************************************************/

FuncGenerator_0_a_bc_de::FuncGenerator_0_a_bc_de(uint32_t nVariables)
        : m_nVariables(nVariables)
{
        m_curFunc.m_a = -1;
        m_curFunc.m_b = -1;
        m_curFunc.m_c = -1;
        m_curFunc.m_d = -1;
        m_curFunc.m_e = -1;
        m_curFunc.m_nVariables = nVariables;
}

FuncGenerator_0_a_bc_de::~FuncGenerator_0_a_bc_de() {}



/*
 * We must be careful for performance and correctness purposes not to try
 * duplicate functions.
 * For this, we ask the function if it is in canonical form. See Function.cpp
 * for full details on canonical forms.
 */
Function_0_a_bc_de *FuncGenerator_0_a_bc_de::getNextFunction() throw (NoMoreFunctionsException)
{
        do {
                generate_next_function(); /* May throw the NoMoreFunctionsException */
        } while (!m_curFunc.isCanonicalForm());

        Function_0_a_bc_de *func = new Function_0_a_bc_de(m_curFunc);
        return func;
}



/*
 * When generating functions, we try to avoid some easy non-canonical forms,
 * for instance some commutativity repetitions can be removed by ensuring
 * that c > b and d > e at any time.
 */
void FuncGenerator_0_a_bc_de::generate_next_function()
{
        /* This will go very far to the right side, so I'll return early instead
         * of using lots of "else" */


        /* If it's the first time we come here */
        if (m_curFunc.m_a == -1) {
                m_curFunc.m_a = 1;

                m_curFunc.m_b = 1;
                m_curFunc.m_c = 2;

                m_curFunc.m_d = 1;
                m_curFunc.m_e = 2;
                return;
        }


        /* Try incrementing E */
        if (m_curFunc.m_e < m_nVariables - 1) {
                m_curFunc.m_e++;
                return;
        }

        /* Try to increment D, then reset E */
        if (m_curFunc.m_d < m_nVariables - 2) {
                m_curFunc.m_d++;
                m_curFunc.m_e = m_curFunc.m_d + 1;
                return;
        }

        /* Try to increment C and reset D and E */
        if (m_curFunc.m_c < m_nVariables - 1) {
                m_curFunc.m_c++;
                m_curFunc.m_d = 1;
                m_curFunc.m_e = 2;
                return;
        }

        /* Try to increment B, and reset C, D, and E */
        if (m_curFunc.m_b < m_nVariables - 2) {
                m_curFunc.m_b++;
                m_curFunc.m_c = m_curFunc.m_b + 1;
                m_curFunc.m_d = 1;
                m_curFunc.m_e = 2;
                return;
        }

        /* Try to increment A, and reset the rest */
        if (m_curFunc.m_a < m_nVariables - 1) {
                m_curFunc.m_a++;
                m_curFunc.m_b = 1;
                m_curFunc.m_c = 2;
                m_curFunc.m_d = 1;
                m_curFunc.m_e = 2;
                return;
        }

        /* If we gen here, it means we have exhausted all functions */
        throw NoMoreFunctionsException();
}
