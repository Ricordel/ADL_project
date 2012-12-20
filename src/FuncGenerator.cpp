#include "FuncGenerator.hpp"



/********* Definition of public functions **********/

FuncGenerator_a_b_cd::FuncGenerator_a_b_cd(uint32_t nVariables)
        : m_nVariables(nVariables)
{}

FuncGenerator_a_b_cd::~FuncGenerator_a_b_cd() {}



Function_a_b_cd *FuncGenerator_a_b_cd::getNextFunction() throw (NoMoreFunctionsException)
{
        return NULL;
}
