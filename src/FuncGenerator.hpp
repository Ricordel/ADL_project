#ifndef __FUNCGENERATOR_H__
#define __FUNCGENERATOR_H__

#include <fstream>

#include "Function.hpp"

/* 
 * This is defining the interface a function generator must implement.
 * Basically, a function generator must be able to iterate through
 * all the functions of a given form (defined by the concrete type
 * of the generator) for a given number of variable (specified at
 * instantiation time).
 */

class NoMoreFunctionsException
{
        public:
                NoMoreFunctionsException() {}
};


/* Exception to be thrown when trying to construct a generator with
 * too few variables */
class NotEnoughVariablesException
{
        public:
                NotEnoughVariablesException() {}
};




/*
 * Generator of functions of the form 
 *      a + b + c.d
 * for a number of variables specified at instanciation time
 */
class FuncGenerator_0_a_b_cd
{
        public:
                /* Construct a function generator of form a + b + c.d over nVariables
                 * different variables */
                FuncGenerator_0_a_b_cd(uint32_t nVariables);
                ~FuncGenerator_0_a_b_cd();


                /* Returns the maximum possible length of a cycle for this kind of function
                 * and this number of variables. This is 2^n - 1 where n is the number of
                 * variables. */
                inline uint32_t getMaxPossibleLength() const
                {
                        return m_maxPossibleLength;
                }

                /* Generate and report functions of maximum cycle length */
                void reportMaxFunctions();


        private:
                int32_t m_nVariables;
                uint32_t m_maxPossibleLength;
};




class FuncGenerator_0_a_bc_de
{
        public:
                /* Construct a function generator of form a + b + c.d over nVariables
                 * different variables */
                FuncGenerator_0_a_bc_de(uint32_t nVariables);
                ~FuncGenerator_0_a_bc_de();

                /* This functions returns the next Function instance belonging to the form
                 * a + b + c.d for nVariables number of variables.
                 * Throws a NoMoreFunctionsException if all possible functions (without
                 * symetric) have been seen.
                 */
                //Function_0_a_bc_de *getNextFunction() throw (NoMoreFunctionsException);

                /* Returns the maximum possible length of a cycle for this kind of function
                 * and this number of variables. This is 2^n - 1 where n is the number of
                 * variables. */
                inline uint32_t getMaxPossibleLength() const
                {
                        return m_maxPossibleLength;
                }

                /* Generate and report functions of maximum cycle length */
                void reportMaxFunctions();


        private:
                int32_t m_nVariables;
                uint32_t m_maxPossibleLength;
};



class FuncGenerator_0_a_b_c_d_ef
{
        public:
                /* Construct a function generator of form a + b + c.d over nVariables
                 * different variables */
                FuncGenerator_0_a_b_c_d_ef(uint32_t nVariables);
                ~FuncGenerator_0_a_b_c_d_ef();

                /* This functions returns the next Function instance belonging to the form
                 * a + b + c.d for nVariables number of variables.
                 * Throws a NoMoreFunctionsException if all possible functions (without
                 * symetric) have been seen.
                 */
                //Function_0_a_bc_de *getNextFunction() throw (NoMoreFunctionsException);

                /* Returns the maximum possible length of a cycle for this kind of function
                 * and this number of variables. This is 2^n - 1 where n is the number of
                 * variables. */
                inline uint32_t getMaxPossibleLength() const
                {
                        return m_maxPossibleLength;
                }

                /* Generate and report functions of maximum cycle length */
                void reportMaxFunctions();


        private:
                int32_t m_nVariables;
                uint32_t m_maxPossibleLength;
};



#endif /* end of include guard: __FUNCGENERATOR_H__ */
