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

//XXX pour éviter la merde avec le polymorphisme bancale de C++, ne créons pas d'interface.
//XXX de toutes façons, y a pas grand chose à abstraire dedans. Ça aurait été mieux, mais
//XXX on fera sans.
#if 0
class FuncGenerator
{
        public:
                /* Construct a NLFSR function of the given form (depending on the derived
                 * class) with nVariables different variables */
                FuncGenerator(uint32_t nVariables);

                /* This returns the next function of the form given by the concrete type
                 * and the number of variables. Must return a pointer so that late
                 * binding on the function type can be made*/
                virtual Function * getNextFunction() throw NoMoreFunctionsException = 0;


        protected:
                /* Number of variables in the NLFSR */
                uint32_t m_nVariables;

                /* Next function to return */
                uint32_t
};
#endif




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

                /* This functions returns the next Function instance belonging to the form
                 * a + b + c.d for nVariables number of variables.
                 * Throws a NoMoreFunctionsException if all possible functions (without
                 * symetric) have been seen.
                 */
                Function_0_a_b_cd *getNextFunction() throw (NoMoreFunctionsException);

                /* Returns the maximum possible length of a cycle for this kind of function
                 * and this number of variables. This is 2^n - 1 where n is the number of
                 * variables. */
                inline uint32_t getMaxPossibleLength() const
                {
                        return (1 << m_nVariables) - 1;
                }




        private:
                int32_t m_nVariables;
                Function_0_a_b_cd m_curFunc;

                void generate_next_function();
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
                Function_0_a_bc_de *getNextFunction() throw (NoMoreFunctionsException);

                /* Returns the maximum possible length of a cycle for this kind of function
                 * and this number of variables. This is 2^n - 1 where n is the number of
                 * variables. */
                inline uint32_t getMaxPossibleLength() const
                {
                        return (1 << m_nVariables) - 1;
                }




        private:
                int32_t m_nVariables;
                Function_0_a_bc_de m_curFunc;

                void generate_next_function();
};



#endif /* end of include guard: __FUNCGENERATOR_H__ */
