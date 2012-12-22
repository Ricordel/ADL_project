#include <vector>


#include "FuncGenerator.hpp"



template <class FuncGenerator>
void find_max_functions_for(FuncGenerator& generator, std::ostream& outStream);

void find_max_functions_of_size(uint32_t nVariables, std::ostream& outStream);
void find_max_functions(uint32_t minNVariables, uint32_t maxNVariavles, std::ostream& outStream);



int main(int argc, const char *argv[])
{
        (void)argc;
        (void)argv;

        FuncGenerator_0_a_b_cd gen(4);
        find_max_functions_for(gen, std::cout);

        return 0;
}



/* The main logic of the program:
 *      - instantiate function generators for the forms and number
 *        of variables we want to test
 *      - cycle through all functions of a given form
 *      - report our successes
 * This logic is implemented in the following functions.
 */


/* Parametric polymorphism over function generator types */
template <class FuncGenerator>
void find_max_functions_for(FuncGenerator& generator, std::ostream& outStream)
{
        bool moreFuncs = true;
        uint32_t maxLength = generator.getMaxPossibleLength();

        std::vector<Function *> maxFunctions;
        Function *pCurFunc;


        while (moreFuncs) {
                try {
                        /* Allocated on the heap, from now on we are responsible for it */
                        pCurFunc = generator.getNextFunction();

                        //XXX debug
                        std::cout << "Checking " << pCurFunc->toPrettyString() << std::endl;

                        //if (pCurFunc->getCycleLength() == maxLength) {
                        uint32_t l = pCurFunc->getCycleLength();
                        std::cout << "Length: " << l << std::endl << std::endl;

                        if (l == maxLength) {
                                maxFunctions.push_back(pCurFunc);
                        } else {
                                delete pCurFunc;
                        }

                } catch (NoMoreFunctionsException& e) {
                        moreFuncs = false;
                } /* Any other exception goes up */
        }

        // Report maximum functions in the log stream
        for (Function *func : maxFunctions) {
                outStream << func->toString() << std::endl;
                delete func;
        }
}


void find_max_functions_of_size(uint32_t nVariables, std::ostream& outStream)
{

        try {
                FuncGenerator_0_a_b_cd generator_0_a_b_cd(nVariables);
                find_max_functions_for(generator_0_a_b_cd, outStream);
        } catch (NotEnoughVariablesException& e) {
                std::cout << nVariables << " is not enough variables for for a + b + c.d" << std::endl;
        }

        /* Place here other forms of generators when time has come */
}


void find_max_functions(uint32_t minNVariables, uint32_t maxNVariavles, std::ostream& outStream)
{
        for (uint32_t nVariables = minNVariables; nVariables < maxNVariavles; nVariables++) {
                find_max_functions_of_size(nVariables, outStream);
        }
}
