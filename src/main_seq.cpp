#include <vector>

#include "dbg.h"
#include "FuncGenerator.hpp"


/*
 * Returns maximum cycle-length functions for different forms and
 * different number of variables.
 */
std::vector<Function *> max_functions(uint32_t minNVariables, uint32_t maxNVariavles);


/*
 * There is no covariance in C++, preventing me from having a superclass
 * for all generators (or at least I don't see how, as different generators
 * of the hierarchy must have different subtypes of Function, which requires
 * variance in heritage). So I use templates to replace.
 */
template <class FuncGenerator>
std::vector<Function *> max_functions_for_generator(FuncGenerator& generator);


/*
 * This function prints the function in "normal form" (that is not pretty printing)
 * to the given stream
 */
void print_functions(std::vector<Function *> maxFunctions, std::ostream& outStream);


/*
 * This is more for debug. It prints the "pretty print" form of functions along with
 * their complete cycle.
 * NOTE: this can be VERY long with big functions !
 */
void print_details(std::vector<Function *> maxFunctions, std::ostream& outStream);



#if 0
template <class FuncGenerator>
void find_max_functions_for(FuncGenerator& generator, std::ostream& outStream);

void find_max_functions_of_size(uint32_t nVariables, std::ostream& outStream);
void find_max_functions(uint32_t minNVariables, uint32_t maxNVariavles, std::ostream& outStream);
#endif



int main(int argc, const char *argv[])
{
        (void)argc;
        (void)argv;

        std::vector<Function *> maxFunctions;

        FuncGenerator_0_a_b_cd gen(4);
        maxFunctions = max_functions_for_generator(gen);

        print_functions(maxFunctions, std::cout);

        print_details(maxFunctions, std::cerr);

        return 0;
}



std::vector<Function *> max_functions(uint32_t minNVariables, uint32_t maxNVariavles)
{
        std::vector<Function *> maxFunctions;

        for (uint32_t nVariables = minNVariables; nVariables <= maxNVariavles; nVariables++) {
                try {
                        FuncGenerator_0_a_b_cd generator_0_a_b_cd(nVariables);
                        std::vector<Function *> maxFuncsForGenerator =
                                max_functions_for_generator(generator_0_a_b_cd);

                        maxFunctions.insert(maxFunctions.end(), maxFuncsForGenerator.begin(),
                                                                maxFuncsForGenerator.end());

                } catch (NotEnoughVariablesException& e) {
                        std::cerr << nVariables << " is not enough variables for for a + b + c.d" << std::endl;
                }

                // Place other generators for other kinds of functions here
        }

        return maxFunctions;
}


/*
 * There is no covariance in C++, preventing me from having a superclass
 * for all generators (or at least I don't see how, as different generators
 * of the hierarchy must have different subtypes of Function, which requires
 * variance in heritage). So I use templates to replace.
 */
template <class FuncGenerator>
std::vector<Function *> max_functions_for_generator(FuncGenerator& generator)
{
        bool moreFuncs = true;
        uint32_t maxLength = generator.getMaxPossibleLength();

        std::vector<Function *> maxFunctions;
        Function *pCurFunc;

        while (moreFuncs) {
                try {
                        /* Allocated on the heap, from now on we are responsible for it */
                        pCurFunc = generator.getNextFunction();

                        debug("Checking %s", pCurFunc->toPrettyString().c_str());
                        uint32_t l = pCurFunc->getCycleLength();
                        debug("Length : %u\n", l);

                        if (l == maxLength) {
                                maxFunctions.push_back(pCurFunc);
                        } else {
                                delete pCurFunc;
                        }

                } catch (NoMoreFunctionsException& e) {
                        moreFuncs = false;
                } /* Any other exception goes up */
        }

        return maxFunctions;
}



void print_functions(std::vector<Function *> maxFunctions, std::ostream& outStream)
{
        for (auto pFunc : maxFunctions) {
                outStream << pFunc->toString() << std::endl;
        }
}



void print_details(std::vector<Function *> maxFunctions, std::ostream& outStream)
{
        outStream << std::endl;
        for (auto pFunc : maxFunctions) {
                outStream << pFunc->toPrettyString() << std::endl << "\t";
                pFunc->printCycle(outStream);
                outStream << std::endl << std::endl;
        }
}



#if 0


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

                        std::cerr << "Checking " << pCurFunc->toPrettyString() << std::endl;

                        uint32_t l = pCurFunc->getCycleLength();
                        std::cerr << "Length: " << l << std::endl << std::endl;

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
                std::cerr << nVariables << " is not enough variables for for a + b + c.d" << std::endl;
        }

        /* Place here other forms of generators when time has come */
}


void find_max_functions(uint32_t minNVariables, uint32_t maxNVariavles, std::ostream& outStream)
{
        for (uint32_t nVariables = minNVariables; nVariables < maxNVariavles; nVariables++) {
                find_max_functions_of_size(nVariables, outStream);
        }
}

#endif
