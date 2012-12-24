#include <vector>

#include "dbg.h"
#include "FuncGenerator.hpp"


/*
 * Returns maximum cycle-length functions for different forms and
 * different number of variables.
 */
void report_max_functions(uint32_t minNVariables, uint32_t maxNVariavles);


/*
 * There is no covariance in C++, preventing me from having a superclass
 * for all generators (or at least I don't see how, as different generators
 * of the hierarchy must have different subtypes of Function, which requires
 * variance in heritage). So I use templates to replace.
 */
template <class FuncGenerator>
void report_max_functions_for_generator(FuncGenerator& generator);



/*
 * This is more for debug. It prints the "pretty print" form of functions along with
 * their complete cycle.
 * NOTE: this can be VERY long with big functions !
 */
void print_details(std::vector<Function *> maxFunctions, std::ostream& outStream);



int main(int argc, const char *argv[])
{
        (void)argc;
        (void)argv;

        report_max_functions(4, 13);

        return 0;
}



void report_max_functions(uint32_t minNVariables, uint32_t maxNVariavles)
{

        for (uint32_t nVariables = minNVariables; nVariables <= maxNVariavles; nVariables++) {

                std::cerr << std::endl
                          << "========= Testing for " << nVariables << " variables =========="
                          << std::endl;

                try {
                        std::cerr << std::endl << "*** Checking functions "
                                  << "x0 + xa + xb + xc.xd for " << nVariables << " variables ***"
                                  << std::endl;
                        FuncGenerator_0_a_b_cd generator_0_a_b_cd(nVariables);
                        report_max_functions_for_generator(generator_0_a_b_cd);

                } catch (NotEnoughVariablesException& e) {
                        std::cerr << nVariables << " is not enough variables for for a + b.c + d.e" << std::endl;
                }


                try {
                        std::cerr << std::endl << "*** Checking functions "
                                  << "x0 + xa + xb.xc + xd.xe for " << nVariables << " variables ***"
                                  << std::endl;
                        FuncGenerator_0_a_bc_de generator_0_a_bc_de(nVariables);
                        report_max_functions_for_generator(generator_0_a_bc_de);

                } catch (NotEnoughVariablesException& e) {
                        std::cerr << nVariables << " is not enough variables for for a + b + c.d" << std::endl;
                }


                // Place other generators for other kinds of functions here
        }
}


/*
 * There is no covariance in C++, preventing me from having a superclass
 * for all generators (or at least I don't see how, as different generators
 * of the hierarchy must have different subtypes of Function, which requires
 * variance in heritage). So I use templates to replace.
 */
template <class FuncGenerator>
void report_max_functions_for_generator(FuncGenerator& generator)
{
        bool moreFuncs = true;
        uint32_t maxLength = generator.getMaxPossibleLength();

        Function *pCurFunc;

        while (moreFuncs) {
                try {
                        /* Allocated on the heap, from now on we are responsible for it */
                        pCurFunc = generator.getNextFunction();

                        debug("Checking %s", pCurFunc->toPrettyString().c_str());
                        uint32_t l = pCurFunc->getCycleLength();
                        debug("Length : %u\n", l);

                        if (l == maxLength) {
#ifndef NDEBUG
                                std::cerr << std::endl << "Found a max function" << std::endl;
                                pCurFunc->printDetails(std::cerr);
#endif
                                std::cout << pCurFunc->toString() << std::endl;
                        }

                        delete pCurFunc;

                } catch (NoMoreFunctionsException& e) {
                        moreFuncs = false;
                } /* Any other exception goes up */
        }
}




void print_details(std::vector<Function *> maxFunctions, std::ostream& outStream)
{
        outStream << std::endl;
        for (auto pFunc : maxFunctions) {
                pFunc->printDetails(outStream);
                outStream << std::endl << std::endl;
        }
}
