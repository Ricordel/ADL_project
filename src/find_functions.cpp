#include <vector>
#include <getopt.h>
#include <omp.h>
#include <stdint.h>

#ifdef WITHOUT_CPP11
#include <cstdlib>
#endif


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
 * Command-line option
 */
static const struct option longOpts[] = {
        {"from", required_argument, NULL, 'f'},
        {"to", required_argument, NULL, 't'},
        {"n-vars", required_argument, NULL, 'n'}
};

const char *shortOpts = "ft";

struct {
        uint32_t from;
        uint32_t to;
} globalOptions;


int main(int argc, char *argv[])
{
        globalOptions.from = 0;
        globalOptions.to   = 0;

        int longIndex; /* unused, but necessary for getopt */

        int opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
        while (opt != -1) {
                switch(opt) {
#ifndef WITHOUT_CPP11
                        case 'f': globalOptions.from = std::stoi(std::string(optarg));
                                  break;
                        case 't': globalOptions.to = std::stoi(std::string(optarg));
                                  break;
                        case 'n': globalOptions.from = std::stoi(std::string(optarg));
                                  globalOptions.to = std::stoi(std::string(optarg));
                                  break;
#else
                        case 'f': globalOptions.from = atoi(optarg);
                                  break;
                        case 't': globalOptions.to = atoi(optarg);
                                  break;
                        case 'n': globalOptions.from = atoi(optarg);
                                  globalOptions.to = atoi(optarg);
                                  break;
#endif
                        default:
                                  std::cerr << "Default case should not be reached" << std::endl;
                                  exit(1);
                }

                opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
        }
                                  
        if (globalOptions.from == 0 || globalOptions.to == 0) {
                std::cerr << "Usage: " << std::endl
                          << argv[0] << " --from start_num_variables --to end_num_variables" << std::endl
                          << argv[0] << " --n-vars num_variables" << std::endl;
                exit(1);
        }

        if (globalOptions.from > globalOptions.to) {
                std::cerr << "'to' must be larger or equal to 'from'" << std::endl;
                exit(1);
        }



#pragma omp parallel
        {
#pragma omp single
        report_max_functions(globalOptions.from, globalOptions.to);
        }

        return 0;
}


template <class FuncGenerator> void report_for_generator(uint32_t nVariables)
{
                FuncGenerator generator(nVariables);

                std::cerr << std::endl << "*** Checking functions "
                        << generator.stringForm() << " for " << nVariables << " variables ***"
                        << std::endl;
                generator.reportMaxFunctions();
}



void report_max_functions(uint32_t minNVariables, uint32_t maxNVariavles)
{

        for (uint32_t nVariables = minNVariables; nVariables <= maxNVariavles; nVariables++) {

                std::cerr << std::endl
                          << "========= Testing for " << nVariables << " variables =========="
                          << std::endl;

                report_for_generator<FuncGenerator_0_a_b_cd>(nVariables);
                report_for_generator<FuncGenerator_0_a_bc_de>(nVariables);
                report_for_generator<FuncGenerator_0_a_b_c_d_ef>(nVariables);
                report_for_generator<FuncGenerator_0_a_b_cde>(nVariables);

                // Place other generators for other kinds of functions here
        }

}
