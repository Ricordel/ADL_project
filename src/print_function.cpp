#include <unistd.h>
#include <getopt.h>
#include <string>

#include "Function.hpp"


// Option parsing definition using getopt_long()

static const struct option longOpts[] = {
        {"n-vars", required_argument, NULL, 'n'},
        {"func", required_argument, NULL, 'f'},
        {"func-kind", required_argument, NULL, 'k'},
        {"print-cycle", no_argument, NULL, 'p'},
        {"help", no_argument, NULL, 'h'}
};

const char *shortOpts = "nfph";



static struct {
        uint32_t nVariables;
        std::string funcString;
        std::string funcKind;
        bool printCycle;
} progOptions;



static void print_usage(char *argv[])
{
        std::cerr << "Usage: " << argv[0] << " --n-vars n_variables --func function_string "
                  << "--func-kind form_of_the_function [--print-cycle] [--help]" << std::endl;
}


static void print_help()
{
        std::cout << "Pretty print a \"canonically representated\" function along with"
                  << " some of its properties (cycle length, complete cycle transcript)" << std::endl
                  << std::endl
                  << "Mandatory options:" << std::endl
                  << "\t --n-vars     number of variables of the function" << std::endl
                  << "\t --func       'canonical' string representation of the function (ex: 0,1,(2,3))" << std::endl
                  << "\t --func-kind  form of the function (ex: 0_a_bc)" << std::endl
                  << std::endl
                  << "Optional options:" << std::endl
                  << "\t --print-cycle: print the whole cycle of the NLFSR." << std::endl
                  << "\t                NOTE: this can be very long for an important number of variables"
                  << std::endl;
}


int main(int argc, char **argv)
{
        // Init program options
        progOptions.nVariables = 0;
        progOptions.funcString = "";
        progOptions.printCycle = false;

        int longIndex;

        // Parse program options
        int opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
        while (opt != -1) {
                switch (opt) {
                        case 'n': std::cout << std::string(optarg) << std::endl;
                        
                                  progOptions.nVariables = std::stoi(std::string(optarg));
                                  break;
                        case 'f': progOptions.funcString = std::string(optarg);
                                  break;
                        case 'k': progOptions.funcKind = std::string(optarg);
                                  break;
                        case 'p': progOptions.printCycle = true;
                                  break;
                        case 'h': print_help();
                                  exit(0);
                                  break;
                        default:
                                  std::cerr << "Unrecognised option " << (char) opt << std::endl;
                                  print_usage(argv);
                                  exit(1);
                }
                opt = getopt_long(argc, argv, shortOpts, longOpts, &longIndex);
        }

        // Check we got all mandatory information
        if (progOptions.nVariables < 2 || progOptions.funcString == "") {
                print_usage(argv);
        }
        

        Function *func;

        // Now construct the function according to the right kind
        if (progOptions.funcKind == "0_a_b_cd") {
                func = new Function_0_a_b_cd(progOptions.funcString, progOptions.nVariables);
        } else if (progOptions.funcKind == "0_a_bc_de") {
                func = new Function_0_a_bc_de(progOptions.funcString, progOptions.nVariables);
        } else {
                std::cerr << "Function kind " << progOptions.funcKind << " not recognized" << std::endl;
                exit(0);
        }

        std::cout << "Function " << func->toPrettyString() << std::endl
                  << "\t cycle length: " << func->getCycleLength() << std::endl
                  << "\t max for " << progOptions.nVariables << ": " << (1 << progOptions.nVariables) - 1;

        if (progOptions.printCycle) {
                std::cout << std::endl << "\t Complete cycle: ";
                func->printCycle(std::cout);
        }

        std::cout << std::endl;
        
        delete func;

        return 0;
}
