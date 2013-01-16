# Compilation is fast, don't bother with dependancies, recompile everything
# everytime

CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp -g


all: omp_version cuda_version print_function

omp_version: omp_version.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

cuda_version: cuda_version.cu
	nvcc -lcudart -o cuda_version cuda_version.cu

print_function: Function.cpp Function.hpp print_function.cpp
	$(CXX) $(CXXFLAGS) -std=c++11 print_function.cpp Function.cpp -o $@

clean:
	rm -f omp_version cuda_version print_function


.PHONY: clean omp_version cuda_version
