# Compilation is fast, don't bother with dependancies, recompile everything
# everytime

CXX = g++
CXXFLAGS = -Wall -Wextra -O3 -fopenmp


all: omp_version cuda_version

omp_version: omp_version.cpp
	$(CXX) $(CXXFLAGS) $< -o $@

cuda_version: cuda_version.cu
	nvcc -lcudart -o cuda_version cuda_version.cu

clean:
	rm -f omp_version cuda_version


.PHONY: clean omp_version cuda_version
