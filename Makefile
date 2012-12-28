BE_SURE_BIN_EXISTS:=$(shell mkdir -p bin)

CUDACC = nvcc
CUDAFLAGS = -c -g

# Compiler options
ifndef WITHOUT_CPP11
CXXFLAGS = -c -O2 -Wall -Wextra -Isrc -DNDEBUG -pedantic -fopenmp -std=c++11 -flto $(OPTFLAGS)
LDFLAGS = -lgomp -flto
else
CXXFLAGS = -c -O2 -Wall -Wextra -Isrc -DNDEBUG -pedantic -fopenmp -DWITHOUT_CPP11 $(OPTFLAGS)
LDFLAGS = -lgomp
endif

LD = g++


# Directories
BIN = bin
SRCDIR = src


ifndef WITHOUT_CPP11
ALL=find_functions_omp find_functions_cuda print_function
else
ALL=find_functions_omp
endif

all: $(ALL)


# Targets
find_functions_omp: $(BIN)/Function.o $(BIN)/FuncGenerator_omp.o $(BIN)/find_functions.o
	$(LD) $(LDFLAGS) $^ -o $@

find_functions_cuda: CXXFLAGS += -D__CUDA
find_functions_cuda: $(BIN)/Function.o $(BIN)/FuncGenerator_cuda.o $(BIN)/find_functions.o
	$(LD) $(LDFLAGS) -lcudart $^ -o $@

print_function: $(BIN)/Function.o $(BIN)/print_function.o
	$(LD) $(LDFLAGS) $^ -o $@

$(BIN)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $< -o $@
	
$(BIN)/%.o: $(SRCDIR)/%.cu
	$(CUDACC) $(CUDAFLAGS) $< -o $@
	

# dev is all + debug options: -g, -O0, -DDEBUG
#XXX I get weird errors of undefined operator= in basic_string when compiling in O0
dev: CXXFLAGS = -c -O1 -g -Wall -Wextra -Isrc -std=c++11 -pedantic $(OPTFLAGS)
dev: CPPFLAGS += -DDEBUG
dev: all



clean:
	rm -f find_functions_omp find_functions_cuda print_function $(BIN)/*.o


tags:
	ctags -R


.PHONY: clean tags
