# Compiler options
ifndef WITHOUT_CPP11
CXXFLAGS = -c -O3 -Wall -Wextra -Isrc -DNDEBUG -pedantic -fopenmp -std=c++11 -flto $(OPTFLAGS)
LDFLAGS = -lgomp -flto
else
CXXFLAGS = -c -O3 -Wall -Wextra -Isrc -DNDEBUG -pedantic -fopenmp -DWITHOUT_CPP11 $(OPTFLAGS)
LDFLAGS = -lgomp
endif

LD = g++

# Project name

# Directories
OBJDIR = bin
SRCDIR = src


# Files and folders
SRCDIRS     = $(shell find $(SRCDIR) -type d | sed 's/$(SRCDIR)/./g' )
CFILES      = $(shell find $(SRCDIR) -name '*.c')
CPPFILES    = $(shell find $(SRCDIR) -name '*.cpp')
OBJSFROMC   = $(patsubst $(SRCDIR)/%.c, $(OBJDIR)/%.o, $(CFILES))
OBJSFROMCPP = $(patsubst $(SRCDIR)/%.cpp, $(OBJDIR)/%.o, $(CPPFILES))
OBJS        = $(OBJSFROMC) $(OBJSFROMCPP)
DEPFILES    = $(patsubst %.o, %.d, $(OBJS))

# Objs common to finder and function printer
COMMON_OBJS = $(OBJDIR)/Function.o

ifndef WITHOUT_CPP11
ALL=find_functions print_function
else
ALL=find_functions
endif

all: $(ALL)


# Targets
find_functions: depends buildrepo $(COMMON_OBJS) $(OBJDIR)/FuncGenerator.o $(OBJDIR)/find_functions.o
	$(LD) $(LDFLAGS) $(COMMON_OBJS) $(OBJDIR)/FuncGenerator.o $(OBJDIR)/find_functions.o -o $@

print_function: depends buildrepo $(COMMON_OBJS) $(OBJDIR)/print_function.o
	$(LD) $(LDFLAGS) $(COMMON_OBJS) $(OBJDIR)/print_function.o -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(OPTS) -c $< -o $@
	

# dev is all + debug options: -g, -O0, -DDEBUG
#XXX I get weird errors of undefined operator= in basic_string when compiling in O0
dev: CXXFLAGS = -c -O1 -g -Wall -Wextra -Isrc -pedantic $(OPTFLAGS)
dev: CPPFLAGS += -DDEBUG
dev: all



depends: buildrepo
	@./scripts/make_depends.sh $(SRCDIR) $(OBJDIR)


clean:
	rm -f find_functions print_function $(OBJDIR) -Rf


tags:
	ctags -R


.PHONY: clean depends buildrepo depends test tags
	
buildrepo:
	@$(call make-repo)

# Create obj directory structure
define make-repo
	mkdir -p $(OBJDIR)
	for dir in $(SRCDIRS); \
	do \
		mkdir -p $(OBJDIR)/$$dir; \
	done
endef


-include $(DEPFILES)
