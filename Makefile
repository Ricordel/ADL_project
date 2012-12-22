# Compiler options
#CC = gcc
#CXX = g++
CFLAGS = -c -O2 -Wall -Wextra -Isrc -DNDEBUG -std=c99 -pedantic $(OPTFLAGS)
CXXFLAGS = -c -O2 -Wall -Wextra -Isrc -DNDEBUG  -std=c++11 -pedantic $(OPTFLAGS)

# Superset
LD = g++
LDFLAGS = $(OPTLIBS)

# Project name
PROJECT = main

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

# Targets
find_functions: depends buildrepo $(COMMON_OBJS) $(OBJDIR)/FuncGenerator.o $(OBJDIR)/find_functions.o
	$(LD) $(LDFLAGS) $(COMMON_OBJS) $(OBJDIR)/FuncGenerator.o $(OBJDIR)/find_functions.o -o $@

print_function: depends buildrepo $(COMMON_OBJS) $(OBJDIR)/expander.o
	$(LD) $(LDFLAGS) $(COMMON_OBJS) $(OBJDIR)/expander.o -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.c
	$(CC) $(CFLAGS) $(OPTS) -c $< -o $@

$(OBJDIR)/%.o: $(SRCDIR)/%.cpp
	$(CXX) $(CXXFLAGS) $(OPTS) -c $< -o $@
	

# dev is all + debug options: -g, -O0, -DDEBUG
dev: CFLAGS=-O0 -g -Wall -Wextra -Isrc -std=c99 -pedantic $(OPTFLAGS)
dev: CXXFLAGS=-O0 -g -Wall -Wextra -Isrc  -std=c++11 -pedantic $(OPTFLAGS)
dev: CPPFLAGS+=-DDEBUG
dev: $(PROJECT)



test: depends $(OBJS)
	cd tests && $(MAKE) run


depends: buildrepo
	@./scripts/make_depends.sh $(SRCDIR) $(OBJDIR)


clean:
	rm $(PROJECT) $(OBJDIR) -Rf
	cd tests && $(MAKE) clean

cleanall: clean
	cd tests && $(MAKE) cleanall

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
