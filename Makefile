# mac
# change this to match your system
#SFTPATH=/Users/wyan/local

# linux
include $(PVFMM_DIR)/MakeVariables

CXX= mpicxx 
LINK= mpicxx 

CXXFLAGS= $(CXXFLAGS_PVFMM) 
LINKFLAGS= $(CXXFLAGS) $(LDLIBS_PVFMM)

# System-specific settings
SHELL = /bin/bash
SYSLIB =	
SIZE =	size

# Source Files. Header files will be automatically processed
SRC = main.cpp SimpleKernel.cpp STKFMM.cpp Util/ChebNodal.cpp Util/PointDistribution.cpp 

# Definitions
EXE := TestSTKFMM.X
OBJ := $(SRC:.cpp=.o)

all: $(EXE) 

# pull in dependency info for *existing* .o files
-include $(OBJ:.o=.d)

# Link rule
$(EXE): $(OBJ)
	$(LINK) $(OBJ)  -o $(EXE) $(LINKFLAGS)
	$(SIZE) $(EXE)

# use the trick from
# http://scottmcpeak.com/autodepend/autodepend.html
# handle header dependency automatically
# compile and generate dependency info;
# more complicated dependency computation, so all prereqs listed
# will also become command-less, prereq-less targets
#   sed:    strip the target (everything before colon)
#   sed:    remove any continuation backslashes
#   fmt -1: list words one per line
#   sed:    strip leading spaces
#   sed:    add trailing colons
%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o
	$(CXX) -MM $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp > $*.d
	@cp -f $*.d $*.d.tmp
	@sed -e 's/.*://' -e 's/\\$$//' < $*.d.tmp | fmt -1 | \
	  sed -e 's/^ *//' -e 's/$$/:/' >> $*.d
	@rm -f $*.d.tmp


# remove compilation products
clean: 
	rm -f ./$(OBJ)
	rm -f ./$(EXE)
	rm -f ./*.d

doc: 
	cd ./Doc && pdflatex ./*.tex