# change this to match your system

# pvffm library
include $(PVFMM_DIR)/MakeVariables

# Compilers 
CXX=mpiicpc
LINK= mpiicpc 

# location of the Python header files
PYTHON_VERSION = 3.7m
PYTHON_DIR = /mnt/home/fbalboausabiaga/.miniconda3/envs/mpi4py
PYTHON_INCLUDE = $(PYTHON_DIR)/include/python$(PYTHON_VERSION)/
PYTHON_LIB = $(PYTHON_DIR)/lib/

# location of the Boost Python include files and library
# for boost installed through conda 
BOOST_INC = $(PYTHON_DIR)/include
BOOST_LIB = $(PYTHON_DIR)/lib

# Define includes
INCLUDE_DIRS = -I../ -I$(PYTHON_INCLUDE) -I$(BOOST_INC)

# Define flags. -Wl,-undefined,error -Wl,-rpath,$(BOOST_LIB) 
CXXFLAGS += $(CXXFLAGS_PVFMM)  -DFMMTIMING -fPIC
LINKFLAGS := $(CXXFLAGS) $(LDLIBS_PVFMM) $(LINKFLAGS)  -Wl,--no-undefined -Wl,-rpath=$(BOOST_LIB) -shared \
			-L$(BOOST_LIB)  -lboost_python -lboost_numpy -lboost_system \
			-L$(PYTHON_LIB) -lpython$(PYTHON_VERSION) 

# Target and sources
TARGET = stkfmm
SRC = ../Interface/STKFMM-py.cpp ../STKFMM.cpp 
INC = ../STKFMM.h 
OBJ = ../Interface/STKFMM-py.o ../STKFMM.o

# Link rule
$(TARGET).so: $(OBJ)
	$(LINK) $(OBJ) $(LINKFLAGS) -o $(TARGET).so

.cpp.o:
	$(CXX) $(CXXFLAGS) $(INCLUDE_DIRS) -c $*.cpp -o $*.o

# Individual dependencies
$(OBJ): $(INC)


clean: 
	rm -f $(OBJ)
	rm -f $(TARGET).so

