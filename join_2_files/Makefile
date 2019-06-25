# Just to activate/deactivate things...
ifeq ("x","x")

CUDA_INSTALL_PATH ?= /usr/local/cuda

CC := gcc
CXX := g++
LINK := g++ -fPIC

#NVCC := nvcc -ccbin /usr/bin
NVCC := /usr/local/cuda/bin/nvcc

# Includes
INCLUDES = -I.
INCLUDES_CUDA = -I$(CUDA_INSTALL_PATH)/include

# Common flags
COMMONFLAGS += -O3 -m64 $(INCLUDES)
#COMMONFLAGS += -g -lineinfo $(INCLUDES)
CFLAGS += $(COMMONFLAGS) #$(INCLUDES_CUDA)
CXXFLAGS += -std=c++0x $(COMMONFLAGS) #$(INCLUDES_CUDA)
NVCCFLAGS += -std=c++11 $(COMMONFLAGS) $(INCLUDES_CUDA)


#Then you specify where to locate the CUDA binaries for linking :
#LIB_CUDA := -L$(CUDA_INSTALL_PATH)/lib -lcudart
LIB_CUDA := -L $(CUDA_INSTALL_PATH)/lib64 -lcudart -lOpenCL
LIB_COMMON := -lpthread -lm


#Then you specify the object files that the executable depends on and the name of the executable flle:
# $(SRC) and $(EXEC) comes from the comand line

# string with all .o files separated by one space
OBJS = $(SRC)
# string with the name of the executable
TARGET = $(EXEC)
# linker command
LINKLINE = $(LINK) -o $(TARGET) $(OBJS)


#Finally you specify the compilation rules.
#The following are generic rules so all you need to do is change the
#OBJS and TARGET variables for compilation:

.SUFFIXES: .c .cpp .cu .o

%.c.o: %.c
	$(CC) -c $< -o $@ $(CFLAGS) $(LIB_COMMON)

%.cpp.o: %.cpp
	$(CXX) -c $< -o $@ $(CXXFLAGS) $(LIB_COMMON)

%.cu.o: %.cu
	$(NVCC) -c $< -o $@ --compiler-options "-O3" $(NVCCFLAGS) $(LIB_COMMON) $(LIB_CUDA)

$(TARGET): $(OBJS) Makefile
	$(LINKLINE) $(LIB_COMMON)


clean:
	rm *.o $(TARGET)

endif
