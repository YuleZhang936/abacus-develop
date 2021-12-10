
#==========================
# Compiler information 
#==========================
CPLUSPLUS = mpicxx
CUDA_COMPILE = nvcc
OBJ_DIR = pw_obj
BIN_DIR = testbin
NP      = 12
#==========================
# Objects
#==========================
VPATH=../../src_parallel\
:../../module_base\
:../

PW_OBJS_0=intarray.o\
matrix.o\
matrix3.o\
tool_quit.o\
mymath3.o\
timer.o\
global_variable.o\
parallel_global.o\
pw_basis.o\
pw_distributer.o\
pw_init.o\
pw_transform.o\
pw_distributeg.o\
pw_distributeg_method1.o\
pw_distributeg_method2.o\
fft.o

DOUBLEFILE=test1.exe\
test2.exe\
test3.exe\
test2-1.exe\
test2-2.exe\
test2-3.exe\
test_t1.exe\
test_t2.exe

FLOATFILE=testf2.exe\
testf3.exe
TESTFILE0 = ${DOUBLEFILE} 

#==========================
# Options
#==========================
#No MPI
# HONG = -D__NORMAL
# CPLUSPLUS = g++

#Mix Precision
# HONG = -D__MIX_PRECISION -D__NORMAL
# TESTFILE0 = ${DOUBLEFILE} ${FLOATFILE}
# CPLUSPULS = g++

#Only MPI
HONG = -D__MPI -D__NORMAL

#MPI + Mix Precision
# HONG = -D__MPI -D__MIX_PRECISION -D__NORMAL
# TESTFILE0 = ${DOUBLEFILE} ${FLOATFILE}

#Cuda
#HONG = -D__MPI -D__CUDA -D__NORMAL

#Cuda & Mix Precision
#HONG = -D__MPI -D__CUDA -D__MIX_PRECISION -D__NORMAL
# TESTFILE0 = ${DOUBLEFILE} ${FLOATFILE}


PW_OBJS=$(patsubst %.o, ${OBJ_DIR}/%.o, ${PW_OBJS_0})
TESTFILE=$(patsubst %.exe, ${BIN_DIR}/%.exe, ${TESTFILE0})

##==========================
## FFTW package needed 
##==========================
#Use fftw package
FFTW_DIR = /home/qianrui/gnucompile/g_fftw-3.3.8-mpi
FFTW_LIB_DIR     = ${FFTW_DIR}/lib
FFTW_INCLUDE_DIR = ${FFTW_DIR}/include
FFTW_LIB         = -L${FFTW_LIB_DIR} -lfftw3 -lfftw3f -Wl,-rpath=${FFTW_LIB_DIR}
# FFTW_LIB         = -L${FFTW_LIB_DIR} -lfftw3 -Wl,-rpath=${FFTW_LIB_DIR}



##==========================
## CUDA needed 
##==========================
# CUDA_DIR = /usr/local/cuda-11.0
# CUDA_INCLUDE_DIR	= ${CUDA_DIR}/include 
# CUDA_LIB_DIR		= ${CUDA_DIR}/lib64
# CUDA_LIB			= -L${CUDA_LIB_DIR} -lcufft -lcublas -lcudart

LIBS = ${FFTW_LIB} ${CUDA_LIB}
#LIBS = ${FFTW_LIB} ${CUDA_LIB} -ltcmalloc -lprofiler
OPTS = -I${FFTW_INCLUDE_DIR} ${HONG} -Ofast -std=c++11 -Wall -g 
#==========================
# MAKING OPTIONS
#==========================
pw : 
	@ make init
	@ make -j $(NP) ${PW_OBJS}
	@ make -j $(NP) ${TESTFILE}

init :
	@ if [ ! -d $(OBJ_DIR) ]; then mkdir $(OBJ_DIR); fi
	@ if [ ! -d $(BIN_DIR) ]; then mkdir $(BIN_DIR); fi
	@ if [ ! -d $(OBJ_DIR)/README ]; then echo "This directory contains all of the .o files" > $(OBJ_DIR)/README; fi

${BIN_DIR}/%.exe: %.cpp
	${CPLUSPLUS} ${OPTS} $< test_tool.cpp ${PW_OBJS}  ${LIBS} -o $@
${OBJ_DIR}/%.o:%.cpp
	${CPLUSPLUS} ${OPTS} -c ${HONG} $< -o $@

.PHONY:clean
clean:
	@ if [ -d $(OBJ_DIR) ]; then rm -rf $(OBJ_DIR); fi
	@ if [ -d $(BIN_DIR) ]; then rm -rf $(BIN_DIR); fi
