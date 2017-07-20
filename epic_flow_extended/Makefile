CCPP=g++ -std=c++11
CC=gcc

LAPACKCFLAGS=#-DTHREADEDLAPACK
LAPACKLDFLAGS=-L/usr/lib64/atlas/ -llapack

CFLAGS=-DFINTEGER=int -fPIC -Wall -g -O3 -msse4 $(LAPACKCFLAGS) -fopenmp -DUSE_OPENMP
LDFLAGS=-fPIC -Wall -g -O3 -msse4 -ljpeg -lpng $(LAPACKLDFLAGS) -fopenmp 

SOURCES_CPP := $(shell find . -name '*.cpp' ! -name "epicflow-variational-seq.cpp" ! -name "epicflow-variational.cpp")
SOURCES_C := $(shell find . -name '*.c')
OBJ := $(SOURCES_CPP:%.cpp=%.o) $(SOURCES_C:%.c=%.o) 
HEADERS := $(shell find . -name '*.h')

all: epicflow

.cpp.o:  %.cpp %.h
	$(CCPP) -o $@ $(CFLAGS) -c $+

.c.o:  %.c %.h
	$(CC) -o $@ $(CFLAGS) -c $+

epicflow: $(HEADERS) $(OBJ)
	$(CCPP) -o $@ $^ $(LDFLAGS)

SOURCES_CPP_VAR := $(shell find . -name '*.cpp' ! -name "epicflow.cpp" ! -name "epicflow-variational-seq.cpp")
OBJ_VAR := $(SOURCES_CPP_VAR:%.cpp=%.o) $(SOURCES_C:%.c=%.o) 
epicflow-variational: $(HEADERS) $(OBJ_VAR)
	$(CCPP) -o $@ $^ $(LDFLAGS)

SOURCES_CPP_VAR_SEQ := $(shell find . -name '*.cpp' ! -name "epicflow.cpp" ! -name "epicflow-variational.cpp")
OBJ_VAR_SEQ := $(SOURCES_CPP_VAR_SEQ:%.cpp=%.o) $(SOURCES_C:%.c=%.o) 
epicflow-variational-seq: $(HEADERS) $(OBJ_VAR_SEQ)
	$(CCPP) -o $@ $^ $(LDFLAGS) -lboost_system -lboost_filesystem `pkg-config --libs opencv`

epicflow-static: $(HEADERS) $(OBJ)
	$(CCPP) -o $@ $^ -static  /usr/lib64/libjpeg.a /usr/lib64/libpng.a /usr/lib64/libz.a /usr/lib64/libm.a /usr/lib64/liblapack.a /usr/lib/gcc/x86_64-redhat-linux/4.7.2/libgfortran.a /usr/lib64/libblas.a


clean:
	rm -f $(OBJ) epicflow
	
clean-variational:
	rm -f $(OBJ) epicflow-variational
	
clean-variational-seq:
	rm -f $(OBJ) epicflow-variational-seq

