CPP=g++-4.9
CC=gcc-4.9
CFLAGS=-O3 -std=c++11
LFLAGS=-O3
LD=g++-4.9

#!!!specify the blas lib location and the blas lib, these may not be the same in different machines
BLAS_LIBS_LOCATION=-L/usr/lib/atlas-blas
BLAS_LIBS=-lf77blas  

###choose the blas implementation
BLAS_DEFINE=-DBLAS_ATLAS
#BLAS_DEFINE=-DBLAS_INTEL_MKL

SRCS=$(wildcard src/*/*.cpp) $(wildcard src/*.cpp)
#no source files with the same name
OBJS=$(patsubst %.cpp,obj/%.o,$(notdir $(SRCS)))

t: obj/depends $(OBJS)
#	$(LD) $(LFLAGS) $(OBJS) -o nngdparser $(BLAS_LIBS_LOCATION) -lboost_regex -lboost_program_options $(BLAS_LIBS)
	$(LD) $(LFLAGS) $(OBJS) -o t

obj/depends:
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -MM $(SRCS) > $@
    
obj/%.o: src/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/tools/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@
obj/%.o: src/ef/%.cpp
	$(CPP) $(CFLAGS) $(BLAS_DEFINE) -c $< -o $@

include obj/depends

.PHONY: clean
clean:
	rm -f obj/*.o nngdparser
	