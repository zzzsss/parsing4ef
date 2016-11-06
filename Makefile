CPP=g++-4.9
CC=gcc-4.9
CFLAGS=-O3 -std=c++11 -Wall -g
LFLAGS=-O3 -g
LD = g++-4.9

#!!!specify the blas lib location and the blas lib, these may not be the same in different machines
#BLAS_LIBS_LOCATION=-L/usr/lib/atlas-blas
#BLAS_LIBS=-lf77blas  
###choose the blas implementation
#BLAS_DEFINE=-DBLAS_ATLAS
#BLAS_DEFINE=-DBLAS_INTEL_MKL

DYNET_INC_FLAG=-I/home/zhangzs/bin/eigen/ -I/home/zhangzs/bin/dynet/
DYNET_LINK_FLAG=-L/home/zhangzs/bin/dynet/build-mkl/dynet/ -lboost_serialization -ldynet

SRCS=$(wildcard src/*/*.cpp) $(wildcard src/*.cpp)
OBJS=$(patsubst %.cpp,obj/%.o,$(notdir $(SRCS)))

t: obj/depends $(OBJS)
#	$(LD) $(LFLAGS) $(OBJS) -o nngdparser $(BLAS_LIBS_LOCATION) -lboost_regex -lboost_program_options $(BLAS_LIBS)
	$(LD) $(LFLAGS) $(OBJS) $(DYNET_LINK_FLAG) -o t

obj/depends:
	$(CPP) $(DYNET_INC_FLAG) $(CFLAGS) $(BLAS_DEFINE) -MM $(SRCS) > $@
    
obj/%.o: src/%.cpp
	$(CPP) $(CFLAGS) -c $< -o $@
obj/%.o: src/tools/%.cpp
	$(CPP) $(CFLAGS) -c $< -o $@
obj/%.o: src/ef/%.cpp
	$(CPP) $(DYNET_INC_FLAG) $(CFLAGS) -c $< -o $@
obj/%.o: src/components/%.cpp
	$(CPP) $(CFLAGS) -c $< -o $@
obj/%.o: src/model/%.cpp
	$(CPP) $(DYNET_INC_FLAG) $(CFLAGS) -c $< -o $@

include obj/depends

.PHONY: clean 
.PHONY: zt
.PHONY: gzt

clean:
	rm -f obj/*.o obj/depends t zt gzt
zt:
	$(CPP) -O3 -std=c++11 -Wall src/*.cpp src/*/*.cpp $(DYNET_INC_FLAG) $(DYNET_LINK_FLAG) -o zt 
gzt:
	$(CPP) -g -std=c++11 -Wall src/*.cpp src/*/*.cpp $(DYNET_INC_FLAG) $(DYNET_LINK_FLAG) -o gzt 
