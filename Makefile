CXXFLAGS=-std=c++14 -fsycl -pthread -g -O2
CL_OPTS=-lOpenCL


WG=$(shell which clang++)
CXX=$(WG)

########################################################################################

default: fp

info:
	@echo "clang     $(WG)"
	@echo "CXXFLAGS  $(OPTS) $(CXXFLAGS)"


fp : main.cpp FindPrimesSYCL.o Crunch.o MSG.h
	$(CXX) $(CXXFLAGS) Crunch.o FindPrimesSYCL.o main.cpp -o fp $(CL_OPTS) 

Crunch.o : Crunch.cpp Crunch.h
	$(CXX) $(CXXFLAGS) -o Crunch.o -c Crunch.cpp

FindPrimesSYCL.o : FindPrimesSYCL.h FindPrimesSYCL.cpp work.h
	$(CXX) $(CXXFLAGS) -o FindPrimesSYCL.o -c FindPrimesSYCL.cpp


clean :
	rm -f fp FindPrimesSYCL.o Crunch.o
