# build file

#CXX = mpicxx
#CXX = /opt/mpich2/mx/bin/mpicxx
CXX = /opt/mpich2/gnu/bin/mpicxx

NVCC = nvcc

CXXFLAGS := -O2 -g -DNDEBUG -I. -I/usr/local/cuda/include/ #-pedantic -Wall
NVCCFLAGS	:= -O2 -g -arch sm_21 -I/usr/local/cuda/include/ -c

# Default flags and libraries

CXX_OBJS	=	CmdLineOptions.o SpeciesBlock.o TreeNode.o FastaFile.o MatchesManager.o wnj.o
NVCC_OBJS	=	NeedlemanWunsch.o AlignmentScore.o AlignmentMatch.o
EXEC := clustal


all: $(EXEC)

$(EXEC): $(NVCC_OBJS) $(CXX_OBJS)
	$(CXX) -o $@ $^ -lcudart -L/usr/local/cuda/lib64 -lrt

$(NVCC_OBJS): %.o : %.cu
	$(NVCC) $(NVCCFLAGS) -o $@ $<

clean:
	$(RM) .*~ *~
	$(RM) $(EXEC) $(CXX_OBJS) $(NVCC_OBJS)
