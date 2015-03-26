#ifndef ALIGNMENTMATCH_CUH
#define ALIGNMENTMATCH_CUH

#include <exception>

#include "definitions.cuh"
#include "MatchesManager.h"

struct iNode
{
  int index;
  int vecSize;
  bool aligned;
  std::vector< int > iSpecies;
};

/*******************************************************************************
 *
 *******************************************************************************/

class AlignmentMatch
{
public:
  AlignmentMatch(int gapOpen, int gapExt, unsigned int wSize, int maxSeqLen);
  ~AlignmentMatch();

  void prepare(std::vector< std::string >& seqs, int myProc);
  void computeAlignment(std::vector< iNode >& aSeqs, std::vector< std::string >& seqs, int myProc);

  TexVariablesAddresses copySeqsToTex(int startSeqs1No, int startSeqs2No);

  void computeMSA(std::vector< iNode >& aSeqs, std::vector< std::string >& seqs, int myProc);
  void computePairwise(std::vector< iNode >& aSeqs, std::vector< std::string >& seqs, int myProc);

private:
  int parts;
  int blockShape;
  int maxSeqLength;
  unsigned int windowSize;
  int maxMultiprocessorCount;
  static const char conversion[256];
  static const unsigned char revConversion[128];

  int nSize;
  bool pairWise;
  std::string seq;
  std::vector<int> starts;
  std::vector<int> lengths;  
};

void ScoreKernel(dim3 grid, dim3 block, short2 *AF, int *scoresDevPtr, bool border);


void MatchKernel(dim3 grid, dim3 block, short2 *AF, unsigned int *back, int *scoresDevPtr, short rowWidth, bool border);
void ReorderMatches(dim3 grid, dim3 block, unsigned int *inX, unsigned int *outX, unsigned int *inY, unsigned int *outY, int maxMatchLength);
void BacktraceKernel(dim3 grid, dim3 block, unsigned int *back, short rowWidth, unsigned int *matchesX, unsigned int *matchesY, bool border);

#endif
