#ifndef ALIGNMENTSCORE_CUH
#define ALIGNMENTSCORE_CUH

#include "definitions.cuh"

/*******************************************************************************
 *
 *******************************************************************************/

class AlignmentScore
{
public:
  AlignmentScore(int gapOpen, int gapExt, unsigned int wSize, int maxSeqLen);
  ~AlignmentScore();

  void loadDB(char *lbuf, int lSz);
  TexVariablesAddresses copySeqsToTex(int startSeqs1No, int startSeqs2No);
  void parse(std::string seqs, std::vector<int>& starts, std::vector<int>& lenghts, char *input, int size);
  void computeDistanceMtx(char *rbuf, int rSz, std::vector< std::vector<float> >& distances, int r0);

private:
  int parts;
  int blockShape;
  int maxSeqLength;
  unsigned int windowSize;
  int maxMultiprocessorCount;
  static const char conversion[256];

  int xSize;
  std::string xSeqs;
  std::vector<int> xStarts;
  std::vector<int> xLengths;

  int ySize;
  std::string ySeqs;
  std::vector<int> yStarts;
  std::vector<int> yLengths;
};

void ScoreKernel(dim3 grid, dim3 block, short2 *AF, int *scoresDevPtr, bool border);

#endif
