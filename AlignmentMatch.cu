#include "AlignmentMatch.cuh"

#define gpuErrchk(code) { \
if(code != cudaSuccess) { \
    printf("GPUassert: %s at function %s on file %s line %d\n", cudaGetErrorString(code), __FUNCTION__, __FILE__, __LINE__); \
    exit(code); \
  } \
}

const char
AlignmentMatch::conversion[256] = {
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  23,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,   0,  20,   4,
  3,   6,  13,   7,   8,   9,  -1,  11,  10,  12,   2,  -1,  14,   5,   1,  15,  16,
 -1,  19,  17,  22,  18,  21,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,
 -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1,  -1
};

const unsigned char
AlignmentMatch::revConversion[128] = {
  65,  82,  78,  68,  67,  81,  69,  71,  72,  73,  76,  75,  77,  70,  80,  83,  84, 
  87,  89,  86,  66,  90,  88,  42,  0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
  0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0,   0, 
  0,   0,   0,   0,   0,   0,   0,   0,   0
};

inline bool
compare(seqEntry* first, seqEntry* second)
{
  return (first->getEstimatedComplexity() < second->getEstimatedComplexity());
}

TexVariablesAddresses
AlignmentMatch::copySeqsToTex(int startSeqs1No, int startSeqs2No)
{
  TexVariablesAddresses result;
  //the size of allocated memory for texSeqs1 and the size of allocated memory for texSeqs2
  int size1, size2;
  char *firstSeqDev, *secondSeqDev;
  std::string firstSeqHost, secondSeqHost;

  //COPYING ARRAYS OF SEQUENCES STARTS INTO CONST MEMORY
  gpuErrchk(cudaMemcpyToSymbol(tex1Starts, &starts[startSeqs1No], sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(tex2Starts, &starts[startSeqs2No], sizeof(int)));

  //COPYING X SEQUENCES TO TEXTURE
  firstSeqHost = seq[starts[startSeqs1No]];
  size1 = lengths[startSeqs1No];//length of all sequences within the window

  gpuErrchk(cudaMalloc((void**) &firstSeqDev, sizeof(char) * size1));
  gpuErrchk(cudaMemcpy(firstSeqDev, firstSeqHost.c_str(), sizeof(char) * size1, cudaMemcpyHostToDevice));
  gpuErrchk(cudaBindTexture(0, texSeqsX, firstSeqDev, size1));

  //COPYING Y SEQUENCES TO TEXTURE
  secondSeqHost = seq[starts[startSeqs2No]];
  size2 = lengths[startSeqs2No];

  gpuErrchk(cudaMalloc((void**) &secondSeqDev, sizeof(char) * size2));
  gpuErrchk(cudaMemcpy(secondSeqDev, secondSeqHost.c_str(), sizeof(char) * size2, cudaMemcpyHostToDevice));
  gpuErrchk(cudaBindTexture(0, texSeqsY, secondSeqDev, size2));

  result.texSeqs1DevPtr = firstSeqDev;
  result.texSeqs2DevPtr = secondSeqDev;

  return result;
}

AlignmentMatch::AlignmentMatch(int gapOpen, int gapExt, unsigned int wSize, int maxSeqLen)
{
  int deviceCount;
  unsigned int memoryOffset;
  cudaDeviceProp deviceProp;

  // This function call returns 0 if there are no CUDA capable devices.
  cudaGetDeviceCount(&deviceCount);
  
  if(deviceCount == 0) {
    fprintf(stderr, "There are no available device(s) that support CUDA\n");
    exit(0);
  }

  cudaGetDeviceProperties(&deviceProp, 0);
  maxMultiprocessorCount = deviceProp.multiProcessorCount;

  pairWise = true;
  windowSize = wSize;
  maxSeqLength = maxSeqLen;
  memoryOffset = windowSize * windowSize;
  blockShape = ALIGNMENT_BLOCK_SHAPE;

  gpuErrchk(cudaMemcpyToSymbol(gapOp, &gapOpen, sizeof(char)));
  gpuErrchk(cudaMemcpyToSymbol(gapEx, &gapExt, sizeof(char)));
  gpuErrchk(cudaMemcpyToSymbol(window, &windowSize, sizeof(unsigned int)));
  gpuErrchk(cudaMemcpyToSymbol(offset, &memoryOffset, sizeof(unsigned int)));
}

AlignmentMatch::~AlignmentMatch()
{
}

void
AlignmentMatch::prepare(std::vector< std::string >& seqs, int myProc)
{
  char tmp;
  std::string actualSeq;
  int i, seqNo, seqLength;

  for (seqNo = 0; seqNo < seqs.size(); seqNo++) {
    actualSeq = seqs[seqNo];
    seqLength = actualSeq.size();
    for (i = 0; i < seqLength / 2; i++) {
      tmp = conversion[(unsigned char) actualSeq[i]];
      actualSeq[i] = conversion[(unsigned char) actualSeq[seqLength - 1 - i]];
      actualSeq[seqLength - 1 - i] = tmp;
    }
  }
}

void
AlignmentMatch::computeAlignment(std::vector< iNode >& aSeqs, std::vector< std::string >& seqs, int myProc)
{
  computePairwise(aSeqs, seqs, myProc);
  computeMSA(aSeqs, seqs, myProc);
}

void
AlignmentMatch::computePairwise(std::vector< iNode >& aSeqs, std::vector< std::string >& seqs, int myProc)
{
  short2 *AF;
  unsigned int *back;
  unsigned int backSize;
  unsigned int *matchesSeqXDevPtr, *matchesSeqYDevPtr; //this array will have 4 characters packed in one int
  unsigned int *outMatchesSeqXDevPtr, *outMatchesSeqYDevPtr; //this array will have 4 characters packed in one int
 
  int *scoresDevPtr;
  TexVariablesAddresses addr;
  int startSeqs1No, startSeqs2No, maxSeqLengthAlignedTo4;
  int i, j, height, offset, numberOfAlignments, windowsNumber;

  dim3 block(blockShape, blockShape);
  dim3 reorderGrid((windowSize * windowSize) / blockShape);
  dim3 grid(((windowSize - 1) / blockShape + 1), ((windowSize - 1) / blockShape + 1));
  
  int partId = 1;
  std::string seq;
  seqEntry *params;
  std::vector< seqEntry* > jobs;

  seq.clear();
  starts.clear();
  starts.push_back(0);
  numberOfAlignments = j = 0;
  for(i = 0; i < aSeqs.size(); i++) {

    // Search for pairwise alignment that can be done now.
    if(aSeqs[i].vecSize == 2) {
      //if (myProc == 0)
        //printf("aSeqs[%d].vecSize: %d iSpecies[0]: %d iSpecies[1]: %d\n", i, aSeqs[i].vecSize, aSeqs[i].iSpecies[0], aSeqs[i].iSpecies[1]);

      aSeqs[i].aligned = true;

      seq.append(seqs[aSeqs[i].iSpecies[0]]);
      lengths.push_back(seqs[aSeqs[i].iSpecies[0]].size());
      starts.push_back(seqs[aSeqs[i].iSpecies[0]].size() + starts[j++]);

      seq.append(seqs[aSeqs[i].iSpecies[1]]);
      lengths.push_back(seqs[aSeqs[i].iSpecies[1]].size());
      starts.push_back(seqs[aSeqs[i].iSpecies[1]].size() + starts[j++]);

      numberOfAlignments++;
    }
  }

  //printf("myProc: %d aSeqs.size: %d pairwise alignments: %d\n", myProc, aSeqs.size() - 3, numberOfAlignments);

  if(numberOfAlignments) {

    windowsNumber = (numberOfAlignments - 1) / windowSize + 1;
    //printf("windowsNumber: %d numberOfAlignments: %d\n", windowsNumber, numberOfAlignments);
    for(i = windowsNumber - 1; i > 0; i -= 2) {
      params = new seqEntry;

      params->windowX = i;
      params->windowSumX = 0;
      offset = windowSize * i;
      params->windowMaxX = lengths[offset];
      for(j = 0; (j < windowSize) && (offset + j < numberOfAlignments); j += blockShape)
        params->windowSumX += lengths[(offset + j)];

      params->windowY = (i - 1);
      params->windowSumY = 0;
      offset = windowSize * (i - 1);
      params->windowMaxY = lengths[offset];
      for(j = 0; (j < windowSize) && (offset + j < numberOfAlignments); j += blockShape)
        params->windowSumY += lengths[(offset + j)];

      params->size = numberOfAlignments;
      params->partId = partId++;
      params->blockShape = blockShape;
      params->windowSize = windowSize;
      params->maxMultiprocessorCount = maxMultiprocessorCount;

      jobs.push_back(params);
    }

    std::sort(jobs.begin(), jobs.end(), compare);

    try {
      //one element in AF matrix:
      // - 2 bytes for element of A matrix
      // - 2 bytes for element of F matrix
      gpuErrchk(cudaMalloc(&AF, sizeof(int) * maxSeqLength * windowSize * windowSize));

      //sizeof(int) - one element in A matrix:
      // - 2 bytes for element of A matrix
      // - 2 bytes for element of F matrix
      gpuErrchk(cudaMalloc(&scoresDevPtr, sizeof(int) * windowSize * windowSize));

      for (i = 0; i < jobs.size(); i++) {
        startSeqs1No = jobs[i]->windowX;
        startSeqs2No = jobs[i]->windowY;

        //height of this array must be dividable by 8 (ALIGNMENT_MATCH_Y_STEPS)
        height = ((maxSeqLength - 1) / 8 + 1) * 8; //8->8, 9->16, 10->16 ...
        
        backSize = sizeof(unsigned int) * (height + 8);
        backSize *= (maxSeqLength + 1) * windowSize * (windowSize / ALIGNMENT_MATCH_Y_STEPS);
        gpuErrchk(cudaMalloc(&back, backSize));
        
        //memory for temporary (intermediate) results (alignments/matches)
        //we need: 2x maxSeqLength * 2 * windowSize * windowSize
        maxSeqLengthAlignedTo4 = ((maxSeqLength - 1) / 4 + 1) * 4;
        gpuErrchk(cudaMalloc(&matchesSeqXDevPtr, sizeof(char) * maxSeqLengthAlignedTo4 * 2 * windowSize * windowSize));
        gpuErrchk(cudaMalloc(&matchesSeqYDevPtr, sizeof(char) * maxSeqLengthAlignedTo4 * 2 * windowSize * windowSize));

        //memory for final results (alignments/matches)
        gpuErrchk(cudaMalloc(&outMatchesSeqXDevPtr, sizeof(char) * maxSeqLengthAlignedTo4 * 2 * windowSize * windowSize));
        gpuErrchk(cudaMalloc(&outMatchesSeqYDevPtr, sizeof(char) * maxSeqLengthAlignedTo4 * 2 * windowSize * windowSize));

        //printf("copySeqsToTex startSeqs1No: %d windowX: %d startSeqs2No: %d windowY: %d\n", startSeqs1No, jobs[i]->windowX, startSeqs2No, jobs[i]->windowY);
        //copying sequences to texture memory
        addr = copySeqsToTex(startSeqs1No, startSeqs2No);

        /***********************************************************************
         * KERNEL 1                                                            *
         * score calculation and "back" matrix fill                            *
         ***********************************************************************/
         //printf("MatchKernel\n");
        //maxSeqLength+1 => +1 because we have to take into account the -1 column
        MatchKernel(grid, block, AF, back, scoresDevPtr, maxSeqLength + 1,
                    (jobs[i]->windowX == jobs[i]->windowY));

        /***********************************************************************
         * KERNER 2                                                            *
         * backtracing - alignment matches generation                          *
         ***********************************************************************/
         //printf("BacktraceKernel\n");
        BacktraceKernel(grid, block, back, maxSeqLength + 1, matchesSeqXDevPtr, matchesSeqYDevPtr,
                        (jobs[i]->windowX == jobs[i]->windowY));
        
        /***********************************************************************
         * KERNER 3                                                            *
         * changing order of the results in GPU memory                         *
         ***********************************************************************/
        
        /***********************************************************************
         * maxSeqLengthAlignedTo4 * 2 / 4                                      *
         *    -> * 2 because alignment can be 2x as long as the longest        *
         *      sequence                                                       *
         *    -> / 4 because we packed chars to int                            *
         ***********************************************************************/
         //printf("ReorderMatches\n");
        ReorderMatches(reorderGrid, block, matchesSeqXDevPtr, outMatchesSeqXDevPtr,
                       matchesSeqYDevPtr, outMatchesSeqYDevPtr, maxSeqLengthAlignedTo4 * 2 / 4);
/*
        gpuErrchk(cudaMemcpy(&seqs[jobs[i]->windowX],
                  outMatchesSeqXDevPtr, sizeof(char) * maxSeqLengthAlignedTo4 * 2 * windowSize * windowSize,
                  cudaMemcpyDeviceToHost));

        gpuErrchk(cudaMemcpy(&seqs[jobs[i]->windowY],
                  outMatchesSeqYDevPtr, sizeof(char) * maxSeqLengthAlignedTo4 * 2 * windowSize * windowSize,
                  cudaMemcpyDeviceToHost));
*/
        //dealocating memory on GPU
        gpuErrchk(cudaFree(back));
        gpuErrchk(cudaFree(matchesSeqXDevPtr));
        gpuErrchk(cudaFree(matchesSeqYDevPtr));
        gpuErrchk(cudaFree(outMatchesSeqXDevPtr));
        gpuErrchk(cudaFree(outMatchesSeqYDevPtr));

        gpuErrchk(cudaFree(addr.texSeqs1DevPtr));
        gpuErrchk(cudaFree(addr.texSeqs2DevPtr));
      }

      gpuErrchk(cudaFree(AF));
      gpuErrchk(cudaFree(scoresDevPtr));

    }
    catch(std::exception &ex) {
      printf("Error: %s\n", ex.what());
    }
  }
}


void
AlignmentMatch::computeMSA(std::vector< iNode >& aSeqs, std::vector< std::string >& seqs, int myProc)
{
  short2 *AF;
  unsigned int *back;
  unsigned int backSize;
  unsigned int *matchesSeqXDevPtr, *matchesSeqYDevPtr; //this array will have 4 characters packed in one int
  unsigned int *outMatchesSeqXDevPtr, *outMatchesSeqYDevPtr; //this array will have 4 characters packed in one int
 
  bool canDoIt;
  int *scoresDevPtr;
  TexVariablesAddresses addr;
  int startSeqs1No, startSeqs2No, maxSeqLengthAlignedTo4;
  int i, j, k, height, offset, numberOfAlignments, windowsNumber;

  dim3 block(blockShape, blockShape);
  dim3 reorderGrid((windowSize * windowSize) / blockShape);
  dim3 grid(((windowSize - 1) / blockShape + 1), ((windowSize - 1) / blockShape + 1));
  
  int partId = 2;
  std::string seq;
  seqEntry *params;
  std::vector< seqEntry* > jobs;


  seq.clear();
  starts.clear();
  starts.push_back(0);
  numberOfAlignments = 0;
  //printf("computeMSA\n");

  for(i = 0; i < aSeqs.size(); i++) {
    if(aSeqs[i].vecSize == 0) {

      canDoIt = true;
      for(j = 0; j < aSeqs[i].iSpecies.size(); j++) {
        if((aSeqs[i].iSpecies[j] > aSeqs.size()) ||
           !aSeqs[aSeqs[i].iSpecies[j]].aligned) {
          canDoIt = false;
          break;
        }
      }

      if(canDoIt) {
        aSeqs[i].aligned = true;
        for(j = 0; j < aSeqs[i].iSpecies.size(); j++) {

          seq.append(seqs[aSeqs[i].iSpecies[j]]);
          lengths.push_back(seqs[aSeqs[i].iSpecies[j]].size());
          starts.push_back(seqs[aSeqs[i].iSpecies[j]].size() + starts[k++]);

          seq.append(seqs[aSeqs[i].iSpecies[j]]);
          lengths.push_back(seqs[aSeqs[i].iSpecies[j]].size());
          starts.push_back(seqs[aSeqs[i].iSpecies[j]].size() + starts[k++]);

          aSeqs[i].vecSize += aSeqs[aSeqs[i].iSpecies[j]].vecSize;
        }

        numberOfAlignments++;
      }
    }
  }
  //printf("myProc: %d MSA alignments: %d\n", myProc, numberOfAlignments);

  if(numberOfAlignments) {

    windowsNumber = (numberOfAlignments - 1) / windowSize + 1;
    //printf("windowsNumber: %d numberOfAlignments: %d\n", windowsNumber, numberOfAlignments);
    for(i = windowsNumber - 1; i > 0; i -= 2) {
      params = new seqEntry;

      params->windowX = i;
      params->windowSumX = 0;
      offset = windowSize * i;
      //params->windowMaxX = lengths[offset];
      //for(j = 0; (j < windowSize) && (offset + j < numberOfAlignments); j += blockShape)
        //params->windowSumX += lengths[(offset + j)];

      params->windowY = (i - 1);
      params->windowSumY = 0;
      offset = windowSize * (i - 1);
      //params->windowMaxY = lengths[offset];
      //for(j = 0; (j < windowSize) && (offset + j < numberOfAlignments); j += blockShape)
        //params->windowSumY += lengths[(offset + j)];

      params->size = numberOfAlignments;
      params->partId = partId++;
      params->blockShape = blockShape;
      params->windowSize = windowSize;
      params->maxMultiprocessorCount = maxMultiprocessorCount;

      jobs.push_back(params);
    }

    std::sort(jobs.begin(), jobs.end(), compare);
    //printf("try catch\n");


    //one element in AF matrix:
    // - 2 bytes for element of A matrix
    // - 2 bytes for element of F matrix
    gpuErrchk(cudaMalloc(&AF, sizeof(int) * maxSeqLength * windowSize * windowSize));

    //sizeof(int) - one element in A matrix:
    // - 2 bytes for element of A matrix
    // - 2 bytes for element of F matrix
    gpuErrchk(cudaMalloc(&scoresDevPtr, sizeof(int) * windowSize * windowSize));

    for(i = 0; i < jobs.size(); i++) {
      ;
    }
    gpuErrchk(cudaFree(AF));
    gpuErrchk(cudaFree(scoresDevPtr));
  }
}