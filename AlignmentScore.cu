#include "AlignmentScore.cuh"

#define gpuErrchk(code) { \
  if(code != cudaSuccess) { \
      printf("GPUassert: %s at function %s on file %s line %d\n", cudaGetErrorString(code), __FUNCTION__, __FILE__, __LINE__); \
      exit(code); \
    } \
}

const char
AlignmentScore::conversion[256] = {
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

inline bool
compare(seqEntry* first, seqEntry* second)
{
  return (first->getEstimatedComplexity() < second->getEstimatedComplexity());
}


TexVariablesAddresses
AlignmentScore::copySeqsToTex(int startSeqs1No, int startSeqs2No)
{
  TexVariablesAddresses result;
  //the size of allocated memory for texSeqs1 and the size of allocated memory for texSeqs2
  int i, size1, size2, maxCount;
  char *firstSeqDev, *secondSeqDev;
  std::string firstSeqHost, secondSeqHost;

  std::vector<int> starts1;
  std::vector<int> starts2;
  std::vector<int> lengths1;
  std::vector<int> lengths2;

  maxCount = startSeqs1No + windowSize;
  maxCount = MIN(maxCount, xSize);
  maxCount = MAX(maxCount - startSeqs1No, 0);

  for(i = 0; i < maxCount; i++) {
    lengths1.push_back(xLengths[i + startSeqs1No]);
    starts1.push_back(xStarts[i + startSeqs1No] - xStarts[startSeqs1No]);//we substract the offset to make starts1[0] == 0
  }
  for(i = maxCount; i <= windowSize; i++) {
    lengths1.push_back(0);
    starts1.push_back(((maxCount > 0) ? (starts1[maxCount - 1] + lengths1[maxCount - 1]) : 0));
  }

  maxCount = startSeqs2No + windowSize;
  maxCount = MIN(maxCount, ySize);
  maxCount = MAX(maxCount - startSeqs2No, 0);

  for(i = 0; i < maxCount; i++) {
    lengths2.push_back(yLengths[i + startSeqs2No]);
    starts2.push_back(yStarts[i + startSeqs2No] - yStarts[startSeqs2No]);//we substract the offset to make starts2[0] == 0
  }
  for(i = maxCount; i <= windowSize; i++) {
    lengths2.push_back(0);
    starts2.push_back(((maxCount > 0) ? (starts2[maxCount - 1] + lengths2[maxCount - 1]) : 0));
  }

  //COPYING ARRAYS OF SEQUENCES STARTS INTO CONST MEMORY
  gpuErrchk(cudaMemcpyToSymbol(tex1Starts, &starts1[0], (windowSize + 1) * sizeof(int)));
  gpuErrchk(cudaMemcpyToSymbol(tex2Starts, &starts2[0], (windowSize + 1) * sizeof(int)));


  //COPYING X SEQUENCES TO TEXTURE
  //printf("COPYING X SEQUENCES TO TEXTURE\n");
  firstSeqHost = xSeqs;
  firstSeqHost += xStarts[startSeqs1No];
  size1 = starts1[windowSize - 1] + lengths1[windowSize - 1];//length of all sequences within the window
  //printf("length of all sequences within the window %d\n", size1);

  gpuErrchk(cudaMalloc((void**) &firstSeqDev, sizeof(char) * size1));
  gpuErrchk(cudaMemcpy(firstSeqDev, firstSeqHost.c_str(), sizeof(char) * size1, cudaMemcpyHostToDevice));
  gpuErrchk(cudaBindTexture(0, texSeqsX, firstSeqDev, size1));


  //COPYING Y SEQUENCES TO TEXTURE
  //printf("COPYING Y SEQUENCES TO TEXTURE\n");
  secondSeqHost = ySeqs;
  secondSeqHost += yStarts[startSeqs2No];
  size2 = starts2[windowSize - 1] + lengths2[windowSize - 1];
  //printf("length of all sequences within the window\n");

  gpuErrchk(cudaMalloc((void**) &secondSeqDev, sizeof(char) * size2));
  gpuErrchk(cudaMemcpy(secondSeqDev, secondSeqHost.c_str(), sizeof(char) * size2, cudaMemcpyHostToDevice));
  gpuErrchk(cudaBindTexture(0, texSeqsY, secondSeqDev, size2));

  result.texSeqs1DevPtr = firstSeqDev;
  result.texSeqs2DevPtr = secondSeqDev;

  return result;
}

AlignmentScore::AlignmentScore(int gapOpen, int gapExt, unsigned int wSize, int maxSeqLen)
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

  windowSize = wSize;
  maxSeqLength = maxSeqLen;
  memoryOffset = windowSize * windowSize;
  blockShape = ALIGNMENT_BLOCK_SHAPE;

  gpuErrchk(cudaMemcpyToSymbol(gapOp, &gapOpen, sizeof(char)));
  gpuErrchk(cudaMemcpyToSymbol(gapEx, &gapExt, sizeof(char)));
  gpuErrchk(cudaMemcpyToSymbol(window, &windowSize, sizeof(unsigned int)));
  gpuErrchk(cudaMemcpyToSymbol(offset, &memoryOffset, sizeof(unsigned int)));
}

AlignmentScore::~AlignmentScore()
{
}

void
AlignmentScore::parse(std::string seqs, std::vector<int>& starts, std::vector<int>& lengths, char *input, int size)
{
  char *p, *actualSeq;
  int i, seqLen, seqNo;
  char tmp, buffer[maxSeqLength + 1];

  p = &input[sizeof(int)];

  starts.resize(size + 1);

  starts[0] = 0;
  for(seqNo = 1; seqNo <= size; seqNo++) {
    
    p += sizeof(int);
    memcpy((char *) &seqLen, p, sizeof(int));
    p += sizeof(int);

    memcpy(buffer, p, seqLen + 1);
    p += seqLen + 1;

    seqs.append(buffer);
    lengths.push_back(seqLen);
    starts[seqNo] = seqLen + starts[seqNo - 1];
  }

  for(seqNo = 0; seqNo < size; seqNo++) {
    actualSeq = &seqs[starts[seqNo]];
    seqLen = lengths[seqNo];
    
    for(i = 0; i < seqLen / 2; i++) {
      tmp = conversion[(unsigned char) actualSeq[i]];
      actualSeq[i] = conversion[(unsigned char) actualSeq[seqLen - 1 - i]];
      actualSeq[seqLen - 1 - i] = tmp;
    }
  }
}

void
AlignmentScore::loadDB(char *lbuf, int lSz)
{
  xSize = lSz;
  parse(xSeqs, xStarts, xLengths, lbuf, xSize);
}

void
AlignmentScore::computeDistanceMtx(char *rbuf, int rSz, std::vector< std::vector<float> >& distances, int r0)
{
  short2 *AF;
  int partId = 1;
  seqEntry *params;
  std::vector<seqEntry*> jobs;
  int *scoresDevPtr, *scoresHostPtr;
  int i, j, k, col, minX, minY, offset, size;
  int startSeqs1No, startSeqs2No, windowsNumber;

  dim3 block(blockShape, blockShape);
  dim3 grid(((windowSize - 1) / blockShape + 1), ((windowSize - 1) / blockShape + 1));

  ySize = rSz;
  size = MAX(xSize, ySize);
  windowsNumber = (size - 1) / windowSize + 1;

  parse(ySeqs, yStarts, yLengths, rbuf, ySize);

  //long jobs first and short jobs last is better for load balancing
  for(i = windowsNumber - 1; i >= 0; i--) { //we iterate through all the windows
    for(j = windowsNumber - 1; j >= 0; j--) {
      params = new seqEntry;

      params->windowX = i;
      params->windowSumX = 0;
      offset = windowSize * i;
      params->windowMaxX = xLengths[offset];
      for(k = 0; (k < windowSize) && (offset + k < ySize); k += blockShape)
        params->windowSumX += xLengths[(offset + k)];

      params->windowY = j;
      params->windowSumY = 0;
      offset = windowSize * j;
      params->windowMaxY = yLengths[offset];
      for(k = 0; (k < windowSize) && (offset + k < xSize); k += blockShape)
        params->windowSumY += yLengths[(offset + k)];

      params->size = size;
      params->partId = partId++;
      params->blockShape = blockShape;
      params->windowSize = windowSize;
      params->maxMultiprocessorCount = maxMultiprocessorCount;

      jobs.push_back(params);
    }
  }

  std::sort(jobs.begin(), jobs.end(), compare);
  scoresHostPtr = new int[windowSize * windowSize];

  for(i = 0; i < jobs.size(); i++) {
    startSeqs1No = jobs[i]->windowX * windowSize;
    startSeqs2No = jobs[i]->windowY * windowSize;

    //printf("cudaMalloc AF\n");
    gpuErrchk(cudaMalloc(&AF, sizeof(int) * maxSeqLength * windowSize * windowSize));
    //sizeof(int) - one element in A matrix:
    // - 2 bytes for element of A matrix
    // - 2 bytes for element of F matrix

    gpuErrchk(cudaMalloc(&scoresDevPtr, sizeof(int) * windowSize * windowSize));

    //copying sequences to texture memory
    TexVariablesAddresses addr = copySeqsToTex(startSeqs1No, startSeqs2No);

    //KERNEL INVOCATION
    ScoreKernel(grid, block, AF, scoresDevPtr, (jobs[i]->windowX == jobs[i]->windowY));

    //reading results from GPU
    gpuErrchk(cudaMemcpy(scoresHostPtr, scoresDevPtr, sizeof(int) * windowSize * windowSize, cudaMemcpyDeviceToHost));

    minX = MIN(windowSize, xSize - jobs[i]->windowX * windowSize);
    minY = MIN(windowSize, ySize - jobs[i]->windowY * windowSize);
    for(k = 0; k < minX; k++)
      for(j = 0; j < minY; j++)
        distances[jobs[i]->windowX * windowSize + k][jobs[i]->windowY * windowSize + j + col] = scoresHostPtr[j * windowSize + k];

    //dealocating memory on GPU
    gpuErrchk(cudaFree(AF));
    gpuErrchk(cudaFree(scoresDevPtr));
    gpuErrchk(cudaFree(addr.texSeqs1DevPtr));
    gpuErrchk(cudaFree(addr.texSeqs2DevPtr));
  }

  gpuErrchk(cudaUnbindTexture(texSeqsX));
  gpuErrchk(cudaUnbindTexture(texSeqsY));

  delete[] scoresHostPtr;
}
