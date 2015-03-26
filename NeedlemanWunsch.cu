#include "definitions.cuh"

//PREPROCESSOR MACROS

#define PACK_0_BYTE(x)          (((int)((unsigned char)(x))) << 24)
#define PACK_1_BYTE(x)          (((int)((unsigned char)(x))) << 16)
#define PACK_2_BYTE(x)          (((int)((unsigned char)(x))) << 8)
#define PACK_3_BYTE(x)          (((int)((unsigned char)(x))))
#define PACK_BYTES(a, b, c, d)     (PACK_0_BYTE(a) | PACK_1_BYTE(b) | PACK_2_BYTE(c) | PACK_3_BYTE(d))

//CONSTANTS
#define SHORT_MIN -32768
#define WIN_SIZE    window
#define MEM_OFFSET  offset
#define BLOCK_SIZE  ALIGNMENT_BLOCK_SHAPE

#define seqXNo      (blockIdx.x * blockDim.x + threadIdx.x)
#define seqYNo      (blockIdx.y * blockDim.y + threadIdx.y)
#define startX      (tex1Starts[seqXNo])
#define startY      (tex2Starts[seqYNo])

__global__ void
NeedlemanWunschGlobalScoreKernel(short2* AF, int *scoresDevPtr, bool border = false)
{
  // SUBSTITUTION MATRIX GOES TO SHARED MEMORY
  __shared__ char shmSM[SUBSTITUTION_MATRIX_SIZE];
  short idx = threadIdx.y * blockDim.x + threadIdx.x;
  shmSM[idx] = substitutionMatrix[idx];
  idx += BLOCK_SIZE;
  shmSM[idx] = substitutionMatrix[idx];
  idx += BLOCK_SIZE;
  if(idx < 576)
    shmSM[idx] = substitutionMatrix[idx];
    
  __syncthreads();
  
  // |\xxx    we do not compute x
  // | \xx
  // |  \x
  // |___\

  if(border && (seqXNo > seqYNo))
    return;

  //we determine our position in the window
  //int seqXNo = blockIdx.x * blockDim.x + threadIdx.x; //there is an idea to do not hold this but
  //int seqYNo = blockIdx.y * blockDim.y + threadIdx.y; //recalculate this every time to save registers
  int blockThread = threadIdx.x + threadIdx.y * blockDim.x; //0...(BLOCK_SIZE-1)

  //int startX = tex1Starts[seqXNo];
  //int startY = tex2Starts[seqYNo];
  short2 lengthXY;
  lengthXY.x = tex1Starts[seqXNo + 1] - startX;
  lengthXY.y = tex2Starts[seqYNo + 1] - startY;

  if((lengthXY.x == 0) || (lengthXY.y == 0))//if there is nothing to do -> quit
    return;

  //startPosA == thread number within whole grid
  int startPosA = seqYNo * WIN_SIZE + seqXNo;

  //initialization of the -1 row in A matrix
  // - 2 bytes for element of A matrix
  // - 2 bytes for element of F matrix
  for(short x = 0; x < lengthXY.x; x++) {
    short2 tmp;
    tmp.x = -gapEx * (x + 1);
    tmp.y = SHORT_MIN + gapEx;
    AF[startPosA + x * MEM_OFFSET] = tmp;
  }

  //one element of sharedA consist of:
  // - one A element
  // - one E element
  __shared__ short2 AE_shared[ALIGNMENT_SCORE_Y_STEPS][BLOCK_SIZE];
  //elements of Y sequence go to sharedYSeq
  __shared__ int sharedYSeq[ALIGNMENT_SCORE_Y_STEPS/4][BLOCK_SIZE];


  short2 AE_current;
  AE_current.x = 0;

  // |
  // |
  // |
  // V
  for (short y = 0; y < lengthXY.y; y += ALIGNMENT_SCORE_Y_STEPS) {
    short2 A_init_upleft;
    A_init_upleft.x = -gapEx * y;

    //initialialization of the -1 column in A matrix
    // - one element of A matrix
    // - one element of E matrix
    for (short i = 0; i < ALIGNMENT_SCORE_Y_STEPS; i++) {
      short2 tmp;
      tmp.x = -gapEx * (y + i + 1);
      tmp.y = SHORT_MIN + gapEx;
      AE_shared[i][blockThread] = tmp;
    }


    //we read elements of the Y sequence
    for (short i = 0; i < ALIGNMENT_SCORE_Y_STEPS/4; i++) {
      sharedYSeq[i][blockThread] = PACK_BYTES(tex1Dfetch(texSeqsY, startY + y + i * 4 + 0),
                                              tex1Dfetch(texSeqsY, startY + y + i * 4 + 1),
                                              tex1Dfetch(texSeqsY, startY + y + i * 4 + 2),
                                              tex1Dfetch(texSeqsY, startY + y + i * 4 + 3));
    }


    //------>
    for (short x = 0; x < lengthXY.x; x++) {
      //actual up_left gets a value of recent read from the global memory
      //and actual read value is stored in first two bites of A_upleft
      A_init_upleft.y = A_init_upleft.x;

      char2 XYSeq;
      XYSeq.x = tex1Dfetch(texSeqsX, startX + x);
      
      //read from global memory
      short2 AF_up = AF[startPosA + x * MEM_OFFSET];
      AE_current.x = AF_up.x;
      

      //A_init -> up element read in previous iteration from global memory (up-left)
      A_init_upleft.x = AF_up.x;

      int F_current = AF_up.y;
      int similarity;
      short ymin = min(ALIGNMENT_SCORE_Y_STEPS, lengthXY.y - y); //(i < ALIGNMENT_SCORE_Y_STEPS) && (i + y < lengthY)
      //  |  /|  /|
      //  | / | / |
      //  |/  |/  V
      //  |  /|  /|
      //  | / | / |
      //  |/  |/  V
      for(short i = 0; i < ymin; i++) {
        XYSeq.y = (sharedYSeq[i/4][blockThread] >> (((15-i)%4) * 8)) & 0xFF;
        
        similarity = shmSM[XYSeq.y*lettersCount + XYSeq.x];
        
        F_current = max(F_current - gapEx, AE_current.x - gapOp);
        AE_current.y = max(AE_shared[i][blockThread].y - gapEx, AE_shared[i][blockThread].x - gapOp);

        AE_current.x = max(AE_current.y, F_current);
        AE_current.x = max(AE_current.x, similarity + A_init_upleft.y);
        
        A_init_upleft.y = AE_shared[i][blockThread].x;
        AE_shared[i][blockThread] = AE_current;
      }
      //write variables to global memory for next loop
      short2 AF_tmp;
      AF_tmp.x = AE_current.x;
      AF_tmp.y = F_current;
      AF[startPosA + x * MEM_OFFSET] = AF_tmp;
    }
  }

  //here write result (A_current) to global memory
  scoresDevPtr[startPosA] = AE_current.x;
}

/*******************************************************************************
 * "back" consist of 4bits x 8 (=32bits):                                      *
 * 4bits:                                                                      *
 * -> 0 and 1 bits:                                                            *
 *      - 00 -> stop                                                           *
 *      - 01 -> up                                                             *
 *      - 10 -> left                                                           *
 *      - 11 -> crosswise                                                      *
 * -> 2 bit:                                                                   *
 *      - 0 -> not continueUp                                                  *
 *      - 1 -> continueUp                                                      *
 * -> 3 bit:                                                                   *
 *      - 0 -> not continueLeft                                                *
 *      - 1 -> continueLeft                                                    *
 *                                                                             *
 * BACK:                                                                       *
 * back[startPosA + ( ( ((y) + 8) / 8) * rowWidth + (x) + 1 ) * MEM_OFFSET]    *
 * BACK(-1,-1) => the firs element of -1 row (and -1 column)                   *
 *******************************************************************************/
#define BACK(x,y)   back[startPosA + ( ( ((y) + 8) / 8) * rowWidth + (x) + 1 ) * MEM_OFFSET]

__global__ void
NeedlemanWunschGlobalMatchKernel(short2* AF, unsigned int *back, int *scoresDevPtr, short rowWidth, bool border = false)
{
  // SUBSTITUTION MATRIX GOES TO SHARED MEMORY
  __shared__ char shmSM[SUBSTITUTION_MATRIX_SIZE];
  short idx = threadIdx.y * blockDim.x + threadIdx.x;

  shmSM[idx] = substitutionMatrix[idx];
  idx += BLOCK_SIZE;
  shmSM[idx] = substitutionMatrix[idx];
  idx += BLOCK_SIZE;
  
  if(idx < 576)
    shmSM[idx] = substitutionMatrix[idx];
  
  __syncthreads();
  
  /***************************************************************************
   * |\xxx                                                                   *
   * | \xx    we do not compute x                                            *
   * |  \x                                                                   *
   * |___\                                                                   *
   ***************************************************************************/
  if(border && (seqXNo > seqYNo))
    return;

  int blockThread = threadIdx.x + threadIdx.y * blockDim.x; //0...(BLOCK_SIZE-1)

  short2 lengthXY;
  lengthXY.x = tex1Starts[seqXNo + 1] - startX;
  lengthXY.y = tex2Starts[seqYNo + 1] - startY;

  if((lengthXY.x == 0) || (lengthXY.y == 0))//if there is nothing to do -> quit
    return;

  //startPosA == thread number within whole grid
  int startPosA = seqYNo * WIN_SIZE + seqXNo;

  //initialization of the -1 row in A matrix
  // - 2 bytes for element of A matrix
  // - 2 bytes for element of F matrix
  for(short x = 0; x < lengthXY.x; x++) {
    short2 tmp;
    //(x + 1) because the first element should be -gapEx
    tmp.x = -gapEx * (x + 1);
    tmp.y = SHORT_MIN + gapEx;
    AF[startPosA + x * MEM_OFFSET] = tmp;

    //fill the -1 row of "back" array
    BACK(x,-1) = 9; //0000 0000 0000 0000 0000 0000 0000 1001 == 9
  }

  //fill the -1 column of "back" array
  for(short y = 0; y < lengthXY.y; y+=ALIGNMENT_MATCH_Y_STEPS)
    BACK(-1,y) = 1717986918; //0110 0110 0110 0110 0110 0110 0110 0110 = 1717986918

  BACK(-1,-1) = 0; //stop element

  //one element of AE_shared consist of:
  // - one A element
  // - one E element
  __shared__ short2 AE_shared[ALIGNMENT_MATCH_Y_STEPS][BLOCK_SIZE];
  //elements of Y sequence go to sharedYSeq
  __shared__ int sharedYSeq[ALIGNMENT_MATCH_Y_STEPS/4][BLOCK_SIZE];


  short2 AE_current;
  AE_current.x = 0;

  // |
  // |
  // |
  // V
  for (short y = 0; y < lengthXY.y; y += ALIGNMENT_MATCH_Y_STEPS) {
    short2 A_init_upleft;
    A_init_upleft.x = -gapEx * y;

    //initialialization of the -1 column in A matrix
    // - one element of A matrix
    // - one element of E matrix
    for (short i = 0; i < ALIGNMENT_MATCH_Y_STEPS; i++) {
      short2 tmp;
      tmp.x = -gapEx * (y + i + 1);
      tmp.y = SHORT_MIN + gapEx;
      AE_shared[i][blockThread] = tmp;
    }


    //we read elements of the Y sequence
    for (short i = 0; i < ALIGNMENT_MATCH_Y_STEPS/4; i++) {
      sharedYSeq[i][blockThread] = PACK_BYTES(tex1Dfetch(texSeqsY, startY + y + i * 4 + 0),
                                              tex1Dfetch(texSeqsY, startY + y + i * 4 + 1),
                                              tex1Dfetch(texSeqsY, startY + y + i * 4 + 2),
                                              tex1Dfetch(texSeqsY, startY + y + i * 4 + 3));
    }


    //------>
    for (short x = 0; x < lengthXY.x; x++) {
      //actual up_left gets a value of recent read value from the global memory
      //and actual read value is stored in first two bites of A_upleft
      A_init_upleft.y = A_init_upleft.x;

      char2 XYSeq;
      XYSeq.x = tex1Dfetch(texSeqsX, startX + x);

      //read from global memory
      short2 AF_up = AF[startPosA + x * MEM_OFFSET];

      //A_init -> up element read in previous iteration from global memory (up-left)
      A_init_upleft.x = AF_up.x;
      int F_up;// = AF_up.y;
      AE_current.x = AF_up.x;

      //short2 AE_left;
      int F_current = AF_up.y;
      //int F_up;
      int similarity;
      unsigned int back8 = 0;
      short ymin = min(ALIGNMENT_MATCH_Y_STEPS, lengthXY.y - y); //(i < ALIGNMENT_MATCH_Y_STEPS) && (i + y < lengthY)
      //  |  /|  /|
      //  | / | / |
      //  |/  |/  V
      //  |  /|  /|
      //  | / | / |
      //  |/  |/  V
      for(short i = 0; i < ymin; i++) {
        //AE_left = AE_shared[i][blockThread];

        XYSeq.y = (sharedYSeq[i/4][blockThread] >> (((15-i)%4) * 8)) & 0xFF;

        similarity = max(F_current - gapEx, AE_current.x - gapOp);
        F_up = (similarity==F_current-gapEx);
        F_current = similarity;
        //similarity = substitutionMatrix[XYSeq.y*lettersCount + XYSeq.x];
        similarity = shmSM[XYSeq.y*lettersCount + XYSeq.x];
        similarity += A_init_upleft.y;

        AE_current.y = max(AE_shared[i][blockThread].y - gapEx, AE_shared[i][blockThread].x - gapOp);
        //F_current = max(F_up - gapEx, AE_current.x - gapOp);

        AE_current.x = max(AE_current.y, F_current);
        AE_current.x = max(AE_current.x, similarity);

        //"back" array
        back8 <<= 1;
        back8 |= ((AE_current.x==AE_current.y) && (AE_current.x!=F_current)) || (AE_current.x==similarity); //if go left
        back8 <<= 1;
        back8 |= (AE_current.x==F_current) || (AE_current.x==similarity); //if go up
        back8 <<= 1;
        back8 |= F_up;//(F_current == (F_up - gapEx)); //if continue up
        back8 <<= 1;
        back8 |= (AE_current.y == (AE_shared[i][blockThread].y - gapEx)); //if continue left

        //initialize variables for next iterations
        //short2 AE_tmp;
        //AE_tmp.x = AF_current.x;
        //AE_tmp.y = E_current;
        A_init_upleft.y = AE_shared[i][blockThread].x;
        AE_shared[i][blockThread] = AE_current;
        //A_init_upleft.y = AE_left.x;
        //F_up = F_current;

      }

      //we want the last row of back8 to be completed
      back8 <<= 4 * (ALIGNMENT_MATCH_Y_STEPS - ymin);


      short2 AF_tmp;
      AF_tmp.x = AE_current.x;
      AF_tmp.y = F_current;
      //write variables to global memory for next loop
      AF[startPosA + x * MEM_OFFSET] = AF_tmp;

      BACK(x,y) = back8;
    }
  }

  //here write result (AF_current) to global memory
  scoresDevPtr[startPosA] = AE_current.x;
}

/*******************************************************************************
 * "back" consist of 4bits x 8 (=32bits):                                      *
 * 4bits:                                                                      *
 * -> 0 and 1 bits:                                                            *
 *      - 00 -> stop                                                           *
 *      - 01 -> up                                                             *
 *      - 10 -> left                                                           *
 *      - 11 -> crosswise                                                      *
 * -> 2 bit:                                                                   *
 *      - 0 -> not continueUp                                                  *
 *      - 1 -> continueUp                                                      *
 * -> 3 bit:                                                                   *
 *      - 0 -> not continueLeft                                                *
 *      - 1 -> continueLeft                                                    *
 *                                                                             *
 *******************************************************************************/

#define STOP         0
#define UP           4
#define LEFT         8
#define CROSSWISE   12
#define DIRECTION   12
#define CONTIN_UP    2
#define CONTIN_LEFT  1
#define ELEMENT     15

__global__ void
NeedlemanWunschGlobalBackKernel(unsigned int *back, short rowWidth, unsigned int *matchesX, unsigned int *matchesY, bool border = false)
{
  if(border && (seqXNo > seqYNo))
    return;

  short2 lengthXY;
  lengthXY.x = tex1Starts[seqXNo + 1] - startX;
  lengthXY.y = tex2Starts[seqYNo + 1] - startY;

  if((lengthXY.x == 0) || (lengthXY.y == 0))//if there is nothing to do -> quit
    return;

  //startPosA == thread number within whole grid
  int startPosA = seqYNo * WIN_SIZE + seqXNo;

  short2 indexXY;
  indexXY.x = lengthXY.x - 1; //lengthX (-1 because of addressing in BACK(x,y))
  indexXY.y = lengthXY.y - 1; //lengthY
  
  unsigned int back8 = BACK(indexXY.x, indexXY.y);

  short carret = 0;
  unsigned char prevDirection = CROSSWISE;// 1100 == 12 =>crosswise
  unsigned char back1; //current element of back array
  unsigned char todo;

  unsigned int tmpMatchX;
  unsigned int tmpMatchY;

  back8 >>= ((8 - ((indexXY.y + 1) % 8)) % 8) * 4;

  back1 = back8 & ELEMENT;
  back8 >>= 4;

  while(back1 & DIRECTION) { //while(direction != STOP)

    if( ((prevDirection & DIRECTION) == UP) && (prevDirection & CONTIN_UP) )
      todo = UP;
    else if( ((prevDirection & DIRECTION) == LEFT) && (prevDirection & CONTIN_LEFT) )
      todo = LEFT;
    else if ((back1 & DIRECTION) == UP)
      todo = UP;
    else if ((back1 & DIRECTION) == LEFT)
      todo = LEFT;
    else todo = CROSSWISE;
    
    tmpMatchY <<= 8;
    tmpMatchX <<= 8;

    if (todo == LEFT) {
      tmpMatchY |= (unsigned char)'-';
      tmpMatchX |= revConvConst[tex1Dfetch(texSeqsX, startX + indexXY.x)];

      indexXY.x--;
      back8 = BACK(indexXY.x, indexXY.y);
      back8 >>= ((8 - ((indexXY.y + 1) % 8)) % 8) * 4; //because of the last row of back array
    }
    else if (todo == UP) {
      tmpMatchX |= (unsigned char)'-';
      tmpMatchY |= revConvConst[tex1Dfetch(texSeqsY, startY + indexXY.y)];

      indexXY.y--;
      if((indexXY.y % 8) == 7)
        back8 = BACK(indexXY.x, indexXY.y);
    }
    else {
      tmpMatchX |= revConvConst[tex1Dfetch(texSeqsX, startX + indexXY.x)];
      tmpMatchY |= revConvConst[tex1Dfetch(texSeqsY, startY + indexXY.y)];

      indexXY.x--;
      indexXY.y--;

      back8 = BACK(indexXY.x, indexXY.y);
      back8 >>= ((8 - ((indexXY.y + 1) % 8)) % 8) * 4; //because of the last row of back array
    }    

    prevDirection = todo | back1&3;
    back1 = back8 & ELEMENT;
    back8 >>= 4;

    carret++;
    if( !(carret % 4) ) {
      //save results to global memory
      matchesX[startPosA + (carret/4 - 1) * MEM_OFFSET] = tmpMatchX;
      matchesY[startPosA + (carret/4 - 1) * MEM_OFFSET] = tmpMatchY;
      tmpMatchX = 0;
      tmpMatchY = 0;
    }
  }
  
  tmpMatchX <<= 8;
  tmpMatchY <<= 8;
  tmpMatchX |= (unsigned char)0;//end of match
  tmpMatchY |= (unsigned char)0;//end of match
  tmpMatchX <<= ((4 - ((carret + 1) % 4)) % 4) * 8;
  tmpMatchY <<= ((4 - ((carret + 1) % 4)) % 4) * 8;

  carret+=4;
  matchesX[startPosA + (carret/4 - 1) * MEM_OFFSET] = tmpMatchX;
  matchesY[startPosA + (carret/4 - 1) * MEM_OFFSET] = tmpMatchY;
}

__global__ void
reorderMatchesMemory(unsigned int *inMatches, unsigned int *outMatches, int maxMatchLength)
{ 
  // blocks must be launched in a grid with shape: dim3(n,1,1)
  //  ____   ____   ____   ____
  // |____| |____| |____| |____| ...
  //
  // each block transcripts 16 sequences using shared memory

  //the number of sequence that is transcripted by this thread
  int seqNoRead = blockIdx.x * BLOCK_X_SIZE + threadIdx.x;
  int seqNoWrite = blockIdx.x * BLOCK_X_SIZE + threadIdx.y;

  //BLOCK_X_SIZE + 1 -> to avoid bank conflicts
  __shared__ unsigned int shmMatches[BLOCK_X_SIZE][BLOCK_X_SIZE + 1];

  unsigned int fetch;
  unsigned int fetch2;

  //14 -> 0, 15 -> 0, 16 -> 16, 17 -> 16 ...
  int end = (maxMatchLength / BLOCK_X_SIZE) * BLOCK_X_SIZE;

  //main loop
  for (int i = 0; i < end; i += BLOCK_X_SIZE) {
    fetch = inMatches[seqNoRead + (i + threadIdx.y) * MEM_OFFSET];
    //changing the order of bytes in int
    fetch2 = fetch & 0xFF;
    fetch2 <<= 8;
    fetch >>= 8;
    fetch2 |= fetch & 0xFF;
    fetch2 <<= 8;
    fetch >>= 8;
    fetch2 |= fetch & 0xFF;
    fetch2 <<= 8;
    fetch >>= 8;
    fetch2 |= fetch & 0xFF;

    shmMatches[threadIdx.y][threadIdx.x] = fetch2;

    __syncthreads();

    outMatches[seqNoWrite * maxMatchLength + i + threadIdx.x] = shmMatches[threadIdx.x][threadIdx.y];

    __syncthreads();
  }

  //transcripting the end of sequecne (if maxMatchLength % BLOCK_X_SIZE != 0)
  if (end + threadIdx.y < maxMatchLength) {
    fetch = inMatches[seqNoRead + (end + threadIdx.y) * MEM_OFFSET];
    fetch2 = fetch & 0xFF;
    fetch2 <<= 8;
    fetch >>= 8;
    fetch2 |= fetch & 0xFF;
    fetch2 <<= 8;
    fetch >>= 8;
    fetch2 |= fetch & 0xFF;
    fetch2 <<= 8;
    fetch >>= 8;
    fetch2 |= fetch & 0xFF;

    shmMatches[threadIdx.y][threadIdx.x] = fetch2;
  }

  __syncthreads();

  if (end + threadIdx.x < maxMatchLength)
    outMatches[seqNoWrite * maxMatchLength + end + threadIdx.x] = shmMatches[threadIdx.x][threadIdx.y];
}

#undef STOP
#undef UP
#undef LEFT
#undef CROSSWISE
#undef DIRECTION
#undef CONTIN_UP
#undef CONTIN_LEFT
#undef ELEMENT

#undef WIN_SIZE
#undef MEM_OFFSET
#undef BLOCK_X_SIZE
#undef BLOCK_SIZE
#undef seqXNo
#undef seqYNo
#undef startX
#undef startY

#undef BACK

void
ScoreKernel(dim3 grid, dim3 block, short2 *AF, int *scoresDevPtr, bool border)
{
  NeedlemanWunschGlobalScoreKernel<<<grid, block>>>(AF, scoresDevPtr, border);
  cudaDeviceSynchronize();
}

void
MatchKernel(dim3 grid, dim3 block, short2 *AF, unsigned int *back, int *scoresDevPtr, short rowWidth, bool border)
{
  NeedlemanWunschGlobalMatchKernel<<<grid, block>>>(AF, back, scoresDevPtr, rowWidth, border);
  cudaDeviceSynchronize();
}

void
ReorderMatches(dim3 grid, dim3 block, unsigned int *inX, unsigned int *outX, unsigned int *inY, unsigned int *outY, int maxMatchLength)
{
  reorderMatchesMemory<<<grid, block>>>(inX, outX, maxMatchLength);
  cudaDeviceSynchronize();
  reorderMatchesMemory<<<grid, block>>>(inY, outY, maxMatchLength);
  cudaDeviceSynchronize();
}

void
BacktraceKernel(dim3 grid, dim3 block, unsigned int *back, short rowWidth, unsigned int *matchesX, unsigned int *matchesY, bool border)
{
  NeedlemanWunschGlobalBackKernel<<<grid, block>>>(back, rowWidth, matchesX, matchesY, border);
  cudaDeviceSynchronize();
}