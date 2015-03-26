#ifndef DEFINITIONS_CUH
#define DEFINITIONS_CUH

#include <cstdio>
#include <vector>
#include <string>
#include <cstring>
#include <iostream>
#include <algorithm>

#include <cuda.h>
#include <vector_types.h>
#include <cuda_runtime.h>
#include <texture_types.h>

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

/*******************************************************************************
 * BLOCK_SHAPE defines the width and the height of the one block of threads.   *
 * Each thread corresponds to one pair of sequences and calculates one         *
 * one alignment. It's a good practice to have it multiple of 8. But the best  *
 * results should give 16.                                                     *
 *******************************************************************************/

#define ALIGNMENT_BLOCK_SHAPE    16

/*******************************************************************************
 * BLOCK_SIZE defines the count of the threads runs in the multiprocessor in   *
 * the one block. We will threat it as a square and so it should have a value: *
 * BLOCK_SIZE = BLOCK_SHAPE^2                                                  *
 *******************************************************************************/

#define ALIGNMENT_BLOCK_SIZE    (ALIGNMENT_BLOCK_SHAPE * ALIGNMENT_BLOCK_SHAPE)

/*******************************************************************************
 * Y_STEPS defines the number of steps within the sequence Y which will be     *
 * computed in one iteration without an access to the global memory.           *
 * Maximum Y_STEPS value is 12 (as long as size of the shared memory size is   *
 * 16KB per multiprocessor - see CUDA documentation).                          *
 *                                                                             *
 *                             !!!! WARNING !!!!                               *
 *    In contrast to the ALIGNMENT_SCORE_Y_STEPS the ALIGNMENT_MATCH_Y_STEPS   *
 *            CANNOT be modify without changes in the implementation!          *
 *******************************************************************************/

#define ALIGNMENT_MATCH_Y_STEPS         8
#define ALIGNMENT_SCORE_Y_STEPS        12

/*******************************************************************************
 * SUBSTITUTION_MATRIX_SIZE defines maximum size of the substitution matrix    *
 * which resides in constant memory of the card and contains the rate values   *
 * at which one character in a sequence changes to other character in second   *
 * sequence. 24*24=576                                                         *
 *******************************************************************************/

#define SUBSTITUTION_MATRIX_SIZE 576
  
/*******************************************************************************
 * MAX_ALGORITHM_WINDOW_SIZE defines maximum value of the WINDOW_SIZE and      *
 * determines the count of start position of the sequences in the continous    *
 * block of the memory in stored in a texture (See CUDA documentation).        *
 *******************************************************************************/

#define MAX_ALGORITHM_WINDOW_SIZE 512

/*******************************************************************************
 * MAX_LETTERS_COUNT defines maximum count of different characters in          *
 * substitutionMatrix.                                                         *
 *******************************************************************************/

#define MAX_LETTERS_COUNT   48

/*******************************************************************************
 * The RESHAPE_MEMORY_BLOCK_SIZE refere to the width and height of the shared  *
 * memory in one half-warp which should be the same value as thread count in   *
 * kernel invocation. This value shouldn't be modified.                        *
 *******************************************************************************/

#define ALIGNMENT_MATCH_BLOCK_X_SIZE   16

/*******************************************************************************
 * Function reorderMatchesMemory should be used to change order of global      *
 * memory allocated to the matches in the way that the consecutive letters of  *
 * the sequence to be in successive global memory cells.                       *
 *******************************************************************************/

#define BLOCK_X_SIZE  ALIGNMENT_MATCH_BLOCK_X_SIZE

/*******************************************************************************
 * substitutionMatrix resides in constant memory of the graphic card and       *
 * contains the rate values at which one character in a sequence changes to    *
 * other character (e.g. BLOSUM62).                                            *
 *******************************************************************************/

__constant__ char substitutionMatrix[SUBSTITUTION_MATRIX_SIZE] = {
 4, -1, -2, -2,  0, -1, -1,  0, -2, -1, -1, -1, -1, -2, -1,  1,  0, -3, -2,  0, -2, -1,  0, -4,
-1,  5,  0, -2, -3,  1,  0, -2,  0, -3, -2,  2, -1, -3, -2, -1, -1, -3, -2, -3, -1,  0, -1, -4,
-2,  0,  6,  1, -3,  0,  0,  0,  1, -3, -3,  0, -2, -3, -2,  1,  0, -4, -2, -3,  3,  0, -1, -4,
-2, -2,  1,  6, -3,  0,  2, -1, -1, -3, -4, -1, -3, -3, -1,  0, -1, -4, -3, -3,  4,  1, -1, -4,
 0, -3, -3, -3,  9, -3, -4, -3, -3, -1, -1, -3, -1, -2, -3, -1, -1, -2, -2, -1, -3, -3, -2, -4,
-1,  1,  0,  0, -3,  5,  2, -2,  0, -3, -2,  1,  0, -3, -1,  0, -1, -2, -1, -2,  0,  3, -1, -4,
-1,  0,  0,  2, -4,  2,  5, -2,  0, -3, -3,  1, -2, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
 0, -2,  0, -1, -3, -2, -2,  6, -2, -4, -4, -2, -3, -3, -2,  0, -2, -2, -3, -3, -1, -2, -1, -4,
-2,  0,  1, -1, -3,  0,  0, -2,  8, -3, -3, -1, -2, -1, -2, -1, -2, -2,  2, -3,  0,  0, -1, -4,
-1, -3, -3, -3, -1, -3, -3, -4, -3,  4,  2, -3,  1,  0, -3, -2, -1, -3, -1,  3, -3, -3, -1, -4,
-1, -2, -3, -4, -1, -2, -3, -4, -3,  2,  4, -2,  2,  0, -3, -2, -1, -2, -1,  1, -4, -3, -1, -4,
-1,  2,  0, -1, -3,  1,  1, -2, -1, -3, -2,  5, -1, -3, -1,  0, -1, -3, -2, -2,  0,  1, -1, -4,
-1, -1, -2, -3, -1,  0, -2, -3, -2,  1,  2, -1,  5,  0, -2, -1, -1, -1, -1,  1, -3, -1, -1, -4,
-2, -3, -3, -3, -2, -3, -3, -3, -1,  0,  0, -3,  0,  6, -4, -2, -2,  1,  3, -1, -3, -3, -1, -4,
-1, -2, -2, -1, -3, -1, -1, -2, -2, -3, -3, -1, -2, -4,  7, -1, -1, -4, -3, -2, -2, -1, -2, -4,
 1, -1,  1,  0, -1,  0,  0,  0, -1, -2, -2,  0, -1, -2, -1,  4,  1, -3, -2, -2,  0,  0,  0, -4,
 0, -1,  0, -1, -1, -1, -1, -2, -2, -1, -1, -1, -1, -2, -1,  1,  5, -2, -2,  0, -1, -1,  0, -4,
-3, -3, -4, -4, -2, -2, -3, -2, -2, -3, -2, -3, -1,  1, -4, -3, -2, 11,  2, -3, -4, -3, -2, -4,
-2, -2, -2, -3, -2, -1, -2, -3,  2, -1, -1, -2, -1,  3, -3, -2, -2,  2,  7, -1, -3, -2, -1, -4,
 0, -3, -3, -3, -1, -2, -2, -3, -3,  3,  1, -2,  1, -1, -2, -2,  0, -3, -1,  4, -3, -2, -1, -4,
-2, -1,  3,  4, -3,  0,  1, -1,  0, -3, -4,  0, -3, -3, -2,  0, -1, -4, -3, -3,  4,  1, -1, -4,
-1,  0,  0,  1, -3,  3,  4, -2,  0, -3, -3,  1, -1, -3, -1,  0, -1, -3, -2, -2,  1,  4, -1, -4,
 0, -1, -1, -1, -2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -2,  0,  0, -2, -1, -1, -1, -1, -1, -4,
-4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4, -4,  1
};

/*******************************************************************************
 * lettersCount contains the count of the different characters in substitution *
 * matrix.                                                                     *
 *******************************************************************************/

__constant__ int  lettersCount = 24;

/*******************************************************************************
 *
 *******************************************************************************/

__constant__ unsigned char revConvConst[MAX_LETTERS_COUNT] = {
  65,  82,  78,  68,  67,  81,  69,  71,  72,  73,  76,  75,  77,  70,  80,  83,  84,
  87,  89,  86,  66,  90,  88,  42
};
    
/*******************************************************************************
 * WINDOW_SIZE defines the size of the window within the alignment algorithms. *
 * The value indicates the width as well as the height of the space in the     *
 * sequences pair matrix that is computed in one graphic card in one           *
 * iteration.                                                                  *
 *                                                                             *
 * NOTICE:                                                                     *
 * WINDOW_SIZE must be multiple of BLOCK_SHAPE e.g. 128, 144, 160              *
 * to satisfy kernel's needs.                                                  *
 *                                                                             *
 * NOTICE2:                                                                    *
 * WINDOW_SIZE should be smaller than MAX_ALGORITHM_WINDOW_SIZE                *
 * (see data_management.cuh)                                                   *
 *                                                                             *
 * ALIGNMENT_SCORE_WINDOW_SIZE referes to the size of the window in algorithms *
 * that as a result has a scalar value of the alignment score.                 *
 *                                                                             *
 * ALIGNMENT (SCORE ONLY) best WINDOW_SIZE:                                    *
 *  - GTS250:   128                                                            *
 *  - 9600GT:    96??                                                          *
 *  - GTX280:   240                                                            *
 *  - 8600MGS:  128                                                            *
 *                                                                             *
 * ALIGNMENT WITH BACKTRACKING best WINDOW_SIZE (3_667.fax):                   *
 *  - GTS250:      64                                                          *
 *  - 9600GT:      64                                                          *
 *  - GTX280:      96                                                          *
 *  - 8600MGS:     32                                                          *
 *  - TeslaC1060: 112                                                          *
 *******************************************************************************/

__constant__ unsigned int  window;

/*******************************************************************************
 * MEMORY_OFFSET defines offset between the cells of global memory resides on  *
 * the graphic card. It helps in setting and retrieving consecutive values by  *
 * consecutive threads and speed it up significantly (see also CUDA            *
 * documentation).                                                             *
 * MEMORY_OFFSET == WINDOW_SIZE * WINDOW_SIZE                                  *
 *******************************************************************************/

__constant__ unsigned int  offset;

/*******************************************************************************
 * gapOp and gapEx contain the value of penalty for respectively opening and   *
 * extension of the gap in alignment algorithm. Resides in constant memory.    *
 *******************************************************************************/

__constant__ char gapOp;
__constant__ char gapEx;

/*******************************************************************************
 * tex1Starts and tex2Starts contain start positions' sequences in the raw     *
 * memory of the texture.                                                      *
 *******************************************************************************/

__constant__ int  tex1Starts[MAX_ALGORITHM_WINDOW_SIZE];
__constant__ int  tex2Starts[MAX_ALGORITHM_WINDOW_SIZE];

/*******************************************************************************
 * texSeqs1 and texSeqs2 contain sequences in a raw memory of the texture.     *
 *                                                                             *
 * t  -------------                                                            *
 * e |             |                                                           *
 * x |  alignment  |                                                           *
 * S |             |                                                           *
 * e |             |                                                           *
 * q |   matrix    |                                                           *
 * s |             |                                                           *
 * Y  ------------                                                             *
 *  t e x S e q s X                                                            *
 *                                                                             *
 * texBack - texture with data needed to backtrace in NW & SW algorithms       *
 *******************************************************************************/

#ifdef __CUDACC__

texture<char, 1, cudaReadModeElementType> texSeqsX;
texture<char, 1, cudaReadModeElementType> texSeqsY;
texture<unsigned int, 1, cudaReadModeElementType> texBack;

#endif

/*******************************************************************************
 * TexVariablesAddresses is a result type of the function copySeqsToTex and    *
 * contains pointers (to free) of a global memory of the graphic card.         *
 *******************************************************************************/

struct TexVariablesAddresses
{
  char* texSeqs1DevPtr;
  char* texSeqs2DevPtr;
};

/*******************************************************************************
 *
 *******************************************************************************/

struct seqEntry
{
  int size;
  int partId;
  int windowX;
  int windowY;
  int blockShape;
  int windowSize;
  int windowMaxX;
  int windowMaxY;
  int windowSumX;
  int windowSumY;
  int maxMultiprocessorCount;

  /*******************************************************************************
 * AlignmentInvokerParams::getEstimatedCompexity() implements the              *
 * functionality of estimation of the complexity of the problem contained in   *
 * the, corresponding to the AlignmentInvokerParams window.                    *
 * Good approximation of the value is a product of the max length on X         *
 * position and on Y position.                                                 *
 *                                                                             *
 * Cpx = max[L(X)]*max[L(Y)]                                                   *
 *                                                                             *
 * Where:                                                                      *
 * L(n) is the length of sequence n                                            *
 *                                                                             *
 *******************************************************************************/

  
  inline long long getEstimatedComplexity()
  {
    long long result;
    int multiprocessorCorrection;

    result = (long long) (windowSumX * windowSumY);

    if(partId == 1) {
      multiprocessorCorrection =  size % windowSize;
      multiprocessorCorrection += blockShape - 1;
      multiprocessorCorrection /= blockShape;
      multiprocessorCorrection *= multiprocessorCorrection;

      if((multiprocessorCorrection < maxMultiprocessorCount) && (multiprocessorCorrection != 0)) {
        result *= maxMultiprocessorCount;
        result /= multiprocessorCorrection;
      }
    }

    result = MAX(result, windowMaxX * windowMaxY * maxMultiprocessorCount); //16 multiprocessors count

    return result;
  }
}; 

#endif
