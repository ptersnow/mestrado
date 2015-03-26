#ifndef SPECIESBLOCK_H
#define SPECIESBLOCK_H

#include <map>
#include <mpi.h>
#include <queue>
#include <cmath>
#include <cstring>
#include <fstream>
#include <sstream>
#include <cstdlib>
#include <iostream>
#include <algorithm>

#include "TreeNode.h"
#include "FastaFile.h"
#include "CmdLineOptions.h"

#include "timer.h"
#include "structs.h"

//#define VERBOSE(s)  if (myProc == 0) std::cout << s
#define VERBOSE(s)  

#define DEBUGPRT(v, s)  if (m_debugFlg == v) std::cout << s

#define CHKERR(ierr) { \
  if(ierr != MPI_SUCCESS) { \
    printf("MPI_Error %d at processor %d: %s %s %d\n", ierr, myProc, __FUNCTION__, __FILE__, __LINE__); \
    MPI_Abort(MPI_COMM_WORLD, ierr); \
  } \
}


typedef std::priority_queue<Dij, std::vector<Dij>, DijCompare>	   PQ_Dij;

class SpeciesBlock
{
public:
  enum {tBuildMtx = 0, tBuildTree, tBuildAlign, tTotal};

  SpeciesBlock(const CmdLineOptions& cmd);
  ~SpeciesBlock();

  void readFasta();
  void buildBins();
  void buildTree();
  void buildAlign();
  void freeMemory();
  void reportTimers();

private:
  void init();
  void setupLinkSystem();
  void buildGblRowSums();
  int findProcForMinIdx(int minI);
  void bcastRow(int minI, std::vector<float>& vd, std::vector<float>& work);
  void modifyMatrix(const QDij & minQ, int & procNumI, int & procNumJ, int riMinL, int riMinG);
  
  bool                              m_saveZeroD;        // Flag to save zero Ds
  bool                              m_dupRemoval;       // Flag to control dup removal.
  std::string                       m_fn;               // Input file name
  std::string                       m_baseFn;           // Base Name from input
  int                               m_debugFlg;         // Debug flag.
  int                               m_n0;               // Number of Global Species at start
  int                               m_n;                // Number of Global Species currently;
  int                               m_l;                // Number of Local Species
  int                               m_nn;               // number of species and merges.
  std::vector<double>               m_T;                // Global T vector
  std::vector< std::vector<float> > m_distances;        // Vector of distances local to this processor.
  double                            m_Tmin, m_Tmax;     // Global Tmin and max
  std::vector<int>                  m_speciesCount;     // number of Species on each processor
  std::vector<int>                  m_speciesOffset;    // The number of Species before this each processor.

  std::vector<int>                  m_binLoc;           // Maps T[i] into 1d bin.
  std::vector<double>               m_binRange;         // Range on each bin (max range)

  std::vector<int>                  m_nextActiveNodeG;  // indices for next active or -1 for inactive
  std::vector<int>                  m_prevActiveNodeG;  // indices for prev active or -1 for inactive
  std::vector<int>                  m_nextActiveNodeL;  // indices for next active or -1 for inactive
  std::vector<int>                  m_prevActiveNodeL;  // indices for prev active or -1 for inactive
  std::vector<int>                  m_redirectG;        // Points to next active node (Global).
  std::vector<int>                  m_redirectL;        // Points to next active node (local).
  std::vector<int>                  m_g2l;              // Convert from global to local numbers
  std::vector<int>                  m_l2g;              // Convert from local to global numbers

  std::vector<PQ_Dij>               m_pqA;              // A vector of priority queues of Dij
  std::vector<int>                  m_binOrder;         // Order of bins sorted in descending maxTSum order.

  std::vector<TreeNode>             m_nodes;            // The tree of life results.
  std::vector<int>                  m_sendProcA;        // Map from global bin to owning proc.
  std::vector<int>                  m_recvProcA;        // Map from recv buff to local bin number.
  std::vector<int>                  m_sendTagA;         // Mpi tag for each send buf.
  std::vector<NumBinPair>           m_numBinPairA;      // Map from local bins to 2D indices.
  int                               m_numBins;          // number of bin in 1d.
  int                               m_binT;             // Number of global bins.
  int                               m_binL;             // Number of local bins.
  int                               m_nRecvBuf;         // Number of Receive buffers
  int                               m_bin0;             // The first bin on this processor

  int                               maxSeqLen;          // 
  std::vector<int>                  seqsLen;            // 
  std::vector< std::string >        seqsName;           // Global array of sequences names.
  std::vector< std::string >        seqs;               // Global array of sequences.

  int                               m_firstActiveNodeG;
  int                               m_firstActiveNodeL;
  int                               m_nextInternalNodeG;
  int                               m_nextInternalNodeL;
  int                               m_numBuildBinCalls;
  float                             m_ratio;            // Rebuild ratio
  float                             m_pqRatio;          // Ratio to rebuild bins when PQ memory usage is too high

  int                               m_QDA;              // Number of q sent between processors.
  bool                              m_predictQ;         // flag for Q prediction
  bool                              m_allreduce;        // use MPI_Allreduce instead of MPI_Allgather
  float                             m_smallD;           // if smaller than this treat as zero.

  std::vector< std::vector<int> >   m_idxA;
  std::vector< float >              m_vd;               // New distances for parent row.
  std::vector< Pair >               m_Dzero;            // locations with zero D's
  timer totalTime;
  timer m_Timer[4];    // Timers
};

#endif	//SPECIESBLOCK_H
