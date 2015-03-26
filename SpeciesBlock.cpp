#include "SpeciesBlock.h"
#include "AlignmentScore.cuh"
#include "AlignmentMatch.cuh"

const int minRebuild  = 200;
const int QMERGE_SZ   = 4096;

extern int myProc, nProcs;

const char * tTitle[] = { "BuildMtx", "BuildTree", "BuildAlign" };

template<typename T, typename cmp>
void
makeEmpty(std::priority_queue<T, std::vector<T>, cmp>& q)
{
  std::priority_queue<T, std::vector<T>, cmp> empty;
  q = empty;
}

bool
activeTest(const ActiveDij& adij)
{
  return adij.active;
}

void
myMergeSort(void *inV, void *inoutV, int *len, MPI_Datatype *dptr)
{
  static QDij q[QMERGE_SZ];
  QDij*       in    = (QDij*)inV;
  QDij*       inout = (QDij*)inoutV;
  int n = *len;

  for (int i = 0; i < n; ++i)
    q[i] = inout[i];

  int i = 0;
  int j = 0;
  for (int k = 0; k < n; ++k) {
    if(q[i].q < in[j].q)
      inout[k] = q[i++];
    else inout[k] = in[j++];
  }
}

void
myQminFun(void *inV, void* inoutV, int * len, MPI_Datatype *dptr)
{
  QDij* in    = (QDij*) inV;
  QDij* inout = (QDij*) inoutV;

  if (in->q < inout->q)
    *inout = *in;
  else if (fabs(in->q - inout->q) < 1.e-12) {
    int lhs_ii = MIN(in->i, in->j);
    int lhs_jj = MAX(in->i, in->j);
    int rhs_ii = MIN(inout->i, inout->j);
    int rhs_jj = MAX(inout->i, inout->j);
    if ((lhs_ii > rhs_ii) || ((lhs_ii == rhs_ii) && lhs_jj > rhs_jj))
      *inout = *in;
  }
}

SpeciesBlock::SpeciesBlock(const CmdLineOptions& cmd)
  : m_saveZeroD(true),         m_dupRemoval(cmd.dupRemoval), m_fn(cmd.inputFile),
    m_baseFn(cmd.baseName),    m_debugFlg(cmd.debugFlg),     m_n0(0),
    m_n(0),          m_l(0),           m_T(0),
    m_numBins(cmd.clustSize),  m_firstActiveNodeG(0),      m_numBuildBinCalls(0),
    m_ratio(cmd.rebuildRatio), m_pqRatio(cmd.pqRatio),       m_QDA(cmd.qda),
    m_predictQ(cmd.predictQ),  m_allreduce(cmd.allreduce),   m_vd(0),
    m_Dzero(0)
{
  int numBinT, rem;

  totalTime.start();
  totalTime.accTime = 0.0;

  numBinT = (m_numBins + 1) * m_numBins / 2;
  if (numBinT < nProcs) {
    m_numBins = (int) ((-1.0 + sqrt(1.0 + 8.0 * nProcs)) / 2.0);
    while (numBinT < nProcs) {
      m_numBins++;
      numBinT = (m_numBins + 1) * m_numBins / 2;
    }
  }

  m_smallD = 1.000e-09;
  m_QDA = MAX(  2, m_QDA);
  m_QDA = MIN(256, m_QDA);
  m_binT = m_numBins * (m_numBins + 1) / 2;
  rem = m_binT % nProcs;

  m_binL     = (m_binT / nProcs) + (myProc < rem);
  m_bin0     = myProc * (m_binT / nProcs) + ((myProc < rem) ? myProc : rem);
  m_nRecvBuf = m_binL * nProcs;

  VERBOSE("-- Leaving SpeciesBlock ctor" << std::endl);
}

SpeciesBlock::~SpeciesBlock()
{
}

void
SpeciesBlock::reportTimers()
{
  if(totalTime.accTime == 0.0)
    totalTime.end();

  if(myProc == 0) {
    for (int k = 0; k < tTotal; ++k)
      fprintf(stderr, "%5.4f,", m_Timer[k].accTime);
    
    fprintf(stderr, "%5.4f\n", totalTime.accTime);
  }
}

void
SpeciesBlock::init()
{
  m_nextInternalNodeG = m_n;
  m_binLoc.resize(m_n);
  m_binOrder.resize(m_binL);
  m_pqA.resize(m_binL);
  m_T.resize(m_n);
  m_vd.resize(m_n);
  m_nn = 2 * m_n - 1;
  m_nextActiveNodeG.resize(m_nn);
  m_prevActiveNodeG.resize(m_nn);
  m_nextActiveNodeL.resize(m_nn);
  m_prevActiveNodeL.resize(m_nn);
  m_redirectG.resize(m_nn);
  m_nodes.resize(m_nn);
  //Build [[m_idxA]] 2-D array for loading distance send buffers
  int gbin = 0;
  m_binRange.resize(m_numBins);
  m_idxA.resize(m_numBins);
  for (int i = 0; i < m_numBins; ++i)
    m_idxA[i].resize(m_numBins);

  for (int i = 0; i < m_numBins; ++i)
    for (int j = i; j < m_numBins; ++j) {
      m_idxA[i][j] = gbin;
      m_idxA[j][i] = gbin++;
    }

  m_g2l.resize(m_nn);
  m_g2l.assign(m_nn, -1);

  for (int i = 0; i < m_nn; ++i) {
    m_nextActiveNodeG[i] = i + 1;
    m_nextActiveNodeL[i] = i + 1;
    m_prevActiveNodeG[i] = i - 1;
    m_prevActiveNodeL[i] = i - 1;
  }

  for (int i = 0; i < m_n; ++i)
    m_redirectG[i] = i;

  for (int i = m_n; i < m_nn; ++i)
    m_redirectG[i] = -1;

  //Compute where to send/recv bin data to what processor
  m_sendProcA.resize(m_binT);
  m_recvProcA.resize(m_nRecvBuf);
  m_sendTagA.resize(m_binT);

  for (gbin = 0; gbin < m_binT; ++gbin) {
    int iproc = gbin % nProcs;
    m_sendProcA[gbin]  = iproc;
  }

  int last  = m_binL;
  int iproc = 0;
  for (int rbin = 0; rbin < m_nRecvBuf; ++rbin) {
    if (rbin == last) {
      iproc++;
      last += m_binL;
    }
    m_recvProcA[rbin] = iproc;
  }

  int rem = m_binT % nProcs;
  for (iproc = 0; iproc < nProcs; ++iproc) {
    int nbins = (m_binT / nProcs) + (iproc < rem);
    int first = myProc * nbins;
    for (gbin = iproc; gbin < m_binT; gbin += nProcs)
      m_sendTagA[gbin] = first++;
  }

  //Compute Local [[NumBinPair]] vector
  m_numBinPairA.resize(m_binL);
  int lbin = 0;
  for (int i = 0; i < m_numBins; ++i) {
    for (int j = i; j < m_numBins; ++j) {
      gbin = m_idxA[i][j];
      if (m_sendProcA[gbin] == myProc)
        m_numBinPairA[lbin++].assign(i,j);
    }
  }
}

void
SpeciesBlock::setupLinkSystem()
{
  m_speciesOffset.resize(nProcs + 1);
  m_speciesOffset[0] = 0;
  for (int iProc = 0; iProc < nProcs; ++iProc)
    m_speciesOffset[iProc + 1] = m_speciesOffset[iProc] + m_speciesCount[iProc];

  m_firstActiveNodeL = 0;
  m_l2g.resize(m_nn);
  m_l2g.assign(m_nn, -1);
  int k = m_speciesOffset[myProc];
  for (int j = 0; j < m_l; ++j) {
    m_g2l[j+k] = j;
    m_l2g[j]   = j+k;
  }

  m_redirectL.resize(m_nn);
  m_nextInternalNodeL = m_l;

  for (int j = 0; j < m_l; ++j)
    m_redirectL[j] = j;

  for (int j = m_l; j < m_nn; ++j)
    m_redirectL[j] = -1;
}

void
SpeciesBlock::readFasta()
{
  MPI_Status status;
  MPI_Request recvReq;
  AlignmentScore *align;
  int nSpecies, seqLen, szA[2];
  char *p, *q, *tmp, *localBuf, *sendBuf, *recvBuf;
  int m0, iS, jS, rem, iProc, iBuf, nextProc, prevProc, bufSz, recvRowG, recvProc;

  VERBOSE("-- SpeciesBlock::readFasta" << std::endl);

  m_Timer[tBuildMtx].start();

  // Read in fasta formatted sequences.
  if (myProc == 0) {
    //Find size of sequence data on Proc 0
    std::string seq;
    FastaFile *file = new FastaFile(m_fn.c_str());

    nSpecies = maxSeqLen = 0;
    seq = file->nextSeq(&seqLen);
    while(seqLen > 0) {
      // Save sequence
      seqs.push_back(seq);
      // Save sequence name
      seqsName.push_back(file->getSeqName());
      // Save sequence size
      seqsLen.push_back(seqLen);

      nSpecies++;

      if(maxSeqLen < seqLen)
        maxSeqLen = seqLen;

      seq = file->nextSeq(&seqLen);
    }

    delete file;
  }

  //Bcast number of species and size of sequence
  if (myProc == 0) {
    szA[0] = nSpecies;
    szA[1] = maxSeqLen;
  }

  CHKERR(MPI_Bcast(&szA[0], 2, MPI_LONG, 0, MPI_COMM_WORLD));
  
  m_n = m_n0 = (int) szA[0];
  maxSeqLen  = (int) szA[1];

  init();  // Must initialize member vars after knowning [[m_n]]
  //Compute [[m_l]] and other local sized variables
  rem = m_n0 % nProcs;
  m0  = m_n0 / nProcs;
  m_l   = m0 + (myProc < rem);
  m_speciesCount.resize(nProcs);
  
  for (iProc = 0; iProc < nProcs; ++iProc)
    m_speciesCount[iProc] = m0 + (iProc < rem);

  setupLinkSystem();

  //Create Sequence Data Send/Receive buffers
  bufSz = sizeof(int) + (m0 + 1) * (2 * sizeof(int) + (maxSeqLen + 1));
  localBuf = new char[bufSz];
  sendBuf = new char[bufSz];

  //Post Receive Buffer for Sequence Data
  CHKERR(MPI_Irecv(&localBuf[0], bufSz, MPI_CHAR, 0, (7 + (myProc << 3)), MPI_COMM_WORLD, &recvReq));

  if (myProc == 0) {

    int rowG, jProc, iSpecies, lSpecies;

    lSpecies = m_l;
    iProc = iS = iSpecies = 0;

    p = &sendBuf[sizeof(int)];
    memcpy(&sendBuf[0], (char *) &iProc, sizeof(int));

    for (iSpecies = 0; iSpecies < m_n; iSpecies++) {
      // Sequence number
      memcpy(p, (char *) &iSpecies, sizeof(int));
      p += sizeof(int);
      // Sequence length
      memcpy(p, (char *) &seqsLen[iSpecies], sizeof(int));
      p += sizeof(int);
      // Sequence symbol
      memcpy(p, seqs[iSpecies].c_str(), seqsLen[iSpecies]);
      p[seqsLen[iSpecies]]  = '\0';
      p += seqsLen[iSpecies] + 1;

      if (++iS == lSpecies) {
        
        memcpy((char *) &jProc, &sendBuf[0], sizeof(int));
        memcpy((char *) &rowG, &sendBuf[sizeof(int)], sizeof(int));

        CHKERR(MPI_Send(&sendBuf[0], bufSz, MPI_CHAR, iProc, (7 + (iProc << 3)), MPI_COMM_WORLD));

        iS = 0;
        iProc++;
        lSpecies = m0 + (iProc < rem);

        p = &sendBuf[sizeof(int)];
        memcpy(&sendBuf[0], (char *) &iProc, sizeof(int));
      }
    }
    
    //Initialize [[m_nodes]] tree
    for (iSpecies = 0; iSpecies < m_n0; iSpecies++)
      m_nodes[iSpecies].init(iSpecies, seqsName[iSpecies].c_str());
  }
  //Wait for [[localBuf]] for local sequence data
  CHKERR(MPI_Wait(&recvReq, &status));

  memcpy((char *) &recvProc, &localBuf[0], sizeof(int));
  memcpy((char *) &recvRowG, &localBuf[sizeof(int)], sizeof(int));

  if (recvRowG != m_speciesOffset[myProc]) {

    std::cout << "myProc: " << myProc << " recvProc: " << recvProc <<
      " recvRowG: " << recvRowG << " Offset: " << m_speciesOffset[myProc] << std::endl;
    abort();
  }
  m_distances.resize(m_l);

  for(iS = 0; iS < m_l; iS++)
    m_distances[iS].resize(m_n);

  align = new AlignmentScore(10, 2, 16, maxSeqLen);
  align->loadDB(localBuf, m_l);
  align->computeDistanceMtx(localBuf, m_l, m_distances, m_speciesOffset[myProc]);

  memcpy(sendBuf, localBuf, bufSz);
  
  p = sendBuf;
  recvBuf = new char[bufSz];
  q = recvBuf;
  
  nextProc = (myProc + 1) % nProcs;
  prevProc = (myProc + (nProcs - 1)) % nProcs;

  iProc = iBuf = myProc;
  memcpy((char *) &iS, localBuf, sizeof(int));

  for (int ja = 0; ja < nProcs - 1; ++ja) {

    iProc = (iProc + (nProcs - 1) ) % nProcs;

    CHKERR(MPI_Irecv(q, bufSz, MPI_CHAR, prevProc, ((iProc << 3) + 5), MPI_COMM_WORLD, &recvReq));
    
    CHKERR(MPI_Send(p, bufSz, MPI_CHAR, nextProc, ((iBuf << 3) + 5), MPI_COMM_WORLD));
    CHKERR(MPI_Wait(&recvReq, &status));

    memcpy((char *) &jS, &q[sizeof(int)], sizeof(int));
    if (jS != m_speciesOffset[iProc]) {

      std::cout << "myProc: " << myProc << " iProc: " << iProc << " iS: " << iS
         << " m_speciesOffset[myProc]: " << m_speciesOffset[myProc] << " jS: " << jS
         << " Offset: " << m_speciesOffset[iProc] << std::endl;
      abort();
    }

    align->computeDistanceMtx(q, m_speciesCount[iProc], m_distances, m_speciesOffset[iProc]);

    tmp = q;
    q = p;
    p = tmp;

    iBuf = (iBuf + (nProcs - 1) ) % nProcs;
  }
  m_Timer[tBuildMtx].end();

  delete align;

  //Clean up memory allocation
  delete [] localBuf;
  delete [] recvBuf;
  delete [] sendBuf;
}

int
SpeciesBlock::findProcForMinIdx(int minI)
{
  int ri = m_redirectG[minI];
  std::vector<int>::iterator lb = lower_bound(m_speciesOffset.begin(), m_speciesOffset.end(), ri);

  if (*lb != ri)
    lb--;

  int iproc = lb - m_speciesOffset.begin();
  return iproc;
}
void
SpeciesBlock::buildGblRowSums()
{
  std::vector<double> T_l(m_n0);
  T_l.assign(m_n0, 0.0);
  m_T.assign(m_n0, 0.0);

  int idx = m_firstActiveNodeL;
  for (int jd = 0; jd < m_l; ++jd) {
    int      i   = m_l2g[idx];
    int      rii = m_redirectL[idx];
    int      ri  = m_redirectG[i];
    int      j   = m_firstActiveNodeG;
    std::vector<float> d   = m_distances[rii];
    double      sum = 0.0;
    
    while (j < m_nextInternalNodeG) {
      int rj  = m_redirectG[j];
      sum    += d[rj];
      j       = m_nextActiveNodeG[j];
    }

    T_l[ri] = sum;
    
    idx = m_nextActiveNodeL[idx];
  }

  CHKERR(MPI_Allreduce(&T_l[0], &m_T[0], m_n0, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD));
}

void
SpeciesBlock::buildBins()
{
  int i, gbin, idx, count;
  m_numBuildBinCalls++;

  VERBOSE("-- Start building bins" << std::endl);
  //Clear bins
  for (int lbin = 0; lbin < m_binL; ++lbin)
    makeEmpty(m_pqA[lbin]);

  //Compute global [[m_Tmin]], [[m_Tmax]] and range on bins
  m_Tmin = DBL_MAX;
  m_Tmax = -DBL_MAX;
  i  = m_firstActiveNodeG;
  std::vector<double> T(m_n);
  count = 0;
  while (i < m_nextInternalNodeG) {
    int ri = m_redirectG[i];
    T[count++] = m_T[ri];

    if (m_T[ri] > m_Tmax)
      m_Tmax = m_T[ri];

    if (m_T[ri] < m_Tmin)
      m_Tmin = m_T[ri];

    i = m_nextActiveNodeG[i];
  }

  int  rem  = m_n % m_numBins;
  int  rdel = m_n / m_numBins;
  int  ir   = rdel;
  double half = 0.5;
  std::sort(T.begin(), T.end());
  m_binRange[m_numBins - 1] = m_Tmax;
  for (int ibin = 0; ibin < m_numBins - 1; ++ibin) {
    int irNext = MIN(ir + 1, m_n - 1);
    m_binRange[ibin]  = half * (T[ir] + T[irNext]);
    ir += rdel + (rem > ibin);
  }

  //Find all bin location for all entries in [[T]]: [[m_binLoc]]
  i = m_firstActiveNodeG;
  m_binLoc.assign(m_n0, -1);
  std::vector<int> hg(m_numBins);
  hg.assign(m_numBins, 0);

  VERBOSE("-- myProc == " << myProc << " m_firstActiveNodeG == " << m_firstActiveNodeG << std::endl);
  while (i < m_nextInternalNodeG) {
    
    int ri = m_redirectG[i];
    int lb = lower_bound(m_binRange.begin(), m_binRange.end(), m_T[ri]) - m_binRange.begin();

    m_binLoc[ri] = lb;
    i = m_nextActiveNodeG[i];
    hg[lb]++;
  }


  //Push all distances into the appropriate send buffer [[send]]
  std::vector< std::vector<Dij> > send(m_binT);

  VERBOSE("-- Start to build send buffers" << std::endl);

  std::vector<int> cntA(m_binT);
  cntA.assign(m_binT, 0);

  idx = m_firstActiveNodeL;
  for (int jd = 0; jd < m_l; ++jd) {
    i          = m_l2g[idx];
    int      ri  = m_redirectG[i];
    int      j   = m_nextActiveNodeG[i];

    while (j < m_nextInternalNodeG) {
      int    rj = m_redirectG[j];
      int    bi = m_binLoc[ri];
      int    bj = m_binLoc[rj];
      gbin  = m_idxA[bi][bj];
      cntA[gbin]++;
      j = m_nextActiveNodeG[j];
    }
    idx = m_nextActiveNodeL[idx];
  }

  for (gbin = 0; gbin < m_binT; ++gbin)
    send[gbin].reserve(cntA[gbin]);


  idx = m_firstActiveNodeL;
  for (int jd = 0; jd < m_l; ++jd) {
    i          = m_l2g[idx];
    int      ri  = m_redirectG[i];
    int      rii = m_redirectL[idx];
    int      j   = m_nextActiveNodeG[i];
    std::vector<float>& d   = m_distances[rii];

    while (j < m_nextInternalNodeG) {
      int    rj = m_redirectG[j];
      float dd = d[rj];
      int    bi = m_binLoc[ri];
      int    bj = m_binLoc[rj];
      gbin  = m_idxA[bi][bj];
      send[gbin].push_back( Dij(dd,i,j) );
      j = m_nextActiveNodeG[j];
    }
    idx = m_nextActiveNodeL[idx];
  }
  VERBOSE("-- Finish building send buffers" << std::endl);

  //Use all to all to exchange send size to receive buffer size
  std::vector< int > sendSz(m_binT);
  std::vector< int > recvSz(m_nRecvBuf);
  std::vector< int > sendcnts(nProcs);
  std::vector< int > recvcnts(nProcs);
  std::vector< int > sdispls(nProcs+1);
  std::vector< int > rdispls(nProcs+1);

  int binProcRatio = m_binT / nProcs;
  rem      = m_binT % nProcs;

  int sbin = 0;
  for (int iproc = 0; iproc < nProcs; ++iproc) {
    for ( gbin = iproc; gbin < m_binT; gbin += nProcs)
      sendSz[sbin++] = send[gbin].size();
  }

  for (int iproc = 0; iproc < nProcs; ++iproc) {
    recvcnts[iproc] = m_binL;
    sendcnts[iproc] = binProcRatio + (rem > iproc);
  }

  rdispls[0] = 0;
  sdispls[0] = 0;
  for (int iproc = 1; iproc <= nProcs; ++iproc) {
    rdispls[iproc] = rdispls[iproc-1] + recvcnts[iproc-1];
    sdispls[iproc] = sdispls[iproc-1] + sendcnts[iproc-1];
  }

  CHKERR(MPI_Alltoallv(&sendSz[0], &sendcnts[0], &sdispls[0], MPI_INT,
           &recvSz[0], &recvcnts[0], &rdispls[0], MPI_INT,
           MPI_COMM_WORLD));

  //Begin exchange [[Dij]] with other processors
  std::vector< std::vector<Dij> > recv(m_nRecvBuf);
  std::vector< MPI_Request >    recvMsgId(m_nRecvBuf);
  std::vector< MPI_Request >    sendMsgId(m_binT);
  MPI_Status status;
  int numRecvMsg = 0;
  int numSendMsg = 0;
  for (int rbin = 0; rbin < m_nRecvBuf; ++rbin) {
    int rsz = recvSz[rbin];
      
    if (rsz == 0 )
      continue;
      
    recv[rbin].resize(rsz);
    CHKERR(MPI_Irecv(&recv[rbin][0], sizeof(Dij)*rsz, MPI_BYTE,
         m_recvProcA[rbin], (rbin << 4) + 1,
         MPI_COMM_WORLD, &recvMsgId[numRecvMsg++]));
  }

  numSendMsg = 0;
  for (sbin = 0; sbin < m_binT; ++sbin) {
    int ssz = send[sbin].size();

    if (ssz == 0)
      continue;

    CHKERR(MPI_Isend(&send[sbin][0], sizeof(Dij)*ssz, MPI_BYTE,
         m_sendProcA[sbin], (m_sendTagA[sbin] << 4) + 1, MPI_COMM_WORLD,
         &sendMsgId[numSendMsg++]));
  }
  VERBOSE("-- Finished exchanging Dij" << std::endl);

  //Handle Received Data with [[MPI_Waitany]]
  for (int imsg = 0; imsg < numRecvMsg; ++imsg) {
    int index;
      
    CHKERR(MPI_Waitany(numRecvMsg, &recvMsgId[0], &index, &status));

    int         rbin  = status.MPI_TAG >> 4;
    int         lbin  = rbin % m_binL;
    PQ_Dij&       pq    = m_pqA[lbin];
    std::vector<Dij>& r     = recv[rbin];
    int         rsz   = r.size();

    for (int ja = 0; ja < rsz; ++ja) {
      i      = r[ja].i;
      int j  = r[ja].j;

      pq.push(Dij(r[ja].d, i, j));
    }
  }
  VERBOSE("-- Finished storing Dij in PQ" << std::endl);

  //[[MPI_Waitall]] for other messages
  std::vector< MPI_Status > statusA(m_binT);
  CHKERR(MPI_Waitall(numSendMsg, &sendMsgId[0], &statusA[0]));

  //Sort bins in descending order
  typedef std::multimap< double, int, std::greater<double> >         MMdi;
  typedef std::multimap< double, int, std::greater<double> >::iterator MMdi_iter;
  MMdi      mm;
  MMdi_iter it;


  for (int lbin = 0; lbin < m_binL; ++lbin) {
    int   bi    = m_numBinPairA[lbin].a;
    int   bj    = m_numBinPairA[lbin].b;
    double maxTSum = m_binRange[bi] + m_binRange[bj];
    mm.insert( std::make_pair(maxTSum, lbin) );
  }

  count = 0;
  for (it = mm.begin(); it != mm.end(); it++) {
    int lbin = (*it).second;
    m_binOrder[count++] = lbin;
  }
}

void
SpeciesBlock::bcastRow(int minI, std::vector<float>& vd, std::vector<float>& work)
{
  VERBOSE("--bcastRow" << std::endl);
  // What processor [[iproc]] has row [[minI]]?
  int iproc = findProcForMinIdx(minI);

  if (myProc == iproc) {
    //copy row into [[work]]
    int      k   = m_firstActiveNodeG;
    int      ii  = m_g2l[minI];
    int      rii = m_redirectL[ii];
    std::vector<float>& d   = m_distances[rii];
    int      ja  = 0;

    while (k < m_nextInternalNodeG) {
      int rk   = m_redirectG[k];
      work[ja++] = d[rk];
      k    = m_nextActiveNodeG[k];
    }
  }

  // Bcast row to all processors
  CHKERR(MPI_Bcast(&work[0], m_n, MPI_FLOAT, iproc, MPI_COMM_WORLD));

  //Copy [[work]] into [[vd]]
  int ja = 0;
  for (int k = m_firstActiveNodeG; k < m_nextInternalNodeG; k = m_nextActiveNodeG[k]) {
    int rk = m_redirectG[k];
    vd[rk] = work[ja++];
  }
}

void
SpeciesBlock::modifyMatrix(const QDij & minQ, int& procNumI, int& procNumJ, int riMinL, int riMinG)
{
  int ii, jj, k, idx;

  std::vector< float > m_vdI(m_n0);              // work vector for I'th row
  std::vector< float > m_vdJ(m_n0);              // work vector for J'th row
  std::vector< float > m_work(m_n0);             // work vector.
  
  VERBOSE("modifyMatrix " << std::endl);

  //Remove global [[minJ]] from active list
  int prev        = m_prevActiveNodeG[minQ.j];
  int next        = m_nextActiveNodeG[minQ.j];
  m_prevActiveNodeG[next]   = prev;
  if (prev == -1)
    m_firstActiveNodeG      = next;
  else
    m_nextActiveNodeG[prev] = next;

  //Remove local [[minJ]] from active list
  procNumJ = findProcForMinIdx(minQ.j);
  if (procNumJ == myProc) {
    m_l--;
    jj            = m_g2l[minQ.j];
    prev          = m_prevActiveNodeL[jj];
    next          = m_nextActiveNodeL[jj];
    m_prevActiveNodeL[next]   = prev;

    if (prev == -1)
      m_firstActiveNodeL      = next;
    else m_nextActiveNodeL[prev] = next;

    m_l2g[jj]         = -1;
  }

  //Remove global [[minI]] from active list
  prev          = m_prevActiveNodeG[minQ.i];
  next          = m_nextActiveNodeG[minQ.i];
  m_prevActiveNodeG[next]   = prev;

  if (prev == -1)
    m_firstActiveNodeG      = next;
  else m_nextActiveNodeG[prev] = next;

  //Remove local [[minI]] from active list
  procNumI = findProcForMinIdx(minQ.i);
  if (procNumI == myProc) {
    m_l--;
    ii            = m_g2l[minQ.i];
    prev          = m_prevActiveNodeL[ii];
    next          = m_nextActiveNodeL[ii];
    m_prevActiveNodeL[next]   = prev;

    if (prev == -1)
      m_firstActiveNodeL      = next;
    else m_nextActiveNodeL[prev] = next;

    m_l2g[ii]         = -1;
  }

  //Bcast Rows
  bcastRow(minQ.j, m_vdJ, m_work); // copy the minQ.j line to m_vdJ
  bcastRow(minQ.i, m_vdI, m_work); // copy the minQ.i line to m_vdI

  //Complete removal of [[minI]] and [[minJ]]
  if (procNumJ == myProc) {
    jj        = m_g2l[minQ.j];
    m_redirectL[jj] = -1;
  }

  if (procNumI == myProc) {
    ii        = m_g2l[minQ.i];
    m_redirectL[ii] = -1;
  }
  m_redirectG[minQ.j]  = m_redirectG[minQ.i]  = -1;
  m_g2l[minQ.j]        = m_g2l[minQ.i]        = -1;

  //Build new D row in [[m_vd]], update [[m_T]]
  double sum = 0.0;
  if (m_saveZeroD) {
    k       = m_firstActiveNodeG;
    while ( k < m_nextInternalNodeG) {
      int    rk  = m_redirectG[k];
      float Dik = m_vdI[rk];
      float Djk = m_vdJ[rk];
      float tmp = ((Dik + Djk) - minQ.d)/2;
      if (myProc == procNumI && (fabs(tmp) < m_smallD))
        m_Dzero.push_back(Pair(k, m_nextInternalNodeG));
    
      sum += tmp;
      m_vd[rk]   = tmp;
      m_T[rk] += tmp - (Dik + Djk);
      k    = m_nextActiveNodeG[k];
    }
  }
  else {
    k       = m_firstActiveNodeG;
    while ( k < m_nextInternalNodeG) {
      int    rk  = m_redirectG[k];
      float Dik = m_vdI[rk];
      float Djk = m_vdJ[rk];
      float tmp = ((Dik + Djk) - minQ.d)/2;
      sum += tmp;
      m_vd[rk]   = tmp;
      m_T[rk] += tmp - (Dik + Djk);
      k    = m_nextActiveNodeG[k];
    }
  }
  m_T[riMinG] = sum;

  //Update D column values
  idx = m_firstActiveNodeL;
  for (int jd = 0; jd < m_l; ++jd) {
    int      rkk = m_redirectL[idx];
    k          = m_l2g[idx];
    int      rk  = m_redirectG[k];
    std::vector<float>& d   = m_distances[rkk];
    d[riMinG]        = m_vd[rk];
    idx          = m_nextActiveNodeL[idx];
  }

  if (myProc == procNumI) {
    //Store new D row
    k        = m_firstActiveNodeG;
    std::vector<float>& d = m_distances[riMinL];
    while ( k < m_nextInternalNodeG) {
      int   rk = m_redirectG[k];
      d[rk]    = m_vd[rk];
      k        = m_nextActiveNodeG[k];
    }
    m_l++;
  }
}

void
SpeciesBlock::buildTree()
{
  const int QDASZ   = 512;
  const int pqRatioInit   = 500;
  const int candidateScan = 50;  // Scan candidate list for non-candidates
         // every so often.
  MPI_Op       myMergeOp = 0;
  MPI_Op       myQminOp  = 0;
  MPI_Datatype qtype   = 0;
  unsigned int inactive;
  QDij         minQ;
  int        procNumI, procNumJ, gbin, k, ri, rj, icand, n = 0;
  int        riMinL, riMinG;
  long       totalPQ = 0L;
  double       maxR;
  int        qdAg_sz = (m_allreduce) ? m_QDA : m_QDA*nProcs;
  std::vector<ActiveDij> adijA; adijA.reserve(1000);
  std::vector<double>   maxT2(m_numBins);
  std::vector<QDij>  qdA(QDASZ), qdAg(qdAg_sz);
  int stepsUntilRebuild = MAX(static_cast<int>(m_n * m_ratio), minRebuild);

  // Setup for MPI_Allreduce on [[QDij]] for merge sort
  CHKERR(MPI_Type_contiguous(sizeof(QDij), MPI_BYTE, &qtype));
  CHKERR(MPI_Type_commit(&qtype));
  CHKERR(MPI_Op_create(myMergeSort, 1, &myMergeOp));

  // Setup for MPI_Allreduce on [[QDij]] for Qmin on zero D work
  CHKERR(MPI_Op_create(myQminFun, 1, &myQminOp));

  VERBOSE("-- N: " << m_n0 << " NP: " << nProcs << " NB: " << m_numBins << " QDA: " << m_QDA << std::endl);
  VERBOSE("tree = {" << std::endl);

  m_Timer[tBuildTree].start();

  buildGblRowSums();

  //Handle duplicate taxa.
  if (m_dupRemoval) {
    VERBOSE("m_dupRemoval " << std::endl);
    while (m_nextInternalNodeG < m_nn) {
      //Find [[minQ]] among zero Ds, [[break]] when none found
      minQ.q = DBL_MAX;
      minQ.i = -1;
      unsigned int ipair = 0;
      const int deadPairsR = 25;
      if ((m_n0 - m_n) % deadPairsR == deadPairsR - 1) {
        // remove dead pairs every deadPairsR iterations
        while (ipair < m_Dzero.size()) {
          Pair p = m_Dzero[ipair];
          ri = m_redirectG[p.i];
          rj = m_redirectG[p.j];

          while (ri == -1 || rj == -1) {
            p = m_Dzero[ipair] = m_Dzero.back();
            ri = m_redirectG[p.i];
            rj = m_redirectG[p.j];
            if (ipair == m_Dzero.size() - 1) {
              m_Dzero.erase(m_Dzero.end()-1, m_Dzero.end());
              goto L1;
            }
            m_Dzero.erase(m_Dzero.end()-1, m_Dzero.end());
          }
                              
          double q = -m_T[ri] - m_T[rj];  // d is zero!
          if (q <= minQ.q) {
            VERBOSE("-- (dup(1)):  i: " << MIN(p.i,p.j) <<  " j: " << MAX(p.i,p.j) <<
                     " minQ: " << q << std::endl);

            minQ.assign(q, 0.0, p.i, p.j);
          }
          ++ipair;
        }
      }
      else {
        // Simple Search the other iterations
        for (ipair = 0; ipair < m_Dzero.size(); ++ipair) {
          Pair p = m_Dzero[ipair];
                              
          ri = m_redirectG[p.i];
          rj = m_redirectG[p.j];
          if (ri == -1 || rj == -1)
            continue;

          double q = -m_T[ri] - m_T[rj];  // d is zero!
          if (q <= minQ.q) {
            VERBOSE("-- (dup(1)):  i: " << MIN(p.i,p.j) <<  " j: " << MAX(p.i,p.j) << " minQ: " << q << std::endl);
            minQ.assign(q, 0.0, p.i, p.j);
          }
        }
      }

      L1:

      QDij tmp = minQ;
      CHKERR(MPI_Allreduce(&tmp, &minQ, 1, qtype, myQminOp, MPI_COMM_WORLD));

      VERBOSE("-- (dup(3)):  i: " << MIN(minQ.i,minQ.j) <<  " j: " << MAX(minQ.i,minQ.j) << " minQ: " << minQ.q << std::endl);

      // If there is no new minimum then break out;
      if (minQ.i == -1)
        break;

      int ii  = MIN(minQ.i,minQ.j);
      minQ.j  = MAX(minQ.i,minQ.j);
      minQ.i  = ii;
      int iMinL = m_g2l[minQ.i];
      riMinL  = ( iMinL > -1) ? m_redirectL[iMinL] : -1;
      riMinG  = m_redirectG[minQ.i];

      VERBOSE("{minI= "        << std::setw(6)  << minQ.i
        << ", minJ= "    << std::setw(6)  << minQ.j
        << ", r= "       << std::setw(6)  << m_n
        << ", q= "       << minQ.q
        << " }," << std::endl);

      if (myProc == 0) {
        ri     = m_redirectG[minQ.i];
        rj     = m_redirectG[minQ.j];
        m_nodes[m_nextInternalNodeG].join(m_nextInternalNodeG, &m_nodes[minQ.i], &m_nodes[minQ.j],
                                          minQ.d, m_T[ri], m_T[rj], m_n);
      }

      modifyMatrix(minQ, procNumI, procNumJ, riMinL, riMinG);
      if (procNumI == myProc) {
        
        m_l2g[m_nextInternalNodeL]     = m_nextInternalNodeG;
        m_g2l[m_nextInternalNodeG]     = m_nextInternalNodeL;
        m_redirectL[m_nextInternalNodeL++] = riMinL;
      }
      m_redirectG[m_nextInternalNodeG] = riMinG;

      m_nextInternalNodeG++;
   
      m_n--;
    }
  }
  m_saveZeroD = false;  // no longer save zero Ds
  VERBOSE( "-- Finished removing duplicates " << std::endl);

  buildBins();

  int  pq_ratio_ck = pqRatioInit;
  long sz0    = m_n;
  long aveSz0 = (((sz0 - 1) * sz0) / 2) / m_binT;
  while (m_nextInternalNodeG < m_nn) {

    minQ = qdA[0];
    if (!m_predictQ)
      minQ.q = DBL_MAX;

    //Find 1D Bins max row sum boundaries
    for (int ibin = 0; ibin < m_numBins; ++ibin)
      m_binRange[ibin] = maxT2[ibin] = -DBL_MAX;

    k = m_firstActiveNodeG;
    while (k < m_nextInternalNodeG) {

      int rk = m_redirectG[k];
      int bi = m_binLoc[rk];
      VERBOSE("-- T: " << m_T[rk] << " rk: " << rk << "\n");
      if (m_T[rk] > maxT2[bi]) {
        if (m_T[rk] > m_binRange[bi]) {
          maxT2[bi]  = m_binRange[bi];
          m_binRange[bi] = m_T[rk];
        }
        else maxT2[bi] = m_T[rk];
      }
      k = m_nextActiveNodeG[k];
    }

    //----------------------------------------------
    //   Search through [[Dij]] candidates
    //----------------------------------------------

    // truncate inactive candidates from end.
    int last = adijA.size() - 1;
    for (icand = last; icand >= 0 ; --icand) {
      ActiveDij& adij = adijA[icand];
      if (adij.active)
        break;
      
      ri = m_redirectG[adij.i];
      rj = m_redirectG[adij.j];
      if (ri != -1 && rj != -1)
        break;

      adij.active = false;
    }
    if (icand != last)
      adijA.erase(adijA.begin()+(icand+1), adijA.end());

    // Go over the list of candidates
    inactive = 0;
    for (icand = adijA.size()-1; icand >= 0; --icand) {
      ActiveDij& adij = adijA[icand];

      if (!adij.active) {
        inactive++;
        continue;
      }

      ri = m_redirectG[adij.i];
      rj = m_redirectG[adij.j];
      if (ri == -1 || rj == -1)
        adij.active = false;
      else {
        double q = ((double)adij.d) * (m_n - 2) - m_T[ri] - m_T[rj];
        VERBOSE("-- adijA i: " << MIN(adij.i,adij.j) << " j: " << MAX(adij.i,adij.j)
           << " d: " << adij.d << " q: " << q << " minQ: " << minQ.q << "\n");

        if (q <= minQ.q) {
          minQ.assign(q, adij.d, adij.i, adij.j);

          VERBOSE("-- minQ (c) i: " << MIN(adij.i,adij.j) << " j: " << MAX(adij.i,adij.j)
             << " d: " << adij.d << " q: " << q << "\n");

          qdA[n++].assign(q, adij.d, adij.i, adij.j);
          if (n == QDASZ) {
            nth_element(qdA.begin(), qdA.begin()+m_QDA, qdA.begin() + n, QDijCompare());
            n = m_QDA;
          }
        }
      }
    }

    // Remove candidates that will not be candidates every 50 or so iterations.
    if((m_n0 - m_n) % candidateScan == 0 && stepsUntilRebuild > candidateScan / 2) {
      for(icand = adijA.size() - 1; icand >= 0; --icand) {
        ActiveDij& adij = adijA[icand];

        if (!adij.active)
          continue;

        ri = m_redirectG[adij.i];
        rj = m_redirectG[adij.j];
        int bi = m_binLoc[ri];
        int bj = m_binLoc[rj];
        double maxTSum = m_binRange[bi] + ( (bi == bj) ? maxT2[bi] : m_binRange[bj]);
        double qLimit = ((double)adij.d)*(m_n - 2) - maxTSum;
        if (qLimit > minQ.q) {
          adij.active = false;
          int lbin    = m_idxA[bi][bj]/nProcs;

          PQ_Dij& pq  = m_pqA[lbin];
          pq.push(Dij(adij.d, adij.i, adij.j));
          VERBOSE("-- pq.push i: " << MIN(adij.i,adij.j) << " j: " << MAX(adij.i,adij.j) << "\n");
        }
      }
    }
    // Compact the candidate list when 20 % are inactive
    if(adijA.size() > 0 && inactive > adijA.size() / 5) {
      std::vector<ActiveDij>::iterator goodEnd = partition(adijA.begin(), adijA.end(), activeTest);
      adijA.erase(goodEnd, adijA.end());
      inactive = 0;
    }

    //Find [[minI]], [[minJ]] and [[minQ]] across local bins
    for (int jb = 0; jb < m_binL; ++jb) {

      int     lbin    = m_binOrder[jb];
      PQ_Dij& pq      = m_pqA[lbin];
      int     bi      = m_numBinPairA[lbin].a;
      int     bj      = m_numBinPairA[lbin].b;
      double   maxTSum = m_binRange[bi] + ( (bi == bj) ? maxT2[bi] : m_binRange[bj]);
      while (! pq.empty() ) {
        const Dij& dij = pq.top();
        ri         = m_redirectG[dij.i];
        rj         = m_redirectG[dij.j];

        if (ri == -1 || rj == -1) {
          pq.pop();
          continue;
        }

        double q       = ((double) dij.d) * (m_n - 2);
        double qLimit  = q - maxTSum;
        q      -= m_T[ri] + m_T[rj];
        VERBOSE("-- PQ i: " << MIN(dij.i,dij.j) << " j: " << MAX(dij.i,dij.j) << " lbin: " << lbin
           << " d: " << dij.d << " q: " << q << "\n");

        if (qLimit > minQ.q) {
          // No reason to search this bin anymore.
          break;
        }

        qdA[n++].assign(q, dij.d, dij.i, dij.j);
        if (n == QDASZ) {
          nth_element(qdA.begin(), qdA.begin()+m_QDA, qdA.begin() + n, QDijCompare());
          n = m_QDA;
        }

        if (q <= minQ.q) {
          minQ.assign(q, dij.d, dij.i, dij.j);
          VERBOSE("-- minQ i: " << MIN(dij.i,dij.j) << " j: " << MAX(dij.i,dij.j)
             << " d: " << dij.d << " q: " << q << "\n");
        }
        adijA.push_back(ActiveDij(dij));
        pq.pop();
      }
    }
    if (m_allreduce)
      partial_sort(qdA.begin(), qdA.begin()+MIN(n,m_QDA), qdA.begin() + n, QDijCompare());
    else
      nth_element(qdA.begin(),  qdA.begin()+MIN(n,m_QDA), qdA.begin() + n, QDijCompare());

    //Find Global [[minI]], [[minJ]]
    for (int jb = n; jb < m_QDA; ++jb)
      qdA[jb].q = DBL_MAX;

    int minI, minJ;
    if (m_allreduce) {
      CHKERR(MPI_Allreduce(&qdA[0], &qdAg[0], m_QDA, qtype, myMergeOp, MPI_COMM_WORLD));
      minQ = qdAg[0];
      minI = MIN(minQ.i, minQ.j);
      minJ = MAX(minQ.i, minQ.j);
    }
    else {
      CHKERR(MPI_Allgather(&qdA[0],  sizeof(QDij)*m_QDA, MPI_BYTE,
         &qdAg[0], sizeof(QDij)*m_QDA, MPI_BYTE, MPI_COMM_WORLD));

      minQ.q = DBL_MAX;
      minI   = INT_MAX;
      minJ   = INT_MAX;

      for (int ja = 0; ja < nProcs*m_QDA; ++ja) {
        if (qdAg[ja].q <= minQ.q) {
          int newI = MIN(qdAg[ja].i,qdAg[ja].j);
          int newJ = MAX(qdAg[ja].i,qdAg[ja].j);
          if ((qdAg[ja].q < minQ.q) || (minJ < newJ || ((minJ == newJ) && (minI < newI)))) {
            minQ = qdAg[ja];
            minI = MIN(minQ.i, minQ.j);
            minJ = MAX(minQ.i, minQ.j);
          }
        }
      }
    }

    int ii     = MIN(minQ.i,minQ.j);
    minQ.j     = MAX(minQ.i,minQ.j);
    minQ.i     = ii;
    riMinL     = -1;
    riMinG     = m_redirectG[minQ.i];
    int    iMinL   = m_g2l[minQ.i];
    if (iMinL > -1)
      riMinL = m_redirectL[iMinL];

    VERBOSE("{minI= "      << std::setw(6)  << minQ.i
      << ", minJ= "    << std::setw(6)  << minQ.j
      << ", r= "       << std::setw(6)  << m_n
      << ", q= "       << minQ.q
      << " }," << std::endl);

    if (myProc == 0) {
      ri   = m_redirectG[minQ.i];
      rj   = m_redirectG[minQ.j];
      m_nodes[m_nextInternalNodeG].join(m_nextInternalNodeG, &m_nodes[minQ.i], &m_nodes[minQ.j],
                                        minQ.d, m_T[ri], m_T[rj], m_n);
    }

    modifyMatrix(minQ, procNumI, procNumJ, riMinL, riMinG);

    //Check Memory and PQ ratio [[=>]] rebuild bins
    pq_ratio_ck--;
    if (pq_ratio_ck == 0) {
      pq_ratio_ck   = pqRatioInit;

      // Check PQ size
      if (stepsUntilRebuild > 0) {
        totalPQ  = 0;
        for (int lbin = 0; lbin < m_binL; ++lbin)
          totalPQ += m_pqA[lbin].size();

        totalPQ = totalPQ/((long)m_binL);
        double r = ((double) totalPQ)/((double)aveSz0);
        CHKERR(MPI_Allreduce(&r, &maxR,  1, MPI_DOUBLE, MPI_MAX, MPI_COMM_WORLD));
        if (maxR > m_pqRatio)
          stepsUntilRebuild = 0;
      }
    }

    m_n--;
    if (stepsUntilRebuild == 0) {
      //Rebuild bins
      if (procNumI == myProc) {
        m_l2g[m_nextInternalNodeL]     = m_nextInternalNodeG;
        m_g2l[m_nextInternalNodeG]     = m_nextInternalNodeL;
        m_redirectL[m_nextInternalNodeL++] = riMinL;
      }
      m_redirectG[m_nextInternalNodeG] = riMinG;

      m_nextInternalNodeG++;

      adijA.clear();
      buildGblRowSums();
      buildBins();
      
      if (m_n < minRebuild)
        stepsUntilRebuild = INT_MAX;
      else stepsUntilRebuild = static_cast<int>(m_n*m_ratio);

      sz0    = m_n;
      aveSz0 = (((sz0 - 1) * sz0)/2)/m_binT;
    }

    else {
      //Insert new [[d]]s in bins
      stepsUntilRebuild--;

      // Pick bin location for new cluster based on T value.
      for (int ibin = 0; ibin < m_numBins; ++ibin) {
        if (m_T[riMinG] <= m_binRange[ibin]) {
          m_binLoc[riMinG] = ibin;
          break;
        }
      }

      k        = m_firstActiveNodeG;
      int bi   = m_binLoc[riMinG];
      if (procNumI == myProc) {
        m_l2g[m_nextInternalNodeL]     = m_nextInternalNodeG;
        m_g2l[m_nextInternalNodeG]     = m_nextInternalNodeL;
        m_redirectL[m_nextInternalNodeL++] = riMinL;
      }
      m_redirectG[m_nextInternalNodeG] = riMinG;


      while (k < m_nextInternalNodeG) {
        int rk, bj, iproc;
        rk    = m_redirectG[k];
        bj    = m_binLoc[rk];
        gbin  = m_idxA[bi][bj];
        iproc = gbin % nProcs;
        if (iproc == myProc) {
          int a, b;
          int lbin = gbin / nProcs;
          if (rk < riMinG) {
            a  = k;
            b  = m_nextInternalNodeG;
          }
          else {
            a  = m_nextInternalNodeG;
            b  = k;
          }
          VERBOSE("-- PQ insert: i: " << MIN(a,b) << " j: " << MAX(a,b) << " lbin: " << lbin
             << " gbin: " << gbin
             << " d: " << m_vd[rk] << "\n");

          m_pqA[lbin].push(Dij(m_vd[rk],a,b));
        }
        k = m_nextActiveNodeG[k];
      }
      m_nextInternalNodeG++;
    }

    //Use [[qdAg]] to update [[qdA]]

    std::vector<QDij>::iterator jt;
    for (jt = qdAg.begin(); jt != qdAg.end(); ++jt) {
      QDij& qd = *jt;
      if (qd.i < 0 || qd.j < 0 || (ri = m_redirectG[qd.i]) == -1 || (rj = m_redirectG[qd.j]) == -1) {
        qd.q = DBL_MAX;
        continue;
      }

      qd.q = ((double)qd.d) * (m_n - 2) - m_T[ri] - m_T[rj];
    }
    int szQ = MIN(2 * m_QDA, static_cast<int>(qdAg.size()));
    partial_sort(qdAg.begin(), qdAg.begin()+szQ, qdAg.end(), QDijCompare());
    qdA[0] = qdAg[0];
    int jc = 0;
    for (int jb = 1; jb < qdAg_sz; ++jb) {
      if (((qdA[jc].i != qdAg[jb].i) || (qdA[jc].j != qdAg[jb].j)) && (qdAg[jb].q != DBL_MAX)) {
        qdA[++jc] = qdAg[jb];
        if (jc == m_QDA - 1)
          break;
      }
    }
    n = jc + 1;
  }
  VERBOSE("}" << std::endl);

  m_Timer[tBuildTree].end();

  if(myProc == 0) {
    std::ostringstream s;
    std::string f(m_baseFn);
    f = f + ".tree";
    std::ofstream o(f.c_str());
    
    m_nodes[m_nn-1].buildTreeOutput(s);

    o << s.str() << ";\n";
    o.close();
  }

  CHKERR(MPI_Type_free(&qtype));
  CHKERR(MPI_Op_free(&myMergeOp));
  CHKERR(MPI_Op_free(&myQminOp));
}

void
SpeciesBlock::freeMemory()
{
  m_T.clear();
  m_speciesCount.clear();
  m_speciesOffset.clear();

  m_binLoc.clear();
  m_binRange.clear();

  m_nextActiveNodeG.clear();
  m_prevActiveNodeG.clear();
  m_nextActiveNodeL.clear();
  m_prevActiveNodeL.clear();
  m_redirectG.clear();
  m_redirectL.clear();
  m_g2l.clear();
  m_l2g.clear();

  m_pqA.clear();
  m_binOrder.clear();

  m_sendProcA.clear();
  m_recvProcA.clear();
  m_sendTagA.clear();
  m_numBinPairA.clear();


  m_idxA.clear();
  m_vd.clear();
  m_Dzero.clear();
}

void
SpeciesBlock::buildAlign()
{
  MPI_Status status;
  MPI_Request recvReq;
  AlignmentMatch *align;
  char *p, *localBuf, *sendBuf, buffer[maxSeqLen + 1];
  int i, j, m0, idx, rem, size, iProc, bufSz, nSeqs, seqLen, sProcs, iSpecies, lSpecies;
  
  iNode aux;
  std::vector<int> nodes;
  std::vector< iNode > alignmentSeqs;
  
  VERBOSE("-- SpeciesBlock::buildAlign " << std::endl);

  m0 = (m_n0 - 1) / (3 * nProcs);
  rem = (m_n0 - 1) % (3 * nProcs);
  lSpecies = m0 + (myProc < rem);
  
  //Create Sequence Data Send/Receive buffers
  bufSz = (m_nn + 1) * (sizeof(int) + (maxSeqLen + 1) + 5);

  localBuf = new char[bufSz];
  sendBuf = new char[bufSz];
  align = new AlignmentMatch(10, 2, 16, maxSeqLen);
  align->prepare(seqs, myProc);

  m_Timer[tBuildAlign].start();

  sProcs = nProcs;
  while((sProcs > 0) && (myProc < sProcs)) {

    //Post Receive Buffer for Sequence Data
    DEBUGPRT(3, "-- SpeciesBlock::buildAlign MPI_Irecv " << myProc << std::endl);
    CHKERR(MPI_Irecv(&localBuf[0], bufSz, MPI_CHAR, 0, (7 + (myProc << 3)), MPI_COMM_WORLD, &recvReq));

    if(myProc == 0) {

      iSpecies = lSpecies;
      for(iProc = sProcs - 1; iProc >= 0; iProc--) {

        nodes.clear();
        m_nodes[m_nn - 1].getNodes(nodes, iSpecies);

        p = &sendBuf[sizeof(int)];
        memcpy(&sendBuf[0], (char *) &iProc, sizeof(int));
        
        size = nodes.size();

        memcpy(p, (char *) &size, sizeof(int));
        p += sizeof(int);
        for(i = 0; i < size; i++) {
          memcpy(p, (char *) &nodes[i], sizeof(int));
          p += sizeof(int);
        }
        
        VERBOSE("Sending to " << iProc << std::endl);
        CHKERR(MPI_Send(&sendBuf[0], bufSz, MPI_CHAR, iProc, (7 + (iProc << 3)), MPI_COMM_WORLD));

        iSpecies = m0 + (iProc < rem);
      }

      for(i = m_n; i < m_nn - 1; i++)
        m_nodes[i].update();

      //printf("\n\n\n");
    }
    
    //Wait for [[localBuf]] for local sequence data
    CHKERR(MPI_Wait(&recvReq, &status));

    p = &localBuf[sizeof(int)];
    memcpy((char *) &size, p, sizeof(int));
    p += sizeof(int);

    //printf("myProc: %d size: %d\t", myProc, size);
    alignmentSeqs.clear();
    for (i = 0; i < size - 3; i += 3) {
      memcpy((char *) &idx, p, sizeof(int));
      p += sizeof(int);
      
      aux.vecSize = 0;
      aux.aligned = false;
      aux.iSpecies.clear();

      aux.index = idx;

      memcpy((char *) &idx, p, sizeof(int));
      p += sizeof(int);

      aux.iSpecies.push_back(idx);
      if(idx < seqs.size()) {
        //printf("s");
        aux.vecSize++; // It's a sequence
      }
      else {
        ;
        //printf("a");
        // It's a alignment
      }

      memcpy((char *) &idx, p, sizeof(int));
      p += sizeof(int);
      aux.iSpecies.push_back(idx);
      if(idx < seqs.size()) {
        //printf("s");
        aux.vecSize++; // It's a sequence
      }
      else {
        ;
        //printf("a");
        // It's a alignment
      }

      alignmentSeqs.push_back(aux);
    }
    //printf("\n");

    DEBUGPRT(3, "computeAlignment " << myProc << std::endl);
    //sleep(nProcs - myProc);
    align->computeAlignment(alignmentSeqs, seqs, myProc);
    //printf("computeAlignment end %d\n", myProc);

    sProcs /= 2;
    if(myProc < sProcs) {
      iProc = myProc + sProcs;
      //printf("\nProc %d waiting data from %d tag %d\n", myProc, iProc, (7 + (myProc << 3)));
      CHKERR(MPI_Recv(&localBuf[0], bufSz, MPI_CHAR, iProc, (7 + (myProc << 3)), MPI_COMM_WORLD, &status));
      //printf("Proc %d received\n", myProc);
      //sleep(myProc * 1);

      p = &localBuf[sizeof(int)];

      memcpy((char *) &size, p, sizeof(int));
      p += sizeof(int);

      //printf("size: %d\n", size);
      for(i = 0; i < size; i++) {

        memcpy((char *) &idx, p, sizeof(int));
        p += sizeof(int);
        memcpy((char *) &nSeqs, p, sizeof(int));
        p += sizeof(int);

        aux.index = idx;
        aux.vecSize = nSeqs;
        aux.iSpecies.clear();

        //printf("i: %d idx: %d nSeqs: %d\n", i, idx, nSeqs);
        for(j = 0; j < nSeqs; j++) {

          memcpy((char *) &iSpecies, p, sizeof(int));
          p += sizeof(int);
          memcpy(buffer, p, seqsLen[iSpecies]);
          p +=  seqsLen[iSpecies] + 1;

          //printf("%d, ", iSpecies);
          seqs[iSpecies] = buffer;
          aux.iSpecies.push_back(iSpecies);
        }
        //printf("\n");

        alignmentSeqs.push_back(aux);
      }
    }
    else if(myProc < (sProcs * 2)) {

      p = &sendBuf[sizeof(int)];
      size = alignmentSeqs.size();

      iProc = myProc - sProcs;
      //printf("Proc %d preparing data to %d tag %d\n", myProc, iProc, (7 + (iProc << 3)));
      memcpy(&sendBuf[0], (char *) &iProc, sizeof(int));
      memcpy(p, (char *) &size, sizeof(int));
      p += sizeof(int);

      for(i = 0; i < size; i++) {

        idx = alignmentSeqs[i].index;
        nSeqs = alignmentSeqs[i].vecSize;

        memcpy(p, (char *) &idx, sizeof(int));
        p += sizeof(int);
        memcpy(p, (char *) &nSeqs, sizeof(int));
        p += sizeof(int);

        for(j = 0; j < nSeqs; j++) {

          iSpecies = alignmentSeqs[i].iSpecies[j];

          memcpy(p, (char *) &iSpecies, sizeof(int));
          p += sizeof(int);
          memcpy(p, seqs[iSpecies].c_str(), seqsLen[iSpecies]);
          p[seqsLen[iSpecies]] = '\0';
          p +=  seqsLen[iSpecies] + 1;
        }
      }

      //printf("Proc %d Sending data to %d tag %d\n", myProc, iProc, (7 + (iProc << 3)));
      CHKERR(MPI_Send(&sendBuf[0], bufSz, MPI_CHAR, iProc, (7 + (iProc << 3)), MPI_COMM_WORLD));
    }
  }

  m_Timer[tBuildAlign].end();
  if (myProc == 0) {
    align->prepare(seqs, myProc);
  }

  delete align;
  
  //Clean up memory allocation
  delete [] localBuf;
  delete [] sendBuf;
}
