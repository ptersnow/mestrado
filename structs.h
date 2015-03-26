#ifndef STRUCTS_H
#define STRUCTS_H

#define MIN(a, b) ((a < b) ? a : b)
#define MAX(a, b) ((a > b) ? a : b)

struct Pair
{
  Pair(int I=-1, int J=-1) : i(I), j(J) {}
  Pair(const Pair & p) : i(p.i), j(p.j) {}

  Pair & operator=(const Pair & p)
  {
    if (this != &p) {
      i = p.i;
      j = p.j;
    }
    return *this;
  }

  int i,j;
};

struct NumBinPair
{
  NumBinPair() : a(-1), b(-1) {}

  void assign(int i, int j)
  {
    a = i;
    b = j;
  }

  int a, b;
};

struct Dij
{
  Dij(float dIn=0, int I=-1, int J=-1) : d(dIn), i(I), j(J) {}

  Dij(const Dij & dij) : d(dij.d), i(dij.i), j(dij.j) {}

  Dij& operator=(const Dij & dij)
  {
    if (this != &dij) {
      d = dij.d;
      i = dij.i;
      j = dij.j;
    }
    return *this;
  }
  
  float d;
  int i,j;
};

struct QDij
{
  QDij(double qIn=DBL_MAX, float dIn=FLT_MAX, int I=-1, int J=-1) : d(dIn), i(I), j(J), q(qIn) {}
  
  QDij(const QDij & qdij) : d(qdij.d), i(qdij.i), j(qdij.j), q(qdij.q) {}

  QDij & operator=(const QDij & qdij)
  {
    if (this != &qdij) {
      q       = qdij.q;
      d       = qdij.d;
      i       = qdij.i;
      j       = qdij.j;
    }
    return *this;
  }

  void assign(double qIn, float dIn, int iIn, int jIn)
  {
    q = qIn;
    d = dIn;
    i = iIn;
    j = jIn;
  }

  float d;
  int i,j;
  double q;
};

struct ActiveDij
{
  ActiveDij(float dIn=0, int I=-1, int J=-1, bool Active = false) : d(dIn), i(I), j(J), active(Active) {}

  ActiveDij(const Dij& dij, bool Active = true) : d(dij.d), i(dij.i), j(dij.j), active(Active) {}

  ActiveDij(const ActiveDij & dij) : d(dij.d), i(dij.i), j(dij.j), active(dij.active) {}
  
  ActiveDij& operator=(const ActiveDij & dij)
  {
    if (this != &dij) {
      d      = dij.d;
      i      = dij.i;
      j      = dij.j;
      active = dij.active;
    }
    return *this;
  }

  float d;
  int i, j;
  bool active;
};

class DijCompare
{
 public:
  bool operator()(const Dij& lhs, const Dij& rhs) const
  {
    return lhs.d > rhs.d;
  }
};

class QDijCompare
{
 public:
  bool operator()(const QDij& lhs, const QDij& rhs) const
  {
    return lhs.q < rhs.q;
  }

  bool operator()(const QDij& lhs, double q) const
  {
    return lhs.q < q;
  }

  bool operator()(double q, const QDij& rhs) const
  {
    return q < rhs.q;
  }
};

class ADijCompare
{
 public:
  bool operator()(const ActiveDij& lhs, const ActiveDij& rhs) const
  {
    return lhs.d > rhs.d;
  }
};

#endif