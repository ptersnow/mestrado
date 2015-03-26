#ifndef MATCHESMANAGER_H
#define MATCHESMANAGER_H

#include <cstdio>
#include <vector>

class MatchesManager
{
public:
  MatchesManager(int sequencesCount, int windowSize, int maxSequenceLength);
  ~MatchesManager();
  
  char *getSequence(int x, int y);
  char *getWindow(int windowX, int windowY);

private:
  int windowSize;
  int windowsXYCount;
  int maxSequenceLength;

  char*** windows;
};

#endif
