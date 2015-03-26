#include "MatchesManager.h"

MatchesManager::MatchesManager(int sequencesCount, int windowSize, int maxSequenceLength)
{
  this->windowSize = windowSize;
  this->maxSequenceLength = maxSequenceLength;
  this->windowsXYCount = sequencesCount;

  printf("windows[%d][%d] = %d  -- %d\n", windowsXYCount, (windowsXYCount * (windowsXYCount + 1) / 2), maxSequenceLength, (windowsXYCount * windowsXYCount * maxSequenceLength));
  windows = new char**[windowsXYCount];
  for (int j = 0; j < windowsXYCount; j++) {
    windows[j] = new char*[windowsXYCount];

    for (int i = 0; i < windowsXYCount; ++i)
      windows[j][i] = new char[maxSequenceLength];
  }
  printf("Alocado\n");
}

char*
MatchesManager::getWindow(int windowX, int windowY)
{
  return windows[windowX][windowY];
}

MatchesManager::~MatchesManager()
{
  for (int j = 0; j < windowsXYCount; j++) {
    for (int i = 0; i < windowsXYCount; i++)
      delete[] windows[j][i];

    delete[] windows[j];
  }
  delete[] windows;
}

char *
MatchesManager::getSequence(int x, int y)
{
  if(x < y)
    return getSequence(y, x);

  char* currentWindow = windows[x / windowSize][y / windowSize];
  return currentWindow;
}
