#ifndef CMDLINEOPTIONS_H
#define CMDLINEOPTIONS_H

#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <unistd.h>

class CmdLineOptions
{
 public:
  CmdLineOptions(int argc, char* argv[]);
  enum state_t { iBAD = 1, iHELP, iGOOD };


  state_t state() {return m_state;}

 private:
  char cleanup(const char* s);

 public:
  bool	       predictQ;
  bool	       allreduce;
  bool	       dupRemoval;
  int	         debugFlg;
  int	         clustSize;
  int	         qda;
  float	       rebuildRatio;
  float	       pqRatio;
  std::string  inputFormat;
  std::string  outputFormat;
  std::string  inputFile;
  std::string  baseName;

 private:
  state_t      m_state;
};

#endif /* CMDLINEOPTIONS_H */
