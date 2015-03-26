#ifndef __FASTAFILE_H
#define __FASTAFILE_H

#include <string>
#include <cstdlib>
#include <fstream>
#include <iostream>

class FastaFile
{
public:
 FastaFile(const char* file);

 ~FastaFile();

 std::string getSeqName();
 std::string nextSeq(int *length);

private:
  int lineNumber;
  std::string name;
  std::fstream fstr;
}; 

#endif