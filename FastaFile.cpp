#include "FastaFile.h"

FastaFile::FastaFile(const char* file)
{
  lineNumber = 0;
  fstr.open(file, std::fstream::in);
}

FastaFile::~FastaFile()
{
  if(fstr.is_open())
    fstr.close();
}

std::string
FastaFile::getSeqName()
{
  return name;
}

std::string
FastaFile::nextSeq(int* length)
{
  std::string seq = "";
  std::string line = "";

  std::getline(fstr, line);
  lineNumber++;

  if(fstr.eof()) {
    *length = 0;
    return seq;
  }
  
  if(line[0] != '>') {
    std::cerr << "Name string does not start with \">\" at line: " << lineNumber << std::endl;
    abort();
  }

  name = line;

  while(!fstr.eof() && (fstr.peek() != (int) '>')) {

    std::getline(fstr, line);
    lineNumber++;

    if (!fstr.good())
      break;

    seq += line;
  }

  if(seq.size() == 0) {
    std::cerr << "Wrong format at line: " << lineNumber << std::endl;
    abort();
  }

  *length = seq.size();

  return seq;
}