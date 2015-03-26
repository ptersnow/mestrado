#include "CmdLineOptions.h"

void
printUsage(const char * execName)
{
  std::cerr << "\nUsage:\n  "
      << execName  << " [options] input_file\n\n"
      << "Options:\n"
      << "  -h, -?   : Print Usage\n"
      << "  -i style : Input style:  'a' for amino sequence data in Fasta format.\n"
      << "			     'd' for dna sequence data in Fasta.\n"
      << "			     'j' for dna sequence data in Fasta (using\n"
      << "				 Jukes-Cantor conversion).\n"
      << "			     'k' for dna sequence data in Fasta (using\n"
      << "				 Kimura2 conversion).\n"
      << "			     'm' for distance matrix.\n"
      << "  -o style : Output style: 't' tree output (default).\n"
      << "			     'm' for distance matrix.\n"
      << "			     'n' output a sequence with no duplicates.\n"
      << "\n"
      << "Expert Options:\n"
      << "  -k	     : keep duplicates, do not remove early\n"
      << "  -R	     : Use MPI_Allreduce instead of MPI_Allgather\n"
      << "  -d	n    : Debug Level (1, 2, 3)\n"
      << "  -s	n    : Cluster Size  (default = 30)\n"
      << "  -r	x    : Rebuilt Ratio (default = 0.5)\n"
      << "  -q	n    : QDA Size (default = 8) (zero turns off q-prediction)\n"
      << std::endl;
}

CmdLineOptions::CmdLineOptions(int argc, char* argv[])
  : m_state(iBAD)
{
  int  opt;
  bool version, help;

  version      = false;
  help	       = false;
  pqRatio      = 2.0;
  rebuildRatio = 0.5;
  clustSize    = 30;
  qda	       = 8;
  inputFormat  = "a";
  outputFormat = "t";
  debugFlg     = 0;
  predictQ     = true;
  allreduce    = false;
  dupRemoval   = true;

  while ( (opt = getopt(argc, argv, "Ri:kh?d:s:r:m:no:q:")) != -1) {
    switch (opt) {
    case 'R':
      allreduce = true;
      break;
    case 'i':
      inputFormat = (optarg);
      break;
    case 'o':
      outputFormat = cleanup(optarg);
      break;
    case 'k':
      dupRemoval = false;
      break;
    case '?':
    case 'h':
      help = true;
      break;
    case 's':
      clustSize = strtol(optarg, (char **) NULL, 10);
      break;
    case 'r':
      rebuildRatio = (float) strtod(optarg, (char **) NULL);
      break;
    case 'm':
      pqRatio = (float) strtod(optarg, (char **) NULL);
      break;
    case 'd':
      debugFlg = strtol(optarg, (char **) NULL, 10);
      break;
    case 'q':
      qda = strtol(optarg, (char **) NULL, 10);
      break;
    default:
      ;
    }
  }

  if (inputFormat != "a" && inputFormat != "d" && inputFormat != "j" && inputFormat != "k" && inputFormat != "m") {
    std::cerr << "Bad input format: " << inputFormat << std::endl;
    printUsage(argv[0]);
    return;
  }

  if (outputFormat != "n" && outputFormat != "t" && outputFormat != "m") {
    std::cerr << "Bad output format: " << outputFormat << std::endl;
    printUsage(argv[0]);
    return;
  }

  if (help || optind >= argc) {
    m_state = iHELP;
    if (!help && optind >= argc)
      std::cerr << "\nNo input file => Quitting\n\n";
    printUsage(argv[0]);
    return;
  }

  if (qda == 0) {
    predictQ = false;
    qda	     = 2;
  }

  inputFile    = argv[optind];

  // Form basename: remove directory and extension.
  std::string::size_type k;
  std::string::size_type end = inputFile.size();
  std::string::size_type i   = ( (k = inputFile.find_last_of("/")) != std::string::npos) ? k : 0;
  std::string::size_type j   = ( (k = inputFile.find_last_of(".")) != std::string::npos) ? k : end;

  if (inputFile.substr(i,1) == "/" && i+1 < end)
    ++i;

  baseName = inputFile.substr(i,j-i);

  std::fstream f(inputFile.c_str(), std::fstream::in);
  if (! f.good()) {
    std::cerr << "\nUnable to open file: " << inputFile << "\n" << std::endl;
    printUsage(argv[0]);
    return;
  }
  f.close();

  m_state = iGOOD;
}

char
CmdLineOptions::cleanup(const char * s)
{
  int c = tolower(s[0]);
  return (char) c;
}
