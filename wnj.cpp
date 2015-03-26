#include "SpeciesBlock.h"

int myProc, nProcs;
MPI_Errhandler newerr;

void eh( MPI_Comm *comm, int *err, ... )
{
  if (*err != MPI_ERR_OTHER) {
      printf( "Unexpected error code\n" );
      fflush(stdout);
  }
  return;
}

int
main(int argc, char * argv[])
{  
  // Parse Command line options
  CmdLineOptions cmd(argc, argv);
  CmdLineOptions::state_t state = cmd.state();

  if (state != CmdLineOptions::iGOOD)
    return (state == CmdLineOptions::iBAD);

  MPI_Init(&argc, &argv);

  printf("MPI_Init\n");
  MPI_Comm_size(MPI_COMM_WORLD, &nProcs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myProc);

  printf("MPI_Comm_create_errhandler\n");
  MPI_Comm_create_errhandler(eh, &newerr);
  MPI_Comm_set_errhandler(MPI_COMM_WORLD, newerr);

  SpeciesBlock sb(cmd);

  printf("readFasta %d\n", myProc);
  sb.readFasta();
  printf("buildTree %d\n", myProc);
  sb.buildTree();

  printf("freeMemory %d\n", myProc);
  sb.freeMemory();

  printf("buildAlign %d\n", myProc);
  sb.buildAlign();

  MPI_Errhandler_free(&newerr);

  printf("MPI_Finalize myProc: %d\n", myProc);

  MPI_Finalize();

  sb.reportTimers();

  return 0;
}
