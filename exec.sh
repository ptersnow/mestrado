#!/bin/bash

#[[ $# == 1 && $1 != *[^0-9]* ]] || { echo "Invalid input." >&2; exit 1; }

make clean
make

if [[ $? -eq 0 ]]; then
  #for k in {1 .. $1}
  #do
    for i in 1 2 4 8
    do
      for j in {1..6}
      do
        echo "Executing clustal case0$j with $i procs."
        /opt/mpich2/gnu/bin/mpiexec -np $i -f machines ./clustal case0$j.fasta 2>> Case${j}proc$i
        #/opt/mpich2/mx/bin/mpiexec -np $i -f machines ./clustal case0$j.fasta 2>> Case${j}proc$i
      done
    done
  #done
else echo "Make failed status $?."
fi