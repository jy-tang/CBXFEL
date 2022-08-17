#!/bin/bash

#SBATCH --partition=shared
#
#SBATCH --job-name=genesis
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.err
#
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=1g
#SBATCH --exclude=rome0273,rome0272,rome0271
#
#SBATCH --time=00:30:00

if [[ $1 == "" ]] 
 then
 echo "You didn't specify a Genesis Input File.  Exiting..."
 exit 0
fi

#Example usage: sbatch RunGenesis.sh inputfile.in
#export SLURM_EXACT=1
#module load openmpi
#module load devtoolset/9
#mpirun /gpfs/slac/staas/fs1/g/g.beamphysics/jytang/software/genesis/genesis_mpi_bmod $1
#/sdf/sw/gcc-4.8.5/openmpi-4.0.4/bin/mpirun /sdf/group/cbxfel/rmargraf/genesis_mpi_bmod $1
#module load mpi/openmpi-3.1.2
mpirun ~/bin/genesis_BP_FFP_centos7 $1
#mpirun ~/bin/genesis_mpi_bmod $1
