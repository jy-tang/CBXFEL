#!/bin/bash

#SBATCH --partition=shared
#
#SBATCH --job-name=rafel
#SBATCH --output=output-%j.txt
#SBATCH --error=output-%j.err
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=8g
#SBATCH --exclude=tur[000-026]
##SBATCH --exclude=ampt[000-020]
#SBATCH --time=36:00:00


export PYTHONPATH="/sdf/home/j/jytang/beamphysics/genesis/CBXFEL:$PYTHONPATH"
export PYTHONPATH="/sdf/home/j/jytang/beamphysics/genesis/CBXFEL/genesis_interface:$PYTHONPATH"
export PYTHONPATH="/sdf/home/j/jytang/beamphysics/genesis/CBXFEL/cavity_codes:$PYTHONPATH"
export PYTHONPATH="/sdf/home/j/jytang/beamphysics/genesis/CBXFEL/genesis_interface/genesis:$PYTHONPATH"


python -u run_recirculation_long.py
