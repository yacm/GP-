#!/bin/tcsh

#SBATCH --job-name=test #1change
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --output=/sciclone/pscr/yacahuanamedra/test/GPIm_%a.log #2changes
#SBATCH --error=/sciclone/pscr/yacahuanamedra/test/GPIm_%a.log #2changes
#SBATCH --time=72:00:00

#cd /pscr
module load anaconda3/2023.09 
conda activate gptorch

python3 combinedKernel.py --i $SLURM_ARRAY_TASK_ID --Nsamples 100000  --L 100 --eps 0.001 --ITD "Im" #change

