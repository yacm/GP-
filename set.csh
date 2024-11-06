#!/bin/tcsh

#SBATCH --job-name=GPKc
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --output=/sciclone/pscr/yacahuanamedra/GPKcomb/GPKcomb_%a.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/GPKcomb/GPKcomb_%a.log
#SBATCH --time=72:00:00

#cd /pscr
module load anaconda3/2023.09 
conda activate gptorch

python3 rbf+logrbf.py --i $SLURM_ARRAY_TASK_ID --Nsamples 100000  --L 200 --eps 0.0005
