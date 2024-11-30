#!/bin/tcsh

#SBATCH --job-name=GPImcom
#SBATCH --nodes=1 --ntasks-per-node=1
#SBATCH --output=/sciclone/pscr/yacahuanamedra/Kcomb/GPIm_%a.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/Kcomb/GPIm_%a.log
#SBATCH --time=72:00:00

#cd /pscr
module load anaconda3/2023.09 
conda activate gptorch

python3 combinedKernel.py --i $SLURM_ARRAY_TASK_ID --Nsamples 100000  --L 150 --eps 0.000666
