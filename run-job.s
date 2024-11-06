######################
#This is just a notes documents that save the commands to run the job in the cluster
#######################


#SBATCH --job-name=GP
#SBATCH --N 2 -n 2
#SBATCH --output=GP.out
#SBATCH --error=GP.err

#SBATCH --output=/sciclone/pscr/yacahuanamedra/GPKcomb/GPKcomb_%a.out
#SBATCH --error=/sciclone/pscr/yacahuanamedra/GPKcomb/GPKcomb_%a.err
#python3 rbf.py --i $SLURM_ARRAY_TASK_ID --Nsamples 50000
#python3 jacobi.py --i $SLURM_ARRAY_TASK_ID --Nsamples 60000 --L 200 --eps 0.0005

#SBATCH --t 00:10:00

cd /pscr
module load anaconda3/2023.09
conda activate gptorch
python3 rbf.py 