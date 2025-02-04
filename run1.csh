#!/bin/sh

#SBATCH --job-name=set_up
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --output=/sciclone/pscr/yacahuanamedra/ToDo/output.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/ToDo/output.log


if [ "$#" -ne 6 ]; then
    echo "Usage: $0 <KERNEL_NAME>"
    exit 1
fi

MODEL=$1
KERNEL_NAME=$2
ITD=$3
TIMES=$4
ITERATIONS=$5
II=$6
# Conditional check for ITD
if [ "$ITD" = "Im" ]; then
    data=12
else
    data=15
fi

CPU=$((TIMES - 1)) #How many CPUs are you going to use
ITER=$((ITERATIONS / TIMES)) #How many iterations per cpu are you going to produce
#NEWSLURMID=$SLURM
AA=$((II + 1))
# Create the folder if it doesn't exist
mkdir -p "${MODEL}_${KERNEL_NAME}"

# Create the SLURM script dynamically
SLURM_SCRIPT="${MODEL}_${KERNEL_NAME}/job_script.slurm"

cat << EOF > $SLURM_SCRIPT
#!/bin/tcsh

#SBATCH --job-name=${MODEL}_${KERNEL_NAME}${ITD}
#SBATCH --output=/sciclone/pscr/yacahuanamedra/${MODEL}_${KERNEL_NAME}/GP${ITD}(z=${AA}a)_%a.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/${MODEL}_${KERNEL_NAME}/GP${ITD}(z=${AA}a)_%a.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-${CPU}

module load anaconda3/2023.09 
conda activate gptorch

python3 run.py --i ${II} --Nsamples ${ITER}  --L 300 --eps 0.00033 --ITD ${ITD} --mean ${MODEL} --ker ${KERNEL_NAME} --mode "all"  --IDslurm \$SLURM_ARRAY_TASK_ID
EOF

# Submit the job
JOB_ID=$(sbatch $SLURM_SCRIPT | awk '{print $4}')

# Check if the submission was successful
if [ $? -eq 0 ]; then
    echo "Job $JOB_ID submitted successfully."
    echo " You are runing $MODEL + $KERNEL_NAME with $ITD(M) in $TIMES chains of $ITER with z=${AA}a"
    # Remove the SLURM script after submission
    rm -f $SLURM_SCRIPT
    echo "SLURM script $SLURM_SCRIPT deleted."
else
    echo "Failed to submit job. SLURM script not deleted."
fi