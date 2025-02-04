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
mode=$6

# Conditional check for ITD
if [ "$ITD" = "Im" ]; then
    data=15
else
    data=15
fi


CPU=$((TIMES * data - 1))
ITER=$((ITERATIONS / TIMES))
#NEWSLURMID=$SLURM

# Create the folder if it doesn't exist
mkdir -p "${MODEL}_${KERNEL_NAME}"

# Create the SLURM script dynamicallyy
SLURM_SCRIPT="${MODEL}_${KERNEL_NAME}/job_script.slurm"

cat << EOF > $SLURM_SCRIPT
#!/bin/tcsh

#SBATCH --job-name=${MODEL}_${KERNEL_NAME}(${mode})${ITD}
#SBATCH --output=/sciclone/pscr/yacahuanamedra/${MODEL}_${KERNEL_NAME}(${mode})/GP${ITD}_%a.log
#SBATCH --error=/sciclone/pscr/yacahuanamedra/${MODEL}_${KERNEL_NAME}(${mode})/GP${ITD}_%a.log
#SBATCH --time=72:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=0-${CPU}
#SBATCH --mem=3000M

module load miniforge3/24.9.2-0
conda activate gptorch

@ NEWSLURMID = \$SLURM_ARRAY_TASK_ID % ${data}

python3 run.py --i \$NEWSLURMID --Nsamples ${ITER}  --L 100 --eps 0.001 --ITD ${ITD} --mean ${MODEL} --ker ${KERNEL_NAME} --mode ${mode}  --IDslurm \$SLURM_ARRAY_TASK_ID
EOF

# Submit the job
JOB_ID=$(sbatch $SLURM_SCRIPT | awk '{print $4}')

# Check if the submission was successful
if [ $? -eq 0 ]; then
    echo "Job $JOB_ID submitted successfully."
    echo " You are runing $MODEL + $KERNEL_NAME ($mode) with $ITD(M) in $TIMES chains of $ITER"
    # Remove the SLURM script after submission
    rm -f $SLURM_SCRIPT
    echo "SLURM script $SLURM_SCRIPT deleted."
else
    echo "Failed to submit job. SLURM script not deleted."
fi
