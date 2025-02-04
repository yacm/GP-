#!/bin/bash --login

if [ "$#" -ne 5 ]; then
    echo "Usage: $0 <KERNEL_NAME>"
    exit 1
fi

MODEL=$1
KERNEL_NAME=$2
ITD=$3
TIMES=$4
ITERATIONS=$5
# Conditional check for ITD
if [ "$ITD" = "Im" ]; then
    data=12
else
    data=13
fi

CPU=$((TIMES * data - 1))
ITER=$((ITERATIONS / TIMES))
#NEWSLURMID=$SLURM

# Create the folder if it doesn't exist
mkdir -p "${MODEL}_${KERNEL_NAME}"

# Create the SLURM script dynamically
SLURM_SCRIPT="${MODEL}_${KERNEL_NAME}/job_script.slurm"

cat << EOF > $SLURM_SCRIPT
#!/bin/bash --login

#SBATCH --account=f4thy
#SBATCH --partition=production
#SBATCH --job-name=${MODEL}_${KERNEL_NAME}${ITD}
#SBATCH --output=${MODEL}_${KERNEL_NAME}/GP${ITD}_%a.log
#SBATCH --error=${MODEL}_${KERNEL_NAME}/GP${ITD}_%a.log
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --array=[0-${CPU}]

source activate /w/lqcdpheno-sciwork24/yamil/GP
conda activate /w/lqcdpheno-sciwork24/yamil/GP

NEWSLURMID=\$((SLURM_ARRAY_TASK_ID % ${data}))

python3 combinedKernel.py --i \$NEWSLURMID --Nsamples ${ITER}  --L 100 --eps 0.001 --IDslurm \$SLURM_ARRAY_TASK_ID
EOF


# Submit the job
JOB_ID=$(sbatch $SLURM_SCRIPT | awk '{print $4}')

# Check if the submission was successful
if [ $? -eq 0 ]; then
    echo "Job $JOB_ID submitted successfully."
    # Remove the SLURM script after submission
    rm -f $SLURM_SCRIPT
    echo "SLURM script $SLURM_SCRIPT deleted."
else
    echo "Failed to submit job. SLURM script not deleted."
fi
