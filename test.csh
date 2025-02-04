#!/bin/tcsh
#SBATCH --job-name=memory_test
#SBATCH --output=output.log
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=1G
#SBATCH --time=00:10:00

echo "Memory allocated per node: $SLURM_MEM_PER_NODE MB"
echo "CPUs allocated per node: $SLURM_JOB_CPUS_PER_NODE"
echo "Memory allocated per CPU: $SLURM_MEM_PER_CPU MB"
echo "CPUs allocated per node: $SLURM_JOB_CPUS_PER_NODE"