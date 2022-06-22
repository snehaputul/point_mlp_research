#!/bin/sh
#SBATCH --account=soscip-3-090
#SBATCH --partition=compute
#SBATCH --nodes=1                               # number of nodes requested
#SBATCH --ntasks=1                              # this should be same as number of nodes
#SBATCH --gpus-per-node=1
#SBATCH --mail-user=sr195@queensu.ca
#SBATCH --mail-type=END,FAIL
#SBATCH --error=/scratch/a/amiilab/shuvendu/OUTPUTS/PointMLP/%A.err
#SBATCH --output=/scratch/a/amiilab/shuvendu/OUTPUTS/PointMLP/%A.out
#SBATCH --open-mode=append                      # Append is important because otherwise preemption resets the file
# SBATCH --array=0-2%1                           # auto submit 2 times
#SBATCH --job-name=main
#SBATCH --time=24:00:00

echo "PointMLP RESEARCH"

export MASTER_ADDR=$(hostname)                  # Store the master node’s IP address in the MASTER_ADDR environment variable.
echo "rank $SLURM_NODEID master: $MASTER_ADDR"
echo "rank $SLURM_NODEID Launching python script"

MASTER=`/bin/hostname -s`
MPORT=$(shuf -i 6000-9999 -n 1)

echo "Master: $MASTER"
echo "Nodelist: $SLURM_JOB_NODELIST"
echo "Port: $MPORT"

module load MistEnv/2020a cuda gcc anaconda3 cmake cudnn swig sox/14.4.2
source activate pytorch_env

COMMAND="python -W ignore main.py --model pointMLP --workers 16"
echo "Command: $COMMAND"
$COMMAND

# sbatch -p debug_full_node --mail-type NONE --time '0:30:00' --array 0 run.sh
# sbatch -p debug --gpus-per-node 1 --mail-type NONE --time '0:30:00' --array 0 run.sh