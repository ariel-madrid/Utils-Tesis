#!/bin/bash
#SBATCH --job-name=generate_disk
#SBATCH --output=generate_disk.out
#SBATCH --error=generate_disk.err
#SBATCH --ntasks=128
#SBATCH --cpus-per-task=1
#SBATCH --array=0-1000

DATA_DIR="/home/XI/aargomedo/Data-Gen/datos/dat-files"
cd $DATA_DIR

FILES=($(ls *.dat))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

FULL_PATH="$DATA_DIR/$FILE"

python3 /home/XI/aargomedo/Data-Gen/Utils-Tesis/main.py --dat_file "$FULL_PATH"
