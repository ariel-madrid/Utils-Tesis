#!/bin/bash
#SBATCH --job-name=generate_disk
#SBATCH --output=generate_disk.out
#SBATCH --error=generate_disk.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-39999

DATA_DIR="editar ruta"
cd $DATA_DIR

FILES=($(ls *.dat))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

FULL_PATH="$DATA_DIR/$FILE"

python3 /path/to/your/generate_fits_files.py --dat_file "$FULL_PATH"