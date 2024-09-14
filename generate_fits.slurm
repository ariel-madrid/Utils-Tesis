#!/bin/bash
#SBATCH --job-name=generate_disk
#SBATCH --output=process_files.out
#SBATCH --error=process_files.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --array=0-39999  # Ajusta el rango según el número total de archivos

# Cambiar al directorio de datos
DATA_DIR="editar ruta"
cd $DATA_DIR

# Listar todos los archivos .dat en un array
FILES=($(ls *.dat))
FILE=${FILES[$SLURM_ARRAY_TASK_ID]}

FULL_PATH="$DATA_DIR/$FILE"

# Ejecutar el script Python con el archivo actual
python3 /path/to/your/process_sky_model.py "$FULL_PATH"