import os
import argparse
import numpy as np
import concurrent.futures
import shutil
import pandas as pd

df = pd.read_csv("./sky-param.csv")

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer pre process', add_help=False)
    parser.add_argument('--dats-path', type=str, required=True, metavar="FILE", help='path to dats files', )

    args, unparsed = parser.parse_known_args()

    return args

def getParameters(index):
    index = int(index)
    df_filtrado = df.iloc[index]
    
    return df_filtrado["mDisk"], df_filtrado["Rc"], df_filtrado["gamma"], df_filtrado["psi"], df_filtrado["H100"]

def skyModel(dat_file):
    npix = 512
    lamb1 = 740
    lamb2 = 2067

    posang = np.random.uniform(0, 180)
    incl = np.random.uniform(0, 80)
    sizeau = 420

    data_path = "/media/yogui/Nuevo vol/dats/dat-files"
    root_dir = os.getcwd()
    father_dir = os.path.dirname(root_dir)

    try:
        #Leer parametros desede el csv.
        indexDat = os.path.splitext(os.path.basename(dat_file))[0].split("-")[0]
        mdisk, rc, gamma, psi, H100 = getParameters(indexDat)
        
        #Llamar a writeDensties.py

        new_dir = os.path.join(root_dir, os.path.splitext(os.path.basename(dat_file))[0])
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        
        #Rename file .dat and move to new directory
        old_path = os.path.join(data_path, os.path.basename(dat_file))
        new_path = os.path.join(new_dir, os.path.basename(dat_file))
        shutil.copy(old_path, new_path) 

        #copy the .inp to new directory
        inp_path = os.path.join(father_dir, "radmc3d.inp")
        inp_new_path = os.path.join(new_dir, "radmc3d.inp")
        shutil.copy(inp_path, inp_new_path)

        wave_inp_path = os.path.join(father_dir, "wavelength_micron.inp")
        wave_inp_new_path = os.path.join(new_dir, "wavelength_micron.inp")
        shutil.copy(wave_inp_path, wave_inp_new_path)

        grid_inp_path = os.path.join(father_dir, "amr_grid.inp")
        grid_inp_new_path = os.path.join(new_dir, "amr_grid.inp")
        shutil.copy(grid_inp_path, grid_inp_new_path)

        opac_inp_path = os.path.join(father_dir, "dustopac.inp")
        opac_inp_new_path = os.path.join(new_dir, "dustopac.inp")
        shutil.copy(opac_inp_path, opac_inp_new_path)

        silicate_inp_path = os.path.join(father_dir, "dustkappa_silicate.inp")
        silicate_inp_new_path = os.path.join(new_dir, "dustkappa_silicate.inp")
        shutil.copy(silicate_inp_path, silicate_inp_new_path)

        mix_inp_path = os.path.join(father_dir, "dustkappa_mix_2species.inp")
        mix_inp_new_path = os.path.join(new_dir, "dustkappa_mix_2species.inp")
        shutil.copy(mix_inp_path, mix_inp_new_path)

        densities_inp_path = os.path.join(father_dir, "writeDensities.py")
        densities_inp_new_path = os.path.join(new_dir, "writeDensities.py")
        shutil.copy(densities_inp_path, densities_inp_new_path)

        os.chdir(new_dir)

        radmc3d_image_dat_name = "dust_temperature.dat"

        os.rename(dat_file, radmc3d_image_dat_name)
        
        #Ejecutar radmc3d image
        
        #Construir densities.inp
        os.system(f"python3 writeDensities.py --mdisk {mdisk} --rc {rc} --gamma {gamma} --psi {psi} --H100 {H100}")

        os.system(f"radmc3d image npix {npix} lambda {lamb1} incl {incl} posang {posang} sizeau {sizeau} nostar > output.txt")

        #Ejecutar simobserve 

        #Ejecutar tclean


    except Exception as e:
        return None 
    finally:
        os.chdir(root_dir)

    return 0
def procesar_archivos_por_lotes(archivos, batch_size, max_workers):
    total_procesados = 0
    for i in range(0, 100, batch_size):
        batch = archivos[i:i+batch_size]
        with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
            resultados = list(executor.map(skyModel, batch))
        
        procesados_en_este_lote = sum(1 for r in resultados if r is not None)  # Contar solo si el procesamiento fue exitoso
        total_procesados += procesados_en_este_lote

    print(f"Total de archivos procesados: {total_procesados}")
    
def main(args):
    #Paso 1 - Leer .dat files
    data_path = args.dats_path
    dat_files = [f for f in os.listdir(data_path)]
    
    #Paso 2 - 
    batch_size = 30  # Tamaño del lote
    max_workers = 8    # Ajusta según el número de núcleos de tu CPU
    # Procesar los archivos por lotes

    root = os.getcwd()
    files = "data"
    if not os.path.exists(files):
        os.mkdir(files)
    os.chdir(files)
    procesar_archivos_por_lotes(dat_files, batch_size, max_workers)
    os.chdir(root)

if __name__ == '__main__':
    args= parse_option()
    main(args)
