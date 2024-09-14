import os
import argparse
import numpy as np
import concurrent.futures
import shutil
import pandas as pd
from casatasks import simobserve
from casatasks import split, concat 
from casatasks import tclean
from radmc3dPy import image
from casatools import image

df = pd.read_csv("./sky-param.csv")

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer pre process', add_help=False)
    parser.add_argument('dat_file', type=str, help='path to dat file', )

    args, unparsed = parser.parse_known_args()

    return args

def getParameters(index):
    index = int(index)
    df_filtrado = df.iloc[index]
    
    return df_filtrado["mDisk"], df_filtrado["Rc"], df_filtrado["gamma"], df_filtrado["psi"], df_filtrado["H100"], df_filtrado["mstar"], df_filtrado["rstar"], df_filtrado["tstar"], df_filtrado["incl"], df_filtrado["posang"]

def move_files(data_path, dat_file, new_dir, father_dir):
    old_path = os.path.join(data_path, os.path.basename(dat_file))
    new_path = os.path.join(new_dir, os.path.basename(dat_file))
    shutil.copy(old_path, new_path) 

    os.chdir(new_dir)

def run_simobserve(name, incenter,chanwidth,antenaCfg, degree, minutes, seconds):
    #Utilizar un archivo de configuracion de antenas para cada banda bgr      
    simobserve(
    project    = name,
    skymodel   = 'image.fits',
    indirection = f'J2000 16h26m10.32 -{degree}d{minutes}m{seconds}',
    incenter   = f'{incenter}GHz',
    inwidth    = f'{chanwidth}GHz',
    obsmode    = 'int',
    hourangle  = '0h14m',
    totaltime  = '24s',
    integration = '2s',
    antennalist = antenaCfg,
    thermalnoise = 'tsys-atm',
    user_pwv   = 1.5,
    mapsize    = ["''","''"],
    graphics   = 'none',
    verbose    = False,
    overwrite  = True
    )

def run_radmc(new_dir, band_dir, father_dir, indexDat, mstar, tstar, rstar, gamma,rc, H100, mdisk, psi, lamb2, incl, posang):
    old_path = os.path.join(new_dir, "dust_temperature.dat")
    new_path = os.path.join(band_dir, "dust_temperature.dat")
    shutil.copy(old_path, new_path) 
    
    silicate_path = os.path.join(father_dir, "dustkappa_silicate.inp")
    new_silicate_path_path = os.path.join(band_dir, "dustkappa_silicate.inp")
    shutil.copy(silicate_path, new_silicate_path_path) 

    mix_silicate_path = os.path.join(father_dir, "dustkappa_mix_2species.inp")
    new_mix_silicate_path_path = os.path.join(band_dir, "dustkappa_mix_2species.inp")
    shutil.copy(mix_silicate_path, new_mix_silicate_path_path) 

    try:
        result = subprocess.run(
            ['python3', f'{father_dir}/setup.py', '--mode', '1', '--nthread', '4', '--name', str(indexDat),
            '--rstar', f'{rstar:.5f}', '--mstar', f'{mstar:.5f}', '--tstar', str(int(tstar)),
            '--mDisk', f'{mdisk:.5f}', '--gamma', f'{gamma:.5f}', '--Rc', f'{rc:.5f}', '--H100', f'{H100:.5f}',
            '--psi', f'{psi:.5f}', '--lamb', f'{lamb2:.5f}', '--incl', str(int(incl)), '--posang', str(int(posang)),
            '--nphot', '4000000', '--npix', '512'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        if result.returncode != 0:
            print(f'Error ejecutando el comando: {result.stderr}')
        else:
            print(f'Comando ejecutado correctamente: {result.stdout}')
    except Exception as e:
        print(f'Error al ejecutar el comando: {e}')

def delete_trash():
    os.system("rm -r *.ms")
    os.system("rm -r image.fits")
    os.system("rm *.inp")
    os.system("rm -r *.image")
    os.system("rm -r *.model")
    os.system("rm -r *.pb")
    os.system("rm -r *.psf")
    os.system("rm -r *.residual")
    os.system("rm -r *.sumwt")
    os.system("rm -r *.out")
    os.system("rm -r *.txt")
    os.system("rm -r *.dat")

import subprocess
import random
def skyModel(dat_file):
    lamb1 = 740.228
    lamb2 = 2067.534

    degree = random.randint(-30, -20)
    minutes = random.randint(0, 60)
    seconds = round(random.uniform(0, 60),2)

    #print(f"Archivo {dat_file} - Degree {degree} - Minutes {minutes} - Seconds {seconds}")

    root_dir = os.getcwd()
    
    father_dir = os.path.dirname(root_dir)

    data_path = os.path.dirname(dat_file)

    try:
        #Leer parametros desede el csv.
        file_name = os.path.basename(dat_file)
        file_name_without_extension = file_name.split(".")[0]
        indexDat = file_name.split('-')[0]

        mdisk, rc, gamma, psi, H100, mstar, rstar, tstar, incl, posang = getParameters(indexDat)
        #Llamar a writeDensties.py

        new_dir = os.path.join(root_dir, file_name_without_extension)
        if not os.path.exists(new_dir):
            os.mkdir(new_dir)
        
        move_files(data_path, dat_file, new_dir, father_dir)

        radmc3d_image_dat_name = "dust_temperature.dat"

        os.rename(file_name, radmc3d_image_dat_name)

        actual_dir = os.getcwd()
        band4_dir = os.path.join(actual_dir, "band4")
        if not os.path.exists(band4_dir):
            os.mkdir(band4_dir)

        os.chdir(band4_dir)

        run_radmc(new_dir, band4_dir,father_dir, indexDat, mstar, tstar, rstar, gamma,rc, H100, mdisk, psi, lamb2, incl, posang)
        #Correr simobserve para 4 frecuencias y despues concatenar
        #138000 Mhz 138 Ghz
        #140000 Mhz 140 Ghz
        #150000 Mhz 150 Ghz
        #152000 Mhz 152 Ghz
        antena4 = os.path.join(father_dir, "antennaB4.cfg")
        antena8 = os.path.join(father_dir, "antennaB8.cfg")
        
        incenter_per_spw = [138,140,150,152]
        tolerance = '15000MHz'
        
        spw_b4 = []
        #Se corre para la banda 4
        for incenter in incenter_per_spw:
            name = str(incenter)+"-B4"
            run_simobserve(name=name, incenter=incenter, chanwidth=0.015625, antenaCfg=antena4, degree=degree, minutes=minutes, seconds=seconds)
            s = f"{incenter}-B4/{incenter}-B4.antennaB4.ms"
            sf = f"{incenter}-B4.antennaB4.ms"
            shutil.move(os.path.join(band4_dir, s), band4_dir)
            shutil.rmtree(f"{incenter}-B4")
            spw_b4.append(sf)

        #Concatenar las ventanas
        output_ms_b4 = f"{indexDat}-B4-concat"
        
        concat(vis=spw_b4, concatvis=output_ms_b4, freqtol=tolerance)
        import time
        t0 = time.time()
        #Correr tclean para el disco concatenado
        tclean(vis=output_ms_b4, cell="0.028arcsec", imagename=output_ms_b4,
                imsize=[512,512], threshold='0mJy', weighting='briggs',
                robust=0, niter=0, interactive=False, gridder='standard')
        tf = time.time()

        print(f"Tiempo tclean {tf-t0}")
        ia = image()
        ia.open(f"{output_ms_b4}.image")
        ia.tofits(outfile=f"{indexDat}-B4.fits", overwrite=True)
        ia.close()

        #Remover archivos basura
        delete_trash()
        #Volver a la ruta anterior
        os.chdir(actual_dir)

        #image para banda 8
        band8_dir = os.path.join(actual_dir, "band8")
        if not os.path.exists(band8_dir):
            os.mkdir(band8_dir)

        os.chdir(band8_dir)
        
        run_radmc(new_dir, band8_dir, father_dir, indexDat, mstar, tstar, rstar, gamma,rc, H100, mdisk, psi, lamb1, incl, posang)

        spw_b8 = []
        for incenter in incenter_per_spw:
            name = str(incenter)+"-B8"
            run_simobserve(name=name, incenter=incenter, chanwidth=0.015625, antenaCfg=antena8, degree=degree, minutes=minutes, seconds=seconds)
            s = f"{incenter}-B8/{incenter}-B8.antennaB8.ms"
            sf = f"{incenter}-B8.antennaB8.ms"
            shutil.move(os.path.join(band8_dir, s), band8_dir)
            shutil.rmtree(f"{incenter}-B8")
            spw_b8.append(sf)

        #Concatenar las ventanas
        output_ms_b8 = f"{indexDat}-B8-concat"
        
        concat(vis=spw_b8, concatvis=output_ms_b8, freqtol=tolerance)

        #Correr tclean para el disco concatenado
        tclean(vis=output_ms_b8, cell="0.028arcsec", imagename=output_ms_b8,
                imsize=[512,512], threshold='0mJy', weighting='briggs',
                robust=0, niter=0, interactive=False, gridder='standard')

        ia = image()
        ia.open(f"{output_ms_b8}.image")
        ia.tofits(outfile=f"{indexDat}-B8.fits", overwrite=True)
        ia.close()

        delete_trash()
        print("Exito")
    except Exception as e:
        return None 
    finally:
        os.chdir(root_dir)

    return 0

def main(args):
    
    root = os.getcwd()
    files = "data"
    if not os.path.exists(files):
        os.mkdir(files)
    os.chdir(files)
    skyModel(args.dat_file)
    os.chdir(root)

    os.system("rm *.log")

if __name__ == '__main__':
    args= parse_option()
    main(args)
