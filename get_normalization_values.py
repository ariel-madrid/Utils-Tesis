from astropy.io import fits
import numpy as np
import pandas as pd
import os
import argparse
import torch

def parse_option():
    parser = argparse.ArgumentParser('Swin Transformer pre process', add_help=False)
    parser.add_argument('--fits-path', type=str, required=True, metavar="FILE", help='path to fits files', )

    args, unparsed = parser.parse_known_args()

    return args

def get_mean(path, fits_files):
    mean_real = []
    mean_imag = []

    for file in fits_files:
        path_file = f"{path}/{file}"
        try:
            with fits.open(path_file) as hdul:
                data = hdul[0].data

            # Calcular la FFT bidimensional
            fft_data = np.fft.fft2(data)

            # Obtener los canales real e imaginario
            real_channel = np.real(fft_data)
            imag_channel = np.imag(fft_data)

            # Añadir las medias de los canales real e imaginario
            mean_real.append(real_channel.mean())
            mean_imag.append(imag_channel.mean())
            
            # Información adicional para depuración
            print(f"Archivo: {file}, Mínimo canal real: {real_channel.min()}, Maximo canal real: {real_channel.max()}, Media parte real: {real_channel.mean()}")
            print(f"Archivo: {file}, Mínimo canal real: {imag_channel.min()}, Maximo canal real: {imag_channel.max()}, Media parte real: {imag_channel.mean()}")
        
        except Exception as e:
            print(f"Error al procesar el archivo {file}: {e}")
            continue

    # Calcular la media final
    mean_real = np.array(mean_real)
    mean_imag = np.array(mean_imag)

    return mean_real.mean(), mean_imag.mean()

def get_sd(path, fits_files, mean_real, mean_imag):
    acum_real = 0
    acum_imag = 0
    N = 0

    # Obtener dimensiones del primer archivo para calcular el tamaño de la imagen
    with fits.open(f"{path}/{fits_files[0]}") as hdul:
        data_shape = hdul[0].data.shape
    num_elements_per_image = np.prod(data_shape)
    total_elements = num_elements_per_image * len(fits_files)

    for file in fits_files:
        path_file = f"{path}/{file}"
        try:
            with fits.open(path_file) as hdul:
                data = hdul[0].data.astype(np.float32)

            # Calcular la FFT bidimensional
            fft_data = np.fft.fft2(data)

            # Obtener los canales real e imaginario
            real_channel = np.real(fft_data)
            imag_channel = np.imag(fft_data)

            # Convertir a tensores para torch
            real_tensor = torch.tensor(real_channel, dtype=torch.float32)
            imag_tensor = torch.tensor(imag_channel, dtype=torch.float32)

            # Parte real
            diff_real = real_tensor - mean_real
            squared_diff_real = diff_real**2
            sum_squared_diff_real = torch.sum(squared_diff_real)

            acum_real += sum_squared_diff_real

            # Parte imaginaria
            diff_imag = imag_tensor - mean_imag
            squared_diff_imag = diff_imag**2
            sum_squared_diff_imag = torch.sum(squared_diff_imag)

            acum_imag += sum_squared_diff_imag

        except Exception as e:
            print(f"Error al procesar el archivo {file}: {e}")
            continue

    # Calcular desviación estándar
    sd_real = torch.sqrt(acum_real / total_elements)
    sd_imag = torch.sqrt(acum_imag / total_elements)

    return sd_real, sd_imag

def main(args):
    path = args.fits_path
    fits_files = [f for f in os.listdir(args.fits_path)]

    #Obtener la media de las visibilidades
    mean_real, mean_imag = get_mean(path, fits_files)
    
    #Obtener la desviacion estandar de las visibilidades
    sd_real, sd_imag = get_sd(path, fits_files, mean_real, mean_imag)

    data = {
         "mean_real": mean_real,
         "mean_imag": mean_imag,
         "sd_real": sd_real.item(),
         "sd_imag": sd_imag.item()
    }
    df = pd.DataFrame(data,index=[0])
    
    df.to_csv('normalization_values.csv', index=False)
    
if __name__ == '__main__':
    args= parse_option()
    main(args)
