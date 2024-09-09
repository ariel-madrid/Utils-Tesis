import os
import argparse
import numpy as np

def writeWavelenght():

    # Se crean los valores de longitud de onda.
    lam1        = 0.1e0
    lam2        = 7.0e0
    lam3        = 25.e0
    lam4        = 1.0e4
    n12         = 20
    n23         = 100
    n34         = 30
    lam12       = np.logspace(np.log10(lam1),np.log10(lam2),n12,endpoint=False)
    lam23       = np.logspace(np.log10(lam2),np.log10(lam3),n23,endpoint=False)
    lam34       = np.logspace(np.log10(lam3),np.log10(lam4),n34,endpoint=True)
    lam    = np.concatenate([lam12,lam23,lam34])

    # Se crea el archivo de entrada wavelenght_micron.inp.
    #print(msg)
    with open('wavelength_micron.inp','w+') as f:
        f.write('%d\n'%(lam.size))
        for value in lam:
            f.write('%13.6e\n'%(value))
        f.close()

if __name__ == '__main__':
    writeWavelenght()