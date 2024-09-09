import os
import argparse
import numpy as np
import concurrent.futures

def parse_option():
    parser = argparse.ArgumentParser('Write RADMC3D.inp', add_help=False)
    parser.add_argument('--nphot', type=str, required=True, metavar="FILE", help='Number of photons', )

    args, unparsed = parser.parse_known_args()

    return args

def writeRADMC(args):
        # Escritura del archivo radmc3d.inp.
        with open('radmc3d.inp','w+') as f:
            f.write('nphot_therm = %d\n'%(int(args.nphot)))
            f.write('nphot_scat = %d\n'%(int(args.nphot)))
            f.write('scattering_mode_max = 0\n')
            f.write('iranfreqmode = 1\n')
            f.write('incl_dust = 1\n')
            f.write('istar_sphere = 1\n')
            f.write('tgas_eq_tdust = 1\n')
            f.write('modified_random_walk = 1\n')
            f.close()

if __name__ == '__main__':
    args= parse_option()
    writeRADMC(args)