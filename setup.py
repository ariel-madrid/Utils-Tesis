# --------------------------------------------------------------------------------------------------------------------
#                                                  LIBRARIES
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

import argparse
from models import *


# --------------------------------------------------------------------------------------------------------------------
#                                                INPUT OF ENTRY
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

# MODEL PARAMETERS.
# --------------------------------------------------------------------------------------------------------------------

# 1.- Parameters of the scrip.
# --name      = name of the disk (is an identificator)
# --mode      = execution mode: 1 - Simulate one disk 2- Database generation
# --ident     = identificator of the disk (is used in --mode 2)

# 1.- Parameters of the star.
# --mstar     = mass of the star                                        (solar masses)
# --rstar     = radius of the star                                      (solar radii)
# --tstar     = temperature of the star                                 (degrees kelvin)

# 2.- Grid parameters.
# --nr        = number of rays                                          (scalar)
# --ntheta    = number of theta angles                                  (scalar)
# --nphi      = number of angles phi                                    (scalar)
# --rMin      = minimum radius                                          (astronomical units)
# --rMax      = maximum radius                                          (astronomical units)

# 3.- Structural parameters.
# --Rc        = characteristic radius                                   (astronomical units)
# --mDisk     = mass of the disk                                        (solar masses)
# --gamma     = surface density gradient                                (scalar)
# --hc        = scale of the height at the characteristic radius Rc     (astronomical units)
# --psi       = degree of disc enlargement                              (scalar)

# 4.- Geometric parameters.
# --lamb      = wavelength (it is a list of incl)                       (micrometers)
# --incl      = disc inclination (it is a list of incl)                 (degrees)
# --posang    = disc posicion angular (it is a list of PA)              (degress)
# --dpc       = distance from the source                                (parsec)
# --npix      = number of pixels for radmc3d image                      (pixels)

# 5.- Execution parameters.
# --nphot     = number of photons in the simulation.
# --nspec     = number of dust species in the simulation.
# --nthread   = number of threads for the simulation.
# --sizeau    = image size in astronomical units.

# --------------------------------------------------------------------------------------------------------------------

# Dependiendo de si se ingresan o no argumentos de entrada setea los parámetros con diferentes valores.
if __name__ == "__main__":

    # Input parameters are received.
    parser = argparse.ArgumentParser(description='Arguments of protoplanetary disk')
    parser.add_argument('--mode',type = int,required=True,help='Specify execution mode: 1 - Simulate one disk 2 - Database generation')
    parser.add_argument('--name',default='disk',type = str,help='Specify the name of the disk')
    parser.add_argument('--ident',default=1,type = int,help='Specify the identificator of the disk')
    parser.add_argument('--mstar',type = float,help='Specify star\'s mass')
    parser.add_argument('--rstar',type = float,help='Specify star\'s radius')
    parser.add_argument('--tstar',type = int,help='Specify star\'s temperature')
    parser.add_argument('--nr',type = int,help='Specify the number of radius for the grid')
    parser.add_argument('--ntheta',type = int,help='Specify the number of thetas for the grid')
    parser.add_argument('--nphi',type = int,help='Specify the number of phis for the grid')
    parser.add_argument('--rMin',type = int,help='Specify the minimum radius of the grid')
    parser.add_argument('--rMax',type = int,help='Specify the maximun radius of the grid')
    parser.add_argument('--Rc',type = float,help='Specify the characteristic radius of the disk')
    parser.add_argument('--mDisk',type = float,help='Specify the mass of the disk')
    parser.add_argument('--gamma',type = float,help='Specify the areal density gradient of the disk')
    parser.add_argument('--H100',type = float,help='Specify the height of the disk to 100 astronomical units')
    parser.add_argument('--psi',type = float,help='Specify the flaring index of the disk')
    parser.add_argument('--lamb',type = float,nargs='+',help='Specify the wavelenght of observation')
    parser.add_argument('--incl',type = int,nargs='+',help='Specify the inclination of the disk (it is a list)')
    parser.add_argument('--posang',type = int,nargs='+',help='Specify the posicion angular of the disk (it is a list)')
    parser.add_argument('--dpc',type = int,help='Specify the distance from the source') #140
    parser.add_argument('--npix',type = int,help='Specify the number of pixels in the RADMC3D image') #512
    parser.add_argument('--nphot',type = int,help='Specify the number of photons for the RADMC simulation')
    parser.add_argument('--nspec',type = int,help='Specify the number of dust species on the disk') #1
    parser.add_argument('--nthread',type = int,required=True,help='Specify the number of threads for the RADMC simulation')
    parser.add_argument('--sizeau',type = int,help='Specify the image size in astronomical units') #420
    args = parser.parse_args()

    disk = protoDisk(
        mstar = args.mstar,
        rstar = args.rstar,
        tstar = args.tstar,
        nr = args.nr,
        ntheta = args.ntheta,
        nphi = args.nphi,
        rMin = args.rMin,
        rMax = args.rMax,
        Rc = args.Rc,
        mDisk = args.mDisk,
        gamma = args.gamma,
        H100 = args.H100,
        psi = args.psi,
        lamb = args.lamb,
        incl = args.incl,
        posang = args.posang,
        dpc = args.dpc,
        npix = args.npix,
        nphot = args.nphot,
        nspec = args.nspec,
        sizeau = args.sizeau
    )


    # MOSTRAR RESUMEN.
    # --------------------------------------------------------------------------------------------------------------------
    #disk.showInputs(show=True)
    
    
    # ESCRITURA DE LOS ARCHIVOS.
    # --------------------------------------------------------------------------------------------------------------------
    disk.writeGrid(msg="Writing amr_grid.inp")
    disk.writeWavelenght(msg="Writing wavelength_micron.inp")
    disk.writeStars(msg="Writing stars.inp")
    disk.writeOpac(msg="Writing dustopac.inp")
    disk.writeRADMC(msg="Writing radmc3d.inp")
    disk.writeDensities(epsilon=0.01,msg="Writing dust_density.inp")

    # EJECUCUCIÓN DE RADMC3D MCTHERM.
    # --------------------------------------------------------------------------------------------------------------------
    #disk.calculeTemp(threads=args.nthread,output=args.mode)
    # GENERACIÓN DE LA IMAGEN.
    # --------------------------------------------------------------------------------------------------------------------
    disk.generateSkyModel(output=args.mode,name=args.name,ident=args.ident)

