import os
import argparse
import numpy as np
import concurrent.futures

au  = 1.49598e13                                    # Astronomical Unit       [cm]
pc  = 3.08572e18                                    # Parsec                  [cm]
ms  = 1.98892e33                                    # Solar mass              [g]
ts  = 5.78e3                                        # Solar temperature       [K]
ls  = 3.8525e33                                     # Solar luminosity        [erg/s]
rs  = 6.96e10                                       # Solar radius            [cm]
pi = 3.14159265358979311

def parse_option():
    parser = argparse.ArgumentParser('Write Densities', add_help=False)
    parser.add_argument('--mdisk', type=str, required=True, metavar="FILE", help='Disk mass', )
    parser.add_argument('--rc', type=str, required=True, metavar="FILE", help='Disk radio', )
    parser.add_argument('--gamma', type=str, required=True, metavar="FILE", help='gamma', )
    parser.add_argument('--psi', type=str, required=True, metavar="FILE", help='psi', )
    parser.add_argument('--H100', type=str, required=True, metavar="FILE", help='H100', )

    args, unparsed = parser.parse_known_args()

    return args

def writeDensities(ntheta,nphi, rmin, rmax, nr, nspec, mDisk, gamma, Rc, psi, H100):
    
    ri  = np.logspace(np.log10(rmin),np.log10(rmax),nr+1)                       # Generación de espacios en el eje radial de Rmin a Rmax.
    thet1    = np.linspace(pi/180,pi/6,int(ntheta*0.1)+1)                                    # Theta de 1° a 30°. 
    thet2    = np.linspace(pi/6,pi/3,int(ntheta*0.2)+1)                                    # Theta de 30° a 60°.
    thet3    = np.linspace(pi/3,pi/2,ntheta-int(ntheta*0.1)-int(ntheta*0.2)+1)     # Theta de 60° a 90°.
    thetai   = np.concatenate((thet1,thet2[1:],thet3[1:]),axis=0)                              # Theta de 0° a 90°.
    phii     = np.linspace(0,2*np.pi,nphi+1) 


    # Normalización de la masa.
    r,theta,phi = np.meshgrid(ri,thetai,phii)
    r2 = r**(2)
    jacob = r2[:-1,:-1,:-1] * np.sin(theta[:-1,:-1,:-1])
    diffRad = r[:-1,1:,:-1]-r[:-1,:-1,:-1]
    diffThet = theta[1:,:-1,:-1]-theta[:-1,:-1,:-1]
    diffphi = phi[:-1,:-1,1:]-phi[:-1,:-1,:-1]
    vol = jacob * diffRad * diffThet * diffphi
    vol = np.swapaxes(vol,0,1)

    # Obtención del parámetro sigma c para realizar el cálculo.
    sigmac  = (mDisk*(2.-gamma))/(2.*pi*(Rc**(2.)))

    # Cálculo de la densidad para cada especie de polvo.
    density = []
    hightScale = []
    surfaceDensity = []
    for s in range(nspec):

        # Se setean los valores en 0 de la densidad, hight scale y densidad superficial.
        rhos = np.zeros((nr,ntheta,nphi))
        haches = np.zeros((nr,ntheta,nphi)) 
        sigmas = np.zeros((nr,ntheta,nphi))  

        for k in range(nphi):
            for j in range(ntheta):
                for i in range(nr):
                    rad, thet, ph = ri[i],thetai[j],phii[k]
        
                    # 1.- PARAMETROS DE ALTURA Y RADIO POR COORDENADA (Z, rCil).
                    altura  = (np.pi/2 - thet)                        # Andrew V1.
                    rCil    = rad*np.sin(thet)                          # Radio Circular.                    

                    # 2.- OBTENCIÓN LAS DENSIDADES SUPERFICIALES.
                    sigma   = sigmac*((rCil/Rc)**(-gamma))                 # Sigma parte 1.
                    sigma   = sigma*np.exp(-((rCil/Rc)**(2.-gamma)))            # Sigma parte 2.

                    # 2.5.- EVALUACIÓN DE MÁS DE UNA ESPECIE.
                    if (nspec > 1):

                        amin = 0.1*10**(-6)
                        amed = 5*10**(-6)
                        amax = 2500*10**(-6)
                        
                        if (s == 0):
                            sigma = sigma*((np.sqrt(amed)-np.sqrt(amin))/(np.sqrt(amax)-np.sqrt(amin)))
                        else:
                            sigma = sigma*((np.sqrt(amax)-np.sqrt(amed))/(np.sqrt(amax)-np.sqrt(amin)))
                    
                    sigmas[i,j,k] = sigma

                    # 3.- ONTENCIÓN DEL HIGHT SCALE DEPENDIENDO SI SON 1 O 2 ESPECIES.
                    h       = (H100/((100*au)**(psi+1)))*(rCil**(psi))       # Andrews V1.
                    h       = h if (s==0) else 0.2*h
                    haches[i,j,k] = h

                    # 4.- EXPANSIÓN CON DISTRIBUCIÓN GAUSSIANA.
                    expe    = -(1/2)*((altura/h)**(2.))                                   # Andrews V1.
                    dens    = (sigma/(h*rCil*np.sqrt(2.*np.pi)))*np.exp(expe)             # Andrews V1.

                    # 5.- APLICACIÓN DE DENSIDAD DE FONDO.
                    rhos[i,j,k] = dens + 1.e-33
        
        density.append(rhos)
        hightScale.append(haches)
        surfaceDensity.append(sigmas)

    # Se guardan los valores de rhos, haches y sigmas.
    rhos = np.array(density)
    haches = np.array(hightScale)
    sigmas = np.array(surfaceDensity)

    sumat = []
    for s in range(nspec):
        sumat.append(2.*rhos[s,:,:,:]*epsilon*vol)
    sumat = np.array(sumat)

    # print(rhos.shape,haches.shape,sigmas.shape)
    total_mass = np.sum(sumat,axis=(0,1,2,3)) 
    mass_norm = mDisk*(epsilon/total_mass)

    # Se normaliza la masa.
    for s in range(nspec):
        for k in range(nphi):
            for j in range(ntheta):
                for i in range(nr):
                    rhos[s,i,j,k] = rhos[s,i,j,k]*epsilon*mass_norm

    # Escritura del archivo dust_density.inp.
    with open('dust_density.inp','w+') as f: 
        f.write('1\n')                                      # Format number
        f.write('%d\n'%(nr*ntheta*nphi))     # Nr of cells
        f.write('%d\n'%(nspec))                        # Nr of dust species
        for s in range(nspec):
            data = rhos[s].ravel(order='F')                   # Create a 1-D view, fortran-style indexing
            data.tofile(f, sep='\n', format="%13.6e")
            f.write('\n')
        f.close()

if __name__ == '__main__':
    rmin = 5
    rmax = 300
    nr = 128
    ntheta = 128
    nphi = 1
    nspec = 1
    
    args= parse_option()

    writeGrid(ntheta,nphi, rmin, rmax, nr, nspec, args.Disk, args.gamma, args.Rc, args.psi, args.H100)