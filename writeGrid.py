import numpy as np

pi = 3.141592653589793116
def writeGrid(ntheta,nphi, rmin, rmax, nr):

    # Se crean los vectores de los puntos de la grilla.
    ri  = np.logspace(np.log10(rmin),np.log10(rmax),nr+1)                       # Generación de espacios en el eje radial de Rmin a Rmax.
    thet1    = np.linspace(pi/180,pi/6,int(ntheta*0.1)+1)                                    # Theta de 1° a 30°. 
    thet2    = np.linspace(pi/6,pi/3,int(ntheta*0.2)+1)                                    # Theta de 30° a 60°.
    thet3    = np.linspace(pi/3,pi/2,ntheta-int(ntheta*0.1)-int(ntheta*0.2)+1)     # Theta de 60° a 90°.
    thetai   = np.concatenate((thet1,thet2[1:],thet3[1:]),axis=0)                              # Theta de 0° a 90°.
    phii     = np.linspace(0,2*np.pi,nphi+1)                                              # Generación de espacios en el eje phial de 0° a 360°.

    # Se escribe el archivo de entrada amr_grid.inp.
    #print(msg)
    with open('amr_grid.inp','w+') as f:
        f.write('1\n')                                              # iformat
        f.write('0\n')                                              # AMR grid style  (0=regular grid, no AMR)
        f.write('100\n')                                            # Coordinate system: spherical
        f.write('0\n')                                              # gridinfo
        f.write('1 1 0\n')                                          # Include r,theta coordinates
        f.write('%d %d %d\n'%(nr,ntheta,nphi))       # Size of grid
        for value in ri:
            f.write('%13.6e\t'%(value))      # X coordinates (cell walls)
        f.write("\n")
        for value in thetai:
            f.write('%13.6e\t'%(value))      # Y coordinates (cell walls)
        f.write("\n")
        for value in phii:
            f.write('%13.6e\t'%(value))      # Z coordinates (cell walls)
        f.write("\n")
        f.close()

if __name__ == '__main__':
    rmin = 5
    rmax = 300
    nr = 128
    ntheta = 128
    nphi = 1
    writeGrid(ntheta, nphi, rmin, rmax, nr)