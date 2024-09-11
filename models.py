# --------------------------------------------------------------------------------------------------------------------
#                                             BIBLIOTECAS Y FUNCIONES
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------

import csv
import numpy as np
import matplotlib.pyplot as plt
from radmc3dPy import *
from radmc3dPy.image import *
from radmc3dPy.analyze import *
from astropy.io import fits


# --------------------------------------------------------------------------------------------------------------------
#                                           MODELO DE DISCO PARAMÉTRICO
# --------------------------------------------------------------------------------------------------------------------
# --------------------------------------------------------------------------------------------------------------------
au  = 1.49598e13                                    # Astronomical Unit       [cm]
pc  = 3.08572e18                                    # Parsec                  [cm]
ms  = 1.98892e33                                    # Solar mass              [g]
ts  = 5.78e3                                        # Solar temperature       [K]
ls  = 3.8525e33                                     # Solar luminosity        [erg/s]
rs  = 6.96e10                                       # Solar radius            [cm]
pi = 3.141592653589793116                           # pi
# MODELO DEL DISCO PROTOPLANETARIO.
class protoDisk:
    # 0. Método constructor.
    def __init__(self,
            # Parámetros de la estrella.
            mstar, rstar, tstar,
            # Parámetros de la grilla.
            nr, ntheta, nphi, rMin, rMax,
            # Parámetros estructurales del disco.
            Rc, mDisk, gamma, H100, psi,
            # Parámetros geométricos.
            lamb, incl, posang, npix, dpc,
            # Parámetros de la simulación.
            nphot, nspec, sizeau):
        
        # Seteo de los parámetros de entrada -------------------------------------------------------------------------
        # ------------------------------------------------------------------------------------------------------------
        self.mstar  = mstar*ms if mstar is not None else 1.3*ms
        self.rstar  = rstar*rs if rstar is not None else 1*rs
        self.tstar  = tstar if tstar is not None else 10000
        self.nr     = nr if nr is not None else 128
        self.ntheta = ntheta if ntheta is not None else 128
        self.nphi   = nphi if nphi is not None else 1
        self.rMin   = rMin*au if rMin is not None else 1*au
        self.rMax   = rMax*au if rMax is not None else 150*au
        self.Rc     = Rc*au if Rc is not None else 50*au
        self.mDisk  = mDisk*ms if mDisk is not None else 0.0095*ms
        self.gamma  = gamma if gamma is not None else 1
        self.H100   = H100*au if H100 is not None else 4.5*au
        self.psi    = psi if psi is not None else 1.25
        self.lamb   = lamb if lamb is not None else [1300]
        self.incl   = incl if incl is not None else [0]
        self.posang = posang if posang is not None else [0]
        self.dpc    = dpc if dpc is not None else 140
        self.npix   = npix if npix is not None else 512
        self.nphot  = nphot if nphot is not None else 2000000
        self.nspec  = nspec if nspec is not None else 1
        self.sizeau = sizeau if sizeau is not None else 420

    # Métodos del modelo -----------------------------------------------------------------------------------------
    # ------------------------------------------------------------------------------------------------------------

    # 1. Método showInputs
    # Muestra un resumen de los parámetros de entrada con los que se va a crear el disco.
    '''def showInputs(self,show=True):

        if show:
            print("\n--------------------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------------------")
            print("                                     GENERATING DISK                                        ")
            print("--------------------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------------------\n")

            print("\n STAR'S PARAMETERS")
            print("--------------------------------------------------------------------------------------------")
            print("            mstar: %.5f ms          rstar: %.1f rs          tstar: %.0f °K"%(round(self.mstar/ms,5),round(self.rstar/rs,1),self.tstar))

            print("\n STRUCTURAL PARAMETERS")
            print("--------------------------------------------------------------------------------------------")
            print("     Rc: %.1f au    mDisk: %.5f ms    gamma: %.2f    H100: %.1f au    flaring: %.2f"%(round(self.Rc/au,1),round(self.mDisk/ms,5),self.gamma,round(self.H100/au,1),self.psi))

            inclinations = " ".join(str(incl)+"°" for incl in self.incl)
            posicionsangular = " ".join(str(posA)+"°" for posA in self.posang)
            print("\n GEOMETRY PARAMETERS")
            print("--------------------------------------------------------------------------------------------")
            print("     Inclinations: "+inclinations+"          Posicions angular: "+posicionsangular)

            lambdas = " ".join("["+str(lamb)+"]" for lamb in self.lamb)
            print("\n SIMULATION PARAMETERS")
            print("--------------------------------------------------------------------------------------------")
            print("     npix: %.0f     dpc: %.0f pc    sizeImg: %.0f au"%(self.npix,self.dpc,self.sizeau)+"    lambda: "+lambdas+" µm")

            print("\n--------------------------------------------------------------------------------------------")
            print("--------------------------------------------------------------------------------------------\n")'''

    # 2. Método writeGrid.
    # Escribe el archivo correspondiente a la grilla del modelo.
    def writeGrid(self,msg):

        # Se crean los vectores de los puntos de la grilla.
        self.ri  = np.logspace(np.log10(self.rMin),np.log10(self.rMax),self.nr+1)                       # Generación de espacios en el eje radial de Rmin a Rmax.
        thet1    = np.linspace(pi/180,pi/6,int(self.ntheta*0.1)+1)                                    # Theta de 1° a 32.1°. 
        thet2    = np.linspace(pi/6,pi/3,int(self.ntheta*0.2)+1)                                    # Theta de 32.1° a 70°.
        thet3    = np.linspace(pi/3,pi/2,self.ntheta-int(self.ntheta*0.1)-int(self.ntheta*0.2)+1)     # Theta de 70° a 90°.
        self.thetai   = np.concatenate((thet1,thet2[1:],thet3[1:]),axis=0)                              # Theta de 0° a 90°.
        self.phii     = np.linspace(0,2*np.pi,self.nphi+1)                                              # Generación de espacios en el eje phial de 0° a 360°.
        
        # Se escribe el archivo de entrada amr_grid.inp.
        #print(msg)
        with open('amr_grid.inp','w+') as f:
            f.write('1\n')                                              # iformat
            f.write('0\n')                                              # AMR grid style  (0=regular grid, no AMR)
            f.write('100\n')                                            # Coordinate system: spherical
            f.write('0\n')                                              # gridinfo
            f.write('1 1 0\n')                                          # Include r,theta coordinates
            f.write('%d %d %d\n'%(self.nr,self.ntheta,self.nphi))       # Size of grid
            for value in self.ri:
                f.write('%13.6e\t'%(value))      # X coordinates (cell walls)
            f.write("\n")
            for value in self.thetai:
                f.write('%13.6e\t'%(value))      # Y coordinates (cell walls)
            f.write("\n")
            for value in self.phii:
                f.write('%13.6e\t'%(value))      # Z coordinates (cell walls)
            f.write("\n")
            f.close()


    # 3. Método writeWavelenght.
    # Escribe el archivo de con la información de las longitudes de ondas.
    def writeWavelenght(self,msg):
        
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
        self.lam    = np.concatenate([lam12,lam23,lam34])

        # Se crea el archivo de entrada wavelenght_micron.inp.
        #print(msg)
        with open('wavelength_micron.inp','w+') as f:
            f.write('%d\n'%(self.lam.size))
            for value in self.lam:
                f.write('%13.6e\n'%(value))
            f.close()


    # 4. Método writeStars.
    # Escribe el archivo que contiene información de la estrella del disco.
    def writeStars(self,msg):
        #print(msg)
        # Se escribe el archivo stars.inp.
        with open('stars.inp','w+') as f:
            f.write('2\n')
            f.write('1 %d\n\n'%(self.lam.size))
            f.write('%13.6e %13.6e %13.6e %13.6e %13.6e\n\n'%(self.rstar,self.mstar,0,0,0))
            for value in self.lam:
                f.write('%13.6e\n'%(value))
            f.write('\n%13.6e\n'%(-self.tstar))
            f.close()
    

    # 5. Método writeOpac.
    # Escribe el archivo que contiene la información de la opacidad del polvo. 
    def writeOpac(self,msg):
        
        # Escritura del archivo dustopac.inp.
        #print(msg)
        with open('dustopac.inp','w+') as f:
            
            f.write('2               Format number of this file\n')
            # Dependiendo del número de especies que se ingrese varía el kappa a utilizar.
            if self.nspec == 1:
                f.write('1               Nr of dust species\n')
                f.write('============================================================================\n')
                f.write('1               Way in which this dust species is read\n')
                f.write('0               0=Thermal grain\n')
                f.write('silicate        Extension of name of dustkappa_***.inp file\n')
                f.write('----------------------------------------------------------------------------\n')
                f.close()

            else:
                f.write('2               Nr of dust species\n')
                f.write('============================================================================\n')
                f.write('1               Way in which this dust species is read\n')
                f.write('0               0=Thermal grain\n')
                f.write('small           Extension of name of dustkappa_***.inp file\n')
                f.write('----------------------------------------------------------------------------\n')
                f.write('1               Way in which this dust species is read\n')
                f.write('0               0=Thermal grain\n')
                f.write('large           Extension of name of dustkappa_***.inp file\n')
                f.write('----------------------------------------------------------------------------\n')


    # 6.- Método writeRADMC.
    # Escribe el archivo con las configuraciones de la simulación de RADMC3D.
    def writeRADMC(self,msg):
        #print(msg)
        # Escritura del archivo radmc3d.inp.
        with open('radmc3d.inp','w+') as f:
            f.write('nphot_therm = %d\n'%(self.nphot))
            f.write('nphot_scat = %d\n'%(self.nphot))
            f.write('scattering_mode_max = 0\n')
            f.write('iranfreqmode = 1\n')
            f.write('incl_dust = 1\n')
            f.write('istar_sphere = 1\n')
            f.write('tgas_eq_tdust = 1\n')
            f.write('modified_random_walk = 1\n')
            f.close()


    # 7. Método writeDensities.
    # Escribe el archivo que tiene los valores de densidad del disco en cada punto de la grilla.
    def writeDensities(self,epsilon,msg):
        
        # Normalización de la masa.
        r,theta,phi = np.meshgrid(self.ri,self.thetai,self.phii)
        r2 = r**(2)
        jacob = r2[:-1,:-1,:-1] * np.sin(theta[:-1,:-1,:-1])
        diffRad = r[:-1,1:,:-1]-r[:-1,:-1,:-1]
        diffThet = theta[1:,:-1,:-1]-theta[:-1,:-1,:-1]
        diffphi = phi[:-1,:-1,1:]-phi[:-1,:-1,:-1]
        vol = jacob * diffRad * diffThet * diffphi
        vol = np.swapaxes(vol,0,1)

        # Obtención del parámetro sigma c para realizar el cálculo.
        self.sigmac  = (self.mDisk*(2.-self.gamma))/(2.*pi*(self.Rc**(2.)))

        # Cálculo de la densidad para cada especie de polvo.
        density = []
        hightScale = []
        surfaceDensity = []
        for s in range(self.nspec):

            # Se setean los valores en 0 de la densidad, hight scale y densidad superficial.
            rhos = np.zeros((self.nr,self.ntheta,self.nphi))
            haches = np.zeros((self.nr,self.ntheta,self.nphi)) 
            sigmas = np.zeros((self.nr,self.ntheta,self.nphi))  

            for k in range(self.nphi):
                for j in range(self.ntheta):
                    for i in range(self.nr):
                        rad, thet, ph = self.ri[i],self.thetai[j],self.phii[k]
            
                        # 1.- PARAMETROS DE ALTURA Y RADIO POR COORDENADA (Z, rCil).
                        altura  = (np.pi/2 - thet)                        # Andrew V1.
                        rCil    = rad*np.sin(thet)                          # Radio Circular.                    

                        # 2.- OBTENCIÓN LAS DENSIDADES SUPERFICIALES.
                        sigma   = self.sigmac*((rCil/self.Rc)**(-self.gamma))                 # Sigma parte 1.
                        sigma   = sigma*np.exp(-((rCil/self.Rc)**(2.-self.gamma)))            # Sigma parte 2.

                        # 2.5.- EVALUACIÓN DE MÁS DE UNA ESPECIE.
                        if (self.nspec > 1):

                            amin = 0.1*10**(-6)
                            amed = 5*10**(-6)
                            amax = 2500*10**(-6)
                            
                            if (s == 0):
                                sigma = sigma*((np.sqrt(amed)-np.sqrt(amin))/(np.sqrt(amax)-np.sqrt(amin)))
                            else:
                                sigma = sigma*((np.sqrt(amax)-np.sqrt(amed))/(np.sqrt(amax)-np.sqrt(amin)))
                        
                        sigmas[i,j,k] = sigma

                        # 3.- ONTENCIÓN DEL HIGHT SCALE DEPENDIENDO SI SON 1 O 2 ESPECIES.
                        h       = (self.H100/((100*au)**(self.psi+1)))*(rCil**(self.psi))       # Andrews V1.
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
        self.rhos = np.array(density)
        self.haches = np.array(hightScale)
        self.sigmas = np.array(surfaceDensity)

        sumat = []
        for s in range(self.nspec):
            sumat.append(2.*self.rhos[s,:,:,:]*epsilon*vol)
        sumat = np.array(sumat)

        # print(self.rhos.shape,self.haches.shape,self.sigmas.shape)
        total_mass = np.sum(sumat,axis=(0,1,2,3)) 
        mass_norm = self.mDisk*(epsilon/total_mass)

        # Se normaliza la masa.
        for s in range(self.nspec):
            for k in range(self.nphi):
                for j in range(self.ntheta):
                    for i in range(self.nr):
                        self.rhos[s,i,j,k] = self.rhos[s,i,j,k]*epsilon*mass_norm

        #print(msg)
        # Escritura del archivo dust_density.inp.
        with open('dust_density.inp','w+') as f: 
            f.write('1\n')                                      # Format number
            f.write('%d\n'%(self.nr*self.ntheta*self.nphi))     # Nr of cells
            f.write('%d\n'%(self.nspec))                        # Nr of dust species
            for s in range(self.nspec):
                data = self.rhos[s].ravel(order='F')                   # Create a 1-D view, fortran-style indexing
                data.tofile(f, sep='\n', format="%13.6e")
                f.write('\n')
            f.close()
    

    # 9.  Método calculeTemp.
    # Realiza la ejecución de RADMC y guarda los resultados.
    def calculeTemp(self,threads,output):

        # Se ejecuta radmc3d mctherm sin pantalla y guarda registro en un archivo.
        os.system("radmc3d mctherm setthreads "+str(threads)+" > output.txt")
            
        # Se guardan los datos en el modelo.
        dataFig = readData()
        self.rr,self.tt = np.meshgrid(dataFig.grid.x,dataFig.grid.y,indexing='ij')
        self.temp = dataFig.dusttemp


    # 10. Método plotDensities.
    # Muestra un gráfico de calor 2D de la densidad del disco mirado del plano medio.
    def plotDensities(self,vMin,vMax,file):

        # Se obtienen los ejes del gráfico.
        ejeX = self.rr*np.sin(self.tt)/au     
        ejeZ = self.rr*np.cos(self.tt)/au

        # Se genera el gráfico.
        size = (10,7.5) if self.nspec == 1 else (7.5,12)
        graf, ax = plt.subplots(nrows=self.nspec,ncols=1,figsize=size)
        graf.suptitle("$R_{c}:$ %.2f au    $M_{Disk}:$ %.5f ms    $\gamma:$ %.2f    $H_{100}:$ %.2f au    $\psi:$ %.2f"%(self.Rc/au,self.mDisk/ms,self.gamma,self.H100/au,self.psi))

        if (self.nspec == 1):
            var = ax.pcolor(ejeX,ejeZ,self.rhos[0,:,:,0],cmap='hot',edgecolors='k',linewidths=0.1,vmin=vMin,vmax=vMax)
            ax.set_xlim(0,(self.rMax/au)*1.02)
            ax.set_ylim(0,(self.rMax/au)*np.cos(4*pi/9)*1.02)
            plt.title("Density "+r'$\rho_{\mathrm{dust}}\;[\mathrm{g}/\mathrm{cm}^3]$')
            plt.xlabel('Radius (Au) [X Axis]')
            plt.ylabel('Height (Au) [Z Axis]')
            plt.colorbar(var,ax=ax,label='$\log_{10}($'+r'$\rho_{dust})$')
        
        else:
            for i in range(self.nspec):
                var = ax[i].pcolor(ejeX,ejeZ,self.rhos[i,:,:,0],cmap='hot',edgecolors='k',linewidths=0.1,vmin=vMin,vmax=vMax)
                ax[i].set_xlim(0,(self.rMax/au)*1.02)
                ax[i].set_ylim(0,(self.rMax/au)*np.cos(4*pi/9)*1.02)
                if (i == 0):
                    ax[i].set_title("Density "+r'$\rho_{\mathrm{dust}}\;[\mathrm{g}/\mathrm{cm}^3]$'+" small dust")
                else:
                    ax[i].set_title("Density "+r'$\rho_{\mathrm{dust}}\;[\mathrm{g}/\mathrm{cm}^3]$'+" large dust")
                ax[i].set_xlabel('Radius (Au) [X Axis]')
                ax[i].set_ylabel('Height (Au) [Z Axis]')
                plt.colorbar(var,ax=ax[i],label='$\log_{10}($'+r'$\rho_{dust})$')
        
        plt.tight_layout()
        plt.savefig(file+"_density.png")
        plt.close(graf)


    # 11. Método plotTemperature.
    # Muestra un gráfico de calor 2D de la temperatura del disco mirado del plano medio.
    def plotTemperature(self,vMin,vMax,file):

        # Se obtienen los ejes del gráfico.
        ejeX = self.rr*np.sin(self.tt)/au     
        ejeZ = self.rr*np.cos(self.tt)/au

        temperatures = []
        for s in range(self.nspec):
            temperature = self.temp[:,:,0,s]
            temperatures.append(temperature)
        temperatures = np.array(temperatures)

        # Se genera el gráfico.
        size = (10,7.5) if self.nspec == 1 else (7.5,12)
        graf, ax = plt.subplots(nrows=self.nspec,ncols=1,figsize=size)
        graf.suptitle("$R_{c}:$ %.2f au    $M_{Disk}:$ %.5f ms    $\gamma:$ %.2f    $H_{100}:$ %.2f au    $\psi:$ %.2f"%(self.Rc/au,self.mDisk/ms,self.gamma,self.H100/au,self.psi))

        if (self.nspec == 1):
            var = ax.pcolor(ejeX,ejeZ,temperatures[0],cmap='hot',vmin=vMin,vmax=vMax)
            ax.set_xlim(0,(self.rMax/au)*1.02)
            ax.set_ylim(0,(self.rMax/au)*1.02)
            plt.title("Temperature "+r'$T_{dust}$'+" °K")
            plt.xlabel('Radius (Au) [X Axis]')
            plt.ylabel('Height (Au) [Z Axis]')
            plt.colorbar(var,ax=ax,label='$(T_{dust})$ °K')
        
        else:
            for i in range(self.nspec):
                var = ax[i].pcolor(ejeX,ejeZ,temperatures[i],cmap='hot',vmin=vMin,vmax=vMax)
                ax[i].set_xlim(0,(self.rMax/au)*1.02)
                ax[i].set_ylim(0,(self.rMax/au)*1.02)
                if (i == 0):
                    ax[i].set_title("Temperature "+r'$T_{dust}$'+" °K"+" small dust")
                else:
                    ax[i].set_title("Temperature "+r'$T_{dust}$'+" °K"+" large dust")
                ax[i].set_xlabel('Radius (Au) [X Axis]')
                ax[i].set_ylabel('Height (Au) [Z Axis]')
                plt.colorbar(var,ax=ax[i],label='$(T_{dust})$ °K')

        plt.tight_layout()
        plt.savefig(file+"_temperature.png")
        plt.close(graf)


    # 12. Método plotLinealDensity
    # Muestra el gráfico de densidad (sigma, no rho) como un gráfico de linea 2D.
    def plotLinealDensity(self,file):

        # Se calcula la densidad lineal para el plano radial.
        sigma = (self.sigmac*(self.ri/self.Rc)**(-self.gamma))*np.exp(-(self.ri/self.Rc)**(2-self.gamma))
        sigmaMax = np.max(sigma)

        # Se genera el gráfico.
        size = (10,7.5)
        graf = plt.figure(figsize=size)
        graf.suptitle("$R_{c}:$ %.2f au    $M_{Disk}:$ %.5f ms    $\gamma:$ %.2f    $H_{100}:$ %.2f au    $\psi:$ %.2f"%(self.Rc/au,self.mDisk/ms,self.gamma,self.H100/au,self.psi))
        ax = plt.gca()

        if (self.nspec == 1):
            np.save(file+"_linealDensity.npy",sigma)
            ax.plot(self.ri/au,sigma,linewidth=4)
            
        else:

            amin = 0.1*10**(-6)
            amed = 5*10**(-6)
            amax = 2500*10**(-6)

            sigmaS = sigma*((np.sqrt(amed)-np.sqrt(amin))/(np.sqrt(amax)-np.sqrt(amin)))
            sigmaL = sigma*((np.sqrt(amax)-np.sqrt(amed))/(np.sqrt(amax)-np.sqrt(amin)))
            ax.plot(self.ri/au,sigmaS,linewidth=4,label=r'$\Sigma$'+" small dust")
            ax.plot(self.ri/au,sigmaL,linewidth=4,label=r'$\Sigma$'+" large dust")
            plt.legend()

        ax.set_xscale("log")
        ax.set_ylim([0,sigmaMax*1.25])
        ax.set_xlim([1,(self.rMax/au)*1.25])
        plt.title("Surface Density "+r'$\Sigma$')
        plt.xlabel('Radius (Au)')
        plt.ylabel(r'$\Sigma$'+" ("+r'$\frac{kg}{m^{2}}$'+")")
        plt.tight_layout()
        plt.savefig(file+"_lineal_density.png")
        plt.close(graf)


    # 13. Método plotLinealHight
    # Muestra el gráfico lineal 2D de como varia la altura (h) a cierto radio.
    def plotLinealHight(self,file):

        # Se calcula el valor de height scale para el plano radial.
        h = self.H100*((self.ri/self.Rc)**(self.psi))
        hMax = np.max(h)/au

        # Se genera el gráfico.
        size = (10,7.5)
        graf = plt.figure(figsize=size)
        graf.suptitle("$R_{c}:$ %.2f au    $M_{Disk}:$ %.5f ms    $\gamma:$ %.2f    $H_{100}:$ %.2f au    $\psi:$ %.2f"%(self.Rc/au,self.mDisk/ms,self.gamma,self.H100/au,self.psi))
        ax = plt.gca()

        if (self.nspec == 1):
            np.save(file+"_linealHigth.npy",h)
            ax.plot(self.ri/au,h/au,linewidth=4)
        
        else:
            ax.plot(self.ri/au,h/au,linewidth=4,label=r'$h$'+" (Au)"+" small dust")
            h = 0.2*(self.H100*((self.ri/self.Rc)**(self.psi)))
            ax.plot(self.ri/au,h/au,linewidth=4,label=r'$h$'+" (Au)"+" large dust")
            plt.legend()
        
        ax.set_xscale("log")
        ax.set_ylim([0,hMax*1.25])
        ax.set_xlim([1,(self.rMax/au)*1.25])
        plt.title("Angular scale height  "+r'$h$'+" (Au)")
        plt.xlabel('Radius (Au)')
        plt.ylabel('h (Au)')
        plt.tight_layout()
        plt.savefig(file+"_lineal_hight.png")
        plt.close(graf)


    # 15. Método plotLinealTemperature
    # Muestra el gráfico de la temperatura en el plano medio del disco (ya que algunas temperaturas obtenidas suelen ser 0 en el plano medio)
    def plotLinealTemperature(self,file):

        # Se obtiene las temperaturas del plano medio (las de los ultimos 5 ángulos mas cercanos al eje horizontal).
        analisis = self.temp[:,-5:,0,:]
        temp5, temp4, temp3, temp2, temp1 = [], [], [], [], []
        for i in analisis:
            temp5.append(i[0])
            temp4.append(i[1])
            temp3.append(i[2])
            temp2.append(i[3])
            temp1.append(i[4])
        
        # Se obtienen esos ultimos 3 ángulos para distingirlos en la leyenda del gráfico.
        thets = self.tt[0,-5:]*180/np.pi

        # Se realiza el gráfico.
        size = (10,7.5) if self.nspec == 1 else (7.5,12)
        graf, ax = plt.subplots(nrows=self.nspec,ncols=1,figsize=size)
        graf.suptitle("$R_{c}:$ %.2f au    $M_{Disk}:$ %.5f ms    $\gamma:$ %.2f    $H_{100}:$ %.2f au    $\psi:$ %.2f"%(self.Rc/au,self.mDisk/ms,self.gamma,self.H100/au,self.psi))

        if (self.nspec == 1):
            np.save(file+"_tempMiddlePlane.npy",temp1)
            ax.plot(self.ri[:-1]/au,temp5,linewidth=2,label=r'$\theta$'+": %.2f"%thets[0])
            ax.plot(self.ri[:-1]/au,temp4,linewidth=2,label=r'$\theta$'+": %.2f"%thets[1])
            ax.plot(self.ri[:-1]/au,temp3,linewidth=2,label=r'$\theta$'+": %.2f"%thets[2])
            ax.plot(self.ri[:-1]/au,temp2,linewidth=2,label=r'$\theta$'+": %.2f"%thets[3])
            ax.plot(self.ri[:-1]/au,temp1,linewidth=2,label=r'$\theta$'+": %.2f"%thets[4])
            plt.title("Temperature Midle Plane")
            plt.xlabel('Radius (Au)')
            plt.ylabel('Temperature (°K)')
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_ylim([10,10e2])
            ax.set_xlim([1,(self.rMax/au)*1.25])
            plt.legend()
        
        else:
            for i in range(self.nspec):
                if (i == 0):
                    ax[i].set_title("Temperature Midle Plane"+" small dust")
                    ax[i].plot(self.ri[:-1]/au,np.array(temp5)[:,0],linewidth=2,label=r'$\theta$'+": %.2f"%thets[0])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp4)[:,0],linewidth=2,label=r'$\theta$'+": %.2f"%thets[1])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp3)[:,0],linewidth=2,label=r'$\theta$'+": %.2f"%thets[2])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp2)[:,0],linewidth=2,label=r'$\theta$'+": %.2f"%thets[3])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp1)[:,0],linewidth=2,label=r'$\theta$'+": %.2f"%thets[4])
                else:
                    ax[i].set_title("Temperature Midle Plane"+" large dust")
                    ax[i].plot(self.ri[:-1]/au,np.array(temp5)[:,1],linewidth=2,label=r'$\theta$'+": %.2f"%thets[0])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp4)[:,1],linewidth=2,label=r'$\theta$'+": %.2f"%thets[1])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp3)[:,1],linewidth=2,label=r'$\theta$'+": %.2f"%thets[2])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp2)[:,1],linewidth=2,label=r'$\theta$'+": %.2f"%thets[3])
                    ax[i].plot(self.ri[:-1]/au,np.array(temp1)[:,1],linewidth=2,label=r'$\theta$'+": %.2f"%thets[4])

                ax[i].set_xlabel('Radius (Au)')
                ax[i].set_ylabel('Temperature (°K)')
                ax[i].set_xscale("log")
                ax[i].set_yscale("log")
                ax[i].set_ylim([10,10e2])
                ax[i].set_xlim([1,(self.rMax/au)*1.25])
                ax[i].legend()

        plt.tight_layout()
        plt.savefig(file+"_lineal_temperature.png")
        plt.close(graf)
    

    # 16. Método generateSkyModel
    # Ejecuta el código de RADMC3D image y exporta la salida de un .out a un .fits.
    def generateSkyModel(self,output,name,ident):

        #print("\n--------------------------------------------------------------------------------------------")
        #print("                              GENERATING IMAGE of DISK                                        ")
        #print("--------------------------------------------------------------------------------------------\n")

        # Crea la imagen en salida .out con el comando radmc3d image.
        for i in range(len(self.incl)):
            for lam in self.lamb:
                os.system("radmc3d image npix "+str(self.npix)+
                        " lambda "+str(lam)+
                        " incl "+str(self.incl[i])+
                        " posang "+str(self.posang[i])+
                        " sizeau "+str(self.sizeau)+
                        " nostar"+" > output.txt")
                im = image.readImage(fname="image.out")

                # Analisis disk.
                if (output == 1):
                    im.writeFits(fname="image.fits",dpc=self.dpc)
                    #plt.imsave(name+"-"+str(i+1)+"-"+str(lam)+".png",np.flip(fits.open(name+"-"+str(i+1)+"-"+str(lam)+".fits")[0].data[0],axis=0),cmap = "hot")

                # Data base disk
                else:
                    im.writeFits(fname=str(ident+i)+"-disk-"+str(lam)+".fits",dpc=self.dpc)
                    plt.imsave(str(ident+i)+"-disk-"+str(lam)+".png",np.flip(fits.open(str(ident+i)+"-disk-"+str(lam)+".fits")[0].data[0],axis=0),cmap = "hot")
        

    # 17. Método writeResumen
    # Genera un archivo de texto estructurado con la densidad y temperatura en cada punto de la grilla.
    def writeResumen(self,file):
        
        # Se abre el archivo para escribirlo.
        with open(file,'w+') as f:

            # Resumen de los resultados.
            if (self.nspec == 1):
                f.write("|------------------------------------------------------------------------------------------------------------|\n")
                f.write("|------------------------------------------------------------------------------------------------------------|\n")
                f.write("|\t Radius (au) \t|\t Theta (°) \t | \t Density (g/cm3) \t | \t Temperature (°K) \t |\n")
                f.write("|------------------------------------------------------------------------------------------------------------|\n")
                for i in range(self.nr):
                    for j in range(self.ntheta):
                        f.write("|\t %4.4f \t|\t %2.2f \t\t | \t %13.6e \t | \t %3.3f \t\t |\n"%(self.rr[i,j]/au,self.tt[i,j]*180/np.pi,self.rhos[0,i,j,0],self.temp[i,j,0,0]))
                    f.write("|------------------------------------------------------------------------------------------------------------|\n")

                # Cierre del archivo.    
                f.close()

            else:
                f.write("|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n")
                f.write("|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n")
                f.write("|\t Radius (au) \t|\t Theta (°) \t | \t Density Small-Dust (g/cm3) \t | \t Temperature Small-Dust (°K) \t | \t Density Large-Dust (g/cm3) \t | \t Temperature Large-Dust (°K) \t |\n")
                f.write("|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n")
                for i in range(self.nr):
                    for j in range(self.ntheta):
                        f.write("|\t %4.4f \t|\t %2.2f \t\t | \t %13.6e \t | \t %3.3f \t\t | \t %13.6e \t | \t %3.3f \t\t |\n"%(self.rr[i,j]/au,self.tt[i,j]*180/np.pi,self.rhos[0,i,j,0],self.temp[i,j,0,0],self.rhos[1,i,j,0],self.temp[i,j,0,1]))
                    f.write("|----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|\n")

                # Cierre del archivo.    
                f.close()


    # 18. Método writeDisk
    # Guarda los datos de los discos generados para la database.
    def writeDisk(self,ident):

        # Verificar si el archivo ya existe
        nombre = "skymodels.csv"
        arcExis = os.path.exists(nombre)

        # Abrir el archivo CSV en modo de escritura
        with open(nombre,"a",newline="") as archivo:
            writer = csv.writer(archivo)
            if not arcExis:
                writer.writerow(["disk","mstar","rstar","tstar","Rc","mDisk","gamma","H100","psi","incl","posang"])
            
            for i in range(len(self.incl)):
                writer.writerow([str(ident+i)+"-disk",
                                 str(round(self.mstar/ms,5)),
                                 str(round(self.rstar/rs,1)),
                                 str(self.tstar),
                                 str(round(self.Rc/au,1)),
                                 str(round(self.mDisk/ms,5)),
                                 str(round(self.gamma,2)),
                                 str(round(self.H100/au,1)),
                                 str(round(self.psi,2)),
                                 str(self.incl[i]),
                                 str(self.posang[i])])

