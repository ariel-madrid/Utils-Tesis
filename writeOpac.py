
def writeOpac(nspec):

    # Escritura del archivo dustopac.inp.
    #print(msg)
    with open('dustopac.inp','w+') as f:

        f.write('2               Format number of this file\n')
        # Dependiendo del número de especies que se ingrese varía el kappa a utilizar.
        if nspec == 1:
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

if __name__ == '__main__':
    nspec = 1
    writeOpac(nspec)