#### How to Run

Type: 
  - python3 generate_fits_files --dats-path "path/to/dats/files"

When you run the command you will see a new folder called "data" which will contain all the diferents 
disks simulated in two different wavelengths (740.228 y 2067 microns). The idea is to take the resultant
.fits files of each band for each disk and rename them with the set of parameters in the name, ej:
mdisk_rc_gamma_psi_H100_B4.fits, but its important to differenciate between bands to put the name B4 or B8 
at the end. 

This data will be used to train the Swin-Regression-Transformer, model that you can find in a public repository
published by me. This model has been made for estimate the five parameters at the end of the trainning. You have to ensure 
that all the images generated can be visualized (for this use ds9).
