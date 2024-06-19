import matplotlib.pylab as pl
from astropy.io import fits
import numpy as np

### READ IN FITS FILE
cube_sigma_file='/Users/ben/Programs/IDL/3Dplotting/benpy/ps23n24/' +\
  'ps23n24_interpolated_cube_sigma.fits'
	# ABOVE IS A VMC MAP PROCESSED VIA MY PYTHON CODES
	#  BRIAN'S IDL VERSION IS SIMILAR BUT NOT IDENTICAL, SO BELOW MAY CHANGE A BIT
ff = fits.open(cube_sigma_file)
dens_all = ff[0].data
gridx0   = ff[1].data
gridy0   = ff[2].data
gridz0   = ff[3].data
z1h_all  = ff[4].data
z2h_all  = ff[5].data
mean_dens_d_fromlog = ff[6].data
x0_log_od = ff[7].data
sigma_log_od = ff[8].data
eyfit_x0log_od2 = ff[9].data
eyfit_sigmalog_od2 = ff[10].data
eyfit_x0log_od5 = ff[11].data
eyfit_sigmalog_od5 = ff[12].data  
nx=len(gridx0)
ny=len(gridy0)
nz=len(gridz0)
od = []
sigma_od = []
for i in range(len(gridz0)):
  od.append( np.log10(dens_all[i]/mean_dens_d_fromlog[i]) )
  sigma_od.append( (od[i]-eyfit_x0log_od5[i]) / eyfit_sigmalog_od5[i] )
od = np.array(od)
sigma_od = np.array(sigma_od)


#---------------------------------------------------
def vmc2uni(xi,yi,zi=0,vmc_file=cube_sigma_file):
  """ CONVERT VMC COORDINATES TO RA, DEC (Z OPTIONAL) """
  ff = fits.open(vmc_file)
  gridx0 = ff[1].data
  gridy0 = ff[2].data
  gridz0 = ff[3].data
  ra_vmc = gridx0[xi]
  dec_vmc = gridy0[yi]
  if zi==0:
    return ra_vmc, dec_vmc
  elif zi>0:
    z_vmc = gridz0[zi]
    return ra_vmc, dec_vmc, z_vmc
 

#---------------------------------------------------
def uni2vmc(ra,dec,z=0,vmc_file=cube_sigma_file):
  """ CONVERT RA, DEC (Z OPTIONAL) TO VMC COORDINATES """
  ff = fits.open(vmc_file)
  gridx0 = ff[1].data
  gridy0 = ff[2].data
  gridz0 = ff[3].data
  xi = np.interp( ra, gridx0[np.argsort(gridx0)], np.arange(len(gridx0))[np.argsort(gridx0)] )  
  yi = np.interp( dec, gridy0[np.argsort(gridy0)], np.arange(len(gridy0))[np.argsort(gridy0)] )  
  if z==0:
    return xi, yi
  elif z>0:
    zi = np.interp( z, gridz0[np.argsort(gridz0)], np.arange(len(gridz0))[np.argsort(gridz0)] )  
    return xi, yi, zi
