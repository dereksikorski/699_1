### CONVERT APERTURE MAGNITUDES FROM COSMOS2020 TO TOTAL MAGNITUDES
from astropy.io import fits
import matplotlib.pylab as pl
import numpy as np
from GalPop import *


def get_total_mags_C20(filename=r"C:/Users/sikor/OneDrive/Desktop/BigData/COSMOS2020/COSMOS2020_CLASSIC_R1_v2.0.fits", aperture=3):
  save_dict = {}
  with fits.open(filename) as f:
    data = f[1].data

  filtnames = ['CFHT_ustar', 'CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y',\
    'UVISTA_Y', 'UVISTA_J', 'UVISTA_H', 'UVISTA_Ks', 'SC_IB427', 'SC_IB464', 'SC_IA484',\
    'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 'SC_IA679', 'SC_IB709', 'SC_IA738',\
    'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816', 'UVISTA_NB118', 'SC_B', 'SC_V',\
    'SC_rp', 'SC_ip', 'SC_zpp', 'IRAC_CH1', 'IRAC_CH2', 'SPLASH_CH3', 'SPLASH_CH4', 'GALEX_FUV', 'GALEX_NUV']
  AlambdaDivEBV = [4.674, 4.807, 3.69, 2.715, 2.0, 1.515, 1.298,\
    1.213, 0.874, 0.565, 0.365, 4.261, 3.844, 3.622,\
    3.425, 3.265, 2.938, 2.694, 2.431, 2.29, 2.151,\
    1.997, 1.748, 2.268, 1.787, 0.946, 4.041, 3.128,\
    2.673, 2.003, 1.466, 0.163, 0.112, 0.075, 0.045, 8.31, 8.742]

  aperture = aperture # ["]
  offset = data['total_off'+str(aperture)]
  ebv    = data['EBV_MW']
  names_noaper = ['SPLASH_CH3', 'SPLASH_CH4', 'IRAC_CH1', 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']

  for i,name in enumerate(filtnames):
    if name not in names_noaper:
        str_flux     = name+'_FLUX_APER'   +str(aperture)
        str_flux_err = name+'_FLUXERR_APER'+str(aperture)
        str_mag      = name+'_MAG_APER'    +str(aperture)
        str_mag_err  = name+'_MAGERR_APER' +str(aperture)
    else:
        str_flux     = name+'_FLUX'
        str_flux_err = name+'_FLUXERR'
        str_mag      = name+'_MAG'
        str_mag_err  = name+'_MAGERR'    
    flux     = data[str_flux]
    flux_err = data[str_flux_err]
    mag      = data[str_mag]
    mag_err  = data[str_mag_err]    

    # apply aperture-to-total offset
    if name not in names_noaper:
        idx = (flux>0)
        flux[idx]     *= 10**(-0.4*offset[idx])
        flux_err[idx] *= 10**(-0.4*offset[idx])
        mag[idx]      += offset[idx]    

    # correct for Milky Way attenuation
    idx = (flux>0)
    atten = 10**(0.4*AlambdaDivEBV[i]*ebv[idx])
    flux[idx]     *= atten
    flux_err[idx] *= atten
    mag[idx]      += -2.5*np.log10(atten)    

    data[str_flux]     = flux
    data[str_flux_err] = flux_err
    data[str_mag]      = mag
    data[str_mag_err]  = mag_err


    save_dict[str_flux] = flux*10**-29    # Convert uJy to erg/s/Hz/cm^2
    save_dict[str_flux_err] = flux_err* 10**-29   # Convert uJy to erg/s/Hz/cm^2

  return save_dict



## Get a dictionary of Mags and errors
c_mags = get_total_mags_C20()




## Save the dictionary for future use
  # Make dtypes
dtypes = []
for k in c_mags.keys():
  dtypes.append((k, np.float32))  

  # Make array to save and fill
d_save = np.full(shape=len(c_mags['CFHT_ustar_FLUX_APER3']), fill_value=0, dtype=dtypes)

for k in c_mags.keys():
  new_col = c_mags[k]
  new_col = np.nan_to_num(new_col, nan=-99)

  d_save[k] = new_col

np.save(r"C:/Users/sikor/OneDrive/Desktop/BigData/COSMOS2020/C20_Fluxes.npy", d_save)

# -----------------------------------------------------------------------
# -----------------------------------------------------------------------


## Make new object
p_data = GalPop.loadFile("GalPops_phot/c20p.npy")
p_data.verbose=False


c_mags = np.load(r"C:/Users/sikor/OneDrive/Desktop/BigData/COSMOS2020/C20_Fluxes.npy", allow_pickle=True)
c_mags = c_mags[p_data.IDs.astype(int) - 1]
c_names = c_mags.dtype.names

for n in c_names:

    p_data.addMag(n, c_mags[n])


p_data.saveFile("c20p_fluxes.npy")
# cat = p_data.create_LP_cat()
