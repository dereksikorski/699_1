### CONVERT APERTURE MAGNITUDES FROM COSMOS2020 TO TOTAL MAGNITUDES
from astropy.io import fits
import matplotlib.pylab as pl
import numpy as np


def get_total_mags_C20(filename='/Users/ben/Programs/python/temp/cosmos2020/COSMOS2020_CLASSIC_R1_v2.0.fits', aperture=3):

  with fits.open(filename) as f:
    data = f[1].data

  filtnames = ['CFHT_ustar', 'CFHT_u', 'HSC_g', 'HSC_r', 'HSC_i', 'HSC_z', 'HSC_y',\
    'UVISTA_Y', 'UVISTA_J', 'UVISTA_H', 'UVISTA_Ks', 'SC_IB427', 'SC_IB464', 'SC_IA484',\
    'SC_IB505', 'SC_IA527', 'SC_IB574', 'SC_IA624', 'SC_IA679', 'SC_IB709', 'SC_IA738',\
    'SC_IA767', 'SC_IB827', 'SC_NB711', 'SC_NB816', 'UVISTA_NB118', 'SC_B', 'SC_V',\
    'SC_rp', 'SC_ip', 'SC_zpp', 'IRAC_CH1', 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']
  AlambdaDivEBV = [4.674, 4.807, 3.69, 2.715, 2.0, 1.515, 1.298,\
    1.213, 0.874, 0.565, 0.365, 4.261, 3.844, 3.622,\
    3.425, 3.265, 2.938, 2.694, 2.431, 2.29, 2.151,\
    1.997, 1.748, 2.268, 1.787, 0.946, 4.041, 3.128,\
    2.673, 2.003, 1.466, 0.163, 0.112, 8.31, 8.742]

  aperture = aperture # ["]
  offset = data['total_off'+str(aperture)]
  ebv    = data['EBV_MW']
  names_noaper = ['IRAC_CH1', 'IRAC_CH2', 'GALEX_FUV', 'GALEX_NUV']

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

  return data
