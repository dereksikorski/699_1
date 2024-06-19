import matplotlib.pylab as pl
from astropy.io import fits
from numpy.random import random, randint
from VMC_conv import *

#from vmc_density import *
vmc_cube = '/Users/ben/Programs/IDL/3Dplotting/benpy/ps23n24/ps23n24_interpolated_cube_sigma.fits'
from get_total_mags_C20 import *


### READ IN LEPHARE RUNS
lp_out = '/Users/ben/Programs/LePhare/LePHARE/lephare_dev/test/C20_hiz_MCz/outputs_z'
lp300_1 = np.loadtxt(lp_out + '3.0/MC_z3.0_set01.out')
lp300_2 = np.loadtxt(lp_out + '3.0/MC_z3.0_set02.out')
lp300_3 = np.loadtxt(lp_out + '3.0/MC_z3.0_set03.out')
lp300_4 = np.loadtxt(lp_out + '3.0/MC_z3.0_set04.out')
lp300_5 = np.loadtxt(lp_out + '3.0/MC_z3.0_set05.out')
lp300_6 = np.loadtxt(lp_out + '3.0/MC_z3.0_set06.out')

lp310_1 = np.loadtxt(lp_out + '3.1/MC_z3.1_set01.out')
lp310_2 = np.loadtxt(lp_out + '3.1/MC_z3.1_set02.out')
lp310_3 = np.loadtxt(lp_out + '3.1/MC_z3.1_set03.out')
lp310_4 = np.loadtxt(lp_out + '3.1/MC_z3.1_set04.out')
lp310_5 = np.loadtxt(lp_out + '3.1/MC_z3.1_set05.out')
lp310_6 = np.loadtxt(lp_out + '3.1/MC_z3.1_set06.out')

lp320_1 = np.loadtxt(lp_out + '3.2/MC_z3.2_set01.out')
lp320_2 = np.loadtxt(lp_out + '3.2/MC_z3.2_set02.out')
lp320_3 = np.loadtxt(lp_out + '3.2/MC_z3.2_set03.out')
lp320_4 = np.loadtxt(lp_out + '3.2/MC_z3.2_set04.out')
lp320_5 = np.loadtxt(lp_out + '3.2/MC_z3.2_set05.out')
lp320_6 = np.loadtxt(lp_out + '3.2/MC_z3.2_set06.out')

lp330_1 = np.loadtxt(lp_out + '3.3/MC_z3.3_set01.out')
lp330_2 = np.loadtxt(lp_out + '3.3/MC_z3.3_set02.out')
lp330_3 = np.loadtxt(lp_out + '3.3/MC_z3.3_set03.out')
lp330_4 = np.loadtxt(lp_out + '3.3/MC_z3.3_set04.out')
lp330_5 = np.loadtxt(lp_out + '3.3/MC_z3.3_set05.out')
lp330_6 = np.loadtxt(lp_out + '3.3/MC_z3.3_set06.out')

lp340_1 = np.loadtxt(lp_out + '3.4/MC_z3.4_set01.out')
lp340_2 = np.loadtxt(lp_out + '3.4/MC_z3.4_set02.out')
lp340_3 = np.loadtxt(lp_out + '3.4/MC_z3.4_set03.out')
lp340_4 = np.loadtxt(lp_out + '3.4/MC_z3.4_set04.out')
lp340_5 = np.loadtxt(lp_out + '3.4/MC_z3.4_set05.out')
lp340_6 = np.loadtxt(lp_out + '3.4/MC_z3.4_set06.out')

lp350_1 = np.loadtxt(lp_out + '3.5/MC_z3.5_set01.out')
lp350_2 = np.loadtxt(lp_out + '3.5/MC_z3.5_set02.out')
lp350_3 = np.loadtxt(lp_out + '3.5/MC_z3.5_set03.out')
lp350_4 = np.loadtxt(lp_out + '3.5/MC_z3.5_set04.out')
lp350_5 = np.loadtxt(lp_out + '3.5/MC_z3.5_set05.out')
lp350_6 = np.loadtxt(lp_out + '3.5/MC_z3.5_set06.out')

lp360_1 = np.loadtxt(lp_out + '3.6/MC_z3.6_set01.out')
lp360_2 = np.loadtxt(lp_out + '3.6/MC_z3.6_set02.out')
lp360_3 = np.loadtxt(lp_out + '3.6/MC_z3.6_set03.out')
lp360_4 = np.loadtxt(lp_out + '3.6/MC_z3.6_set04.out')
lp360_5 = np.loadtxt(lp_out + '3.6/MC_z3.6_set05.out')
lp360_6 = np.loadtxt(lp_out + '3.6/MC_z3.6_set06.out')

lp370_1 = np.loadtxt(lp_out + '3.7/MC_z3.7_set01.out')
lp370_2 = np.loadtxt(lp_out + '3.7/MC_z3.7_set02.out')
lp370_3 = np.loadtxt(lp_out + '3.7/MC_z3.7_set03.out')
lp370_4 = np.loadtxt(lp_out + '3.7/MC_z3.7_set04.out')
lp370_5 = np.loadtxt(lp_out + '3.7/MC_z3.7_set05.out')
lp370_6 = np.loadtxt(lp_out + '3.7/MC_z3.7_set06.out')


lp305_1 = np.loadtxt(lp_out + '3.05/MC_z3.05_set01.out')
lp305_2 = np.loadtxt(lp_out + '3.05/MC_z3.05_set02.out')
lp305_3 = np.loadtxt(lp_out + '3.05/MC_z3.05_set03.out')
lp305_4 = np.loadtxt(lp_out + '3.05/MC_z3.05_set04.out')
lp305_5 = np.loadtxt(lp_out + '3.05/MC_z3.05_set05.out')
lp305_6 = np.loadtxt(lp_out + '3.05/MC_z3.05_set06.out')

lp315_1 = np.loadtxt(lp_out + '3.15/MC_z3.15_set01.out')
lp315_2 = np.loadtxt(lp_out + '3.15/MC_z3.15_set02.out')
lp315_3 = np.loadtxt(lp_out + '3.15/MC_z3.15_set03.out')
lp315_4 = np.loadtxt(lp_out + '3.15/MC_z3.15_set04.out')
lp315_5 = np.loadtxt(lp_out + '3.15/MC_z3.15_set05.out')
lp315_6 = np.loadtxt(lp_out + '3.15/MC_z3.15_set06.out')

lp325_1 = np.loadtxt(lp_out + '3.25/MC_z3.25_set01.out')
lp325_2 = np.loadtxt(lp_out + '3.25/MC_z3.25_set02.out')
lp325_3 = np.loadtxt(lp_out + '3.25/MC_z3.25_set03.out')
lp325_4 = np.loadtxt(lp_out + '3.25/MC_z3.25_set04.out')
lp325_5 = np.loadtxt(lp_out + '3.25/MC_z3.25_set05.out')
lp325_6 = np.loadtxt(lp_out + '3.25/MC_z3.25_set06.out')

lp335_1 = np.loadtxt(lp_out + '3.35/MC_z3.35_set01.out')
lp335_2 = np.loadtxt(lp_out + '3.35/MC_z3.35_set02.out')
lp335_3 = np.loadtxt(lp_out + '3.35/MC_z3.35_set03.out')
lp335_4 = np.loadtxt(lp_out + '3.35/MC_z3.35_set04.out')
lp335_5 = np.loadtxt(lp_out + '3.35/MC_z3.35_set05.out')
lp335_6 = np.loadtxt(lp_out + '3.35/MC_z3.35_set06.out')

lp345_1 = np.loadtxt(lp_out + '3.45/MC_z3.45_set01.out')
lp345_2 = np.loadtxt(lp_out + '3.45/MC_z3.45_set02.out')
lp345_3 = np.loadtxt(lp_out + '3.45/MC_z3.45_set03.out')
lp345_4 = np.loadtxt(lp_out + '3.45/MC_z3.45_set04.out')
lp345_5 = np.loadtxt(lp_out + '3.45/MC_z3.45_set05.out')
lp345_6 = np.loadtxt(lp_out + '3.45/MC_z3.45_set06.out')

lp355_1 = np.loadtxt(lp_out + '3.55/MC_z3.55_set01.out')
lp355_2 = np.loadtxt(lp_out + '3.55/MC_z3.55_set02.out')
lp355_3 = np.loadtxt(lp_out + '3.55/MC_z3.55_set03.out')
lp355_4 = np.loadtxt(lp_out + '3.55/MC_z3.55_set04.out')
lp355_5 = np.loadtxt(lp_out + '3.55/MC_z3.55_set05.out')
lp355_6 = np.loadtxt(lp_out + '3.55/MC_z3.55_set06.out')

lp365_1 = np.loadtxt(lp_out + '3.65/MC_z3.65_set01.out')
lp365_2 = np.loadtxt(lp_out + '3.65/MC_z3.65_set02.out')
lp365_3 = np.loadtxt(lp_out + '3.65/MC_z3.65_set03.out')
lp365_4 = np.loadtxt(lp_out + '3.65/MC_z3.65_set04.out')
lp365_5 = np.loadtxt(lp_out + '3.65/MC_z3.65_set05.out')
lp365_6 = np.loadtxt(lp_out + '3.65/MC_z3.65_set06.out')



### READ IN COSMOS2020 CATALOG AND P(Z), FIND 'HIGH-REDSHIFT' WITH LEPHARE RUNS
c20 = get_total_mags_C20()
lppzfile = '/Users/ben/Programs/python/temp/cosmos2020/PZ/COSMOS2020_CLASSIC_R1_v2.0_LEPHARE_PZ.fits'
lp_pz = fits.getdata(lppzfile)
wg = np.where( (c20['lp_zPDF_u68']>2.5) )[0]
id_c20 = c20['ID'][wg]
ra_c20 = c20['ALPHA_J2000'][wg]
dec_c20 = c20['DELTA_J2000'][wg]
zp_c20 = c20['lp_zPDF'][wg]


### READ IN SPEC. CONFIRMED CATALOG, GET GOOD SPEC-Z'S
specfile = '/Users/ben/Programs/python/py_speccat/master_specz_COSMOS_BF_v3_t1.cat'
sid_c20, ras, decs, zss, qfs, masss, sfrs, ages, Mnuvs, Mus, Mrs, Mvs, Mjs = np.loadtxt(specfile,\
  unpack=True, usecols = (0,3,5,11,13,15,18,21,24,25,26,27,28) )


### INITIALIZE FILES
niter=100
niter_print = np.arange(niter)
file_base = '/Users/ben/Programs/LePhare/LePHARE/lephare_dev/test/C20_hiz_MCz/' +\
  'zupGT2p5_comp_3z3p7_d05/'
for i in niter_print:
  file_i = open(file_base + 'iter' + str(i).zfill(4) + '.dat', 'w')
  file_i.write('#  ID_C20   R.A.            Dec.       z_best    spec?   z_MC    mass    sfr       age         Mu      Mv      Mj     Mnuv    Mr       Sigma_g  log(1+delta_g)  z_<-\n') 
  file_i.close()


### FOR EACH GALAXY IN 'HIGH-REDSHIFT' SAMPLE
z_lolim = 3.0
z_hilim = 3.7

### IF SPEC. CONF:
for i in range(len(sid_c20)):

	### CHECK WHETHER TO USE SPEC-Z VALUES OR NOT
  if qfs[i]>0:
    if int(qfs[i])%10==2:
      qfi = 2 
      f_specz = 0.7
    elif int(qfs[i])%10==9:
      qfi = 9 
      f_specz = 0.7
    elif int(qfs[i])%10==3:
      qfi = 3 
      f_specz = 0.993
    elif int(qfs[i])%10==4: 
      qfi = 4
      f_specz = 0.993
    else: f_specz = 0
  else: f_specz = 0
  i_idc20 = sid_c20[i]
  i_ra    = ras[i]
  i_dec   = decs[i]
  i_zbest = zss[i]
  i_z     = []
  i_mass  = []
  i_sfr   = []
  i_age   = []
  i_Mu    = []
  i_Mr    = []
  i_Mv    = []
  i_Mj    = []
  i_Mnuv  = []
  i_sig   = []
  i_od    = []
  i_zod   = []

  ft = random(size=niter)
  mc_z = monte_carlo_z(lp_pz[id_c20[i]][1:], niter=niter)

  for j in range(niter):		# FOR EACH ITERATION

    if ft[j]<f_specz:		# USE SPEC-Z
      if ((zss[i]>z_lolim) & (zss[i]<z_hilim)):	# IN CUBE RANGE
        i_z.append(    zss[i]  )
        i_mass.append( masss[i])
        i_sfr.append(  sfrs[i] )
        i_age.append(  ages[i] )
        i_Mu.append(   Mus[i]  )
        i_Mr.append(   Mrs[i]  )
        i_Mv.append(   Mvs[i]  )
        i_Mj.append(   Mjs[i]  )
        i_Mnuv.append( Mnuvs[i])
        vmccoords = uni2vmc( i_ra, i_dec, i_z[j] )
        xx = int(vmccoords[0])
        yy = int(vmccoords[1])
        zz = int(vmccoords[2])
        i_sig.append(  dens_all[zz,yy,xx] )
        i_od.append(   np.log10(dens_all[zz,yy,xx]/mean_dens_d_fromlog[zz]) )
        i_zod.append(  (i_od[j]-eyfit_x0log_od5[zz]) / eyfit_sigmalog_od5[zz] )
      else:
        i_z.append(    zss[i]  )
        i_mass.append( masss[i])
        i_sfr.append(  sfrs[i] )
        i_age.append(  ages[i] )
        i_Mu.append(   Mus[i]  )
        i_Mr.append(   Mrs[i]  )
        i_Mv.append(   Mvs[i]  )
        i_Mj.append(   Mjs[i]  )
        i_Mnuv.append( Mnuvs[i])
        i_sig.append(  -99 )
        i_od.append(   -99 )
        i_zod.append(   -99 )

    elif ft[j]>=f_specz:		# DRAW FROM P(Z)
      i_z.append(    mc_z[j] )
      set = get_set_lprun(id_c20[i])
      zmc = get_zmc_lprun(mc_z[j])

      if zmc==-99:			### IF NOT IN REDSHIFT RANGE OF CUBE
        i_mass.append( -99 )
        i_sfr.append(  -99 )
        i_age.append(  -99 )
        i_Mu.append(   -99 )
        i_Mv.append(   -99 )
        i_Mj.append(   -99 )
        i_Mnuv.append( -99 )
        i_Mr.append(   -99 )
        i_sig.append(  -99 )
        i_od.append(   -99 )
        i_zod.append(  -99 )

      else:				### GET PROPER DATASET
        if zmc==3.0:
          if set==1:   lp_data = lp300_1
          elif set==2: lp_data = lp300_2
          elif set==3: lp_data = lp300_3
          elif set==4: lp_data = lp300_4
          elif set==5: lp_data = lp300_5
          elif set==6: lp_data = lp300_6
        elif zmc==3.05:
          if set==1:   lp_data = lp305_1
          elif set==2: lp_data = lp305_2
          elif set==3: lp_data = lp305_3
          elif set==4: lp_data = lp305_4
          elif set==5: lp_data = lp305_5
          elif set==6: lp_data = lp305_6
  
        elif zmc==3.1:
          if set==1:   lp_data = lp310_1
          elif set==2: lp_data = lp310_2
          elif set==3: lp_data = lp310_3
          elif set==4: lp_data = lp310_4
          elif set==5: lp_data = lp310_5
          elif set==6: lp_data = lp310_6
        elif zmc==3.15:
          if set==1:   lp_data = lp315_1
          elif set==2: lp_data = lp315_2
          elif set==3: lp_data = lp315_3
          elif set==4: lp_data = lp315_4
          elif set==5: lp_data = lp315_5
          elif set==6: lp_data = lp315_6
  
        elif zmc==3.2:
          if set==1:   lp_data = lp320_1
          elif set==2: lp_data = lp320_2
          elif set==3: lp_data = lp320_3
          elif set==4: lp_data = lp320_4
          elif set==5: lp_data = lp320_5
          elif set==6: lp_data = lp320_6
        elif zmc==3.25:
          if set==1:   lp_data = lp325_1
          elif set==2: lp_data = lp325_2
          elif set==3: lp_data = lp325_3
          elif set==4: lp_data = lp325_4
          elif set==5: lp_data = lp325_5
          elif set==6: lp_data = lp325_6
  
        elif zmc==3.3:
          if set==1:   lp_data = lp330_1
          elif set==2: lp_data = lp330_2
          elif set==3: lp_data = lp330_3
          elif set==4: lp_data = lp330_4
          elif set==5: lp_data = lp330_5
          elif set==6: lp_data = lp330_6
        elif zmc==3.35:
          if set==1:   lp_data = lp335_1
          elif set==2: lp_data = lp335_2
          elif set==3: lp_data = lp335_3
          elif set==4: lp_data = lp335_4
          elif set==5: lp_data = lp335_5
          elif set==6: lp_data = lp335_6
  
        elif zmc==3.4:
          if set==1:   lp_data = lp340_1
          elif set==2: lp_data = lp340_2
          elif set==3: lp_data = lp340_3
          elif set==4: lp_data = lp340_4
          elif set==5: lp_data = lp340_5
          elif set==6: lp_data = lp340_6
        elif zmc==3.45:
          if set==1:   lp_data = lp345_1
          elif set==2: lp_data = lp345_2
          elif set==3: lp_data = lp345_3
          elif set==4: lp_data = lp345_4
          elif set==5: lp_data = lp345_5
          elif set==6: lp_data = lp345_6
  
        elif zmc==3.5:
          if set==1:   lp_data = lp350_1
          elif set==2: lp_data = lp350_2
          elif set==3: lp_data = lp350_3
          elif set==4: lp_data = lp350_4
          elif set==5: lp_data = lp350_5
          elif set==6: lp_data = lp350_6
        elif zmc==3.55:
          if set==1:   lp_data = lp355_1
          elif set==2: lp_data = lp355_2
          elif set==3: lp_data = lp355_3
          elif set==4: lp_data = lp355_4
          elif set==5: lp_data = lp355_5
          elif set==6: lp_data = lp355_6
  
        elif zmc==3.6:
          if set==1:   lp_data = lp360_1
          elif set==2: lp_data = lp360_2
          elif set==3: lp_data = lp360_3
          elif set==4: lp_data = lp360_4
          elif set==5: lp_data = lp360_5
          elif set==6: lp_data = lp360_6
        elif zmc==3.65:
          if set==1:   lp_data = lp365_1
          elif set==2: lp_data = lp365_2
          elif set==3: lp_data = lp365_3
          elif set==4: lp_data = lp365_4
          elif set==5: lp_data = lp365_5
          elif set==6: lp_data = lp365_6
  
        elif zmc==3.7:
          if set==1:   lp_data = lp370_1
          elif set==2: lp_data = lp370_2
          elif set==3: lp_data = lp370_3
          elif set==4: lp_data = lp370_4
          elif set==5: lp_data = lp370_5
          elif set==6: lp_data = lp370_6
  
        id_mc = lp_data[:,0].astype(int)
        mass_mc = lp_data[:,47]
        sfr_mc = lp_data[:,50]
        age_mc = lp_data[:,44]
        Mu_mc = lp_data[:,7]	# f2
        Mv_mc = lp_data[:,33]	# f28
        Mj_mc = lp_data[:,14]	# f9
        Mnuv_mc = lp_data[:,40]	# f35
        Mr_mc = lp_data[:,34]	# f29
  
        wi_mc = np.where( id_mc == id_c20[i] )[0]
        i_mass.append( mass_mc[wi_mc] )
        i_sfr.append(  sfr_mc[wi_mc]  )
        i_age.append(  age_mc[wi_mc]  )
        i_Mu.append(   Mu_mc[wi_mc]   )
        i_Mv.append(   Mv_mc[wi_mc]   )
        i_Mj.append(   Mj_mc[wi_mc]   )
        i_Mnuv.append( Mnuv_mc[wi_mc] )
        i_Mr.append(   Mr_mc[wi_mc]   )
        vmccoords = uni2vmc( i_ra, i_dec, i_z[j] )
        xx = int(vmccoords[0])
        yy = int(vmccoords[1])
        zz = int(vmccoords[2])
        i_sig.append(  dens_all[zz,yy,xx] )
        i_od.append(   np.log10(dens_all[zz,yy,xx]/mean_dens_d_fromlog[zz]) )
        i_zod.append(  (i_od[j]-eyfit_x0log_od5[zz]) / eyfit_sigmalog_od5[zz] )


  ### WRITE TO FILE, ONE FOR EACH ITERATION
  for j in niter_print:
    file_ij = open(file_base + 'iter' + str(j).zfill(4) + '.dat', 'a')
    if i_zod[j]>-10:
      file_ij.write( str(int(i_idc20)).rjust(10) + ('%.6f' % i_ra).rjust(13) +\
        ('%.6f' % i_dec).rjust(13) + ('%.4f' % i_zbest).rjust(10) +\
        '   1  '+ ('%.4f' % i_z[j]).rjust(8) +\
        ('%.3f' % i_mass[j]).rjust(8) + ('%.3f' % i_sfr[j]).rjust(8) +\
        ('%.3e' % i_age[j]).rjust(14) + ('%.3f' % i_Mu[j]).rjust(8) +\
        ('%.3f' % i_Mv[j]).rjust(8) + ('%.3f' % i_Mj[j]).rjust(8) +\
        ('%.3f' % i_Mnuv[j]).rjust(8) + ('%.3f' % i_Mr[j]).rjust(8) +\
        ('%.4f' % i_sig[j]).rjust(10) + ('%.4f' % i_od[j]).rjust(10) +\
        ('%.4f' % i_zod[j]).rjust(10) + '\n')
    file_ij.close()

  if i%100==0:
    print( str(i) + ' out of ' + str(len(sid_c20)) + ' (' + '%.2f' % (i/len(sid_c20)*100) + '%) spec. objects completed' )




#for i in range(len(wg)):
#for i in np.arange(len(wg)-300000)+300000:
for i in np.arange(300000)+300000:

  ### DETERMINE IF SPEC CONFIRMED OR NOT
  ws_yn = np.where(sid_c20==id_c20[i])[0]

  ### IF NOT SPEC CONF.:
  if ws_yn.size==0:

    ### RUN niter REDSHIFT SAMPLINGS
    mc_z = monte_carlo_z(lp_pz[id_c20[i]][1:], niter=niter)

    ### COMPILE 10000 MASS, SFR, AGE, COLOR VALUES FROM LEPHARE
    i_mass = []
    i_sfr = []
    i_age = []
    i_Mu = []
    i_Mv = []
    i_Mj = []
    i_Mnuv = []
    i_Mr = []
    set = get_set_lprun(id_c20[i])
    for j in range(niter):
      zj = mc_z[j]
      zmc = get_zmc_lprun(zj)

      if zmc==-99:			### IF NOT IN REDSHIFT RANGE
        i_mass.append( -99 )
        i_sfr.append( -99 )
        i_age.append( -99 )
        i_Mu.append( -99 )
        i_Mv.append( -99 )
        i_Mj.append( -99 )
        i_Mnuv.append( -99 )
        i_Mr.append( -99 )

      else:				### GET PROPER DATASET

        if zmc==3.0:
          if set==1:   lp_data = lp300_1
          elif set==2: lp_data = lp300_2
          elif set==3: lp_data = lp300_3
          elif set==4: lp_data = lp300_4
          elif set==5: lp_data = lp300_5
          elif set==6: lp_data = lp300_6
        elif zmc==3.05:
          if set==1:   lp_data = lp305_1
          elif set==2: lp_data = lp305_2
          elif set==3: lp_data = lp305_3
          elif set==4: lp_data = lp305_4
          elif set==5: lp_data = lp305_5
          elif set==6: lp_data = lp305_6
  
        elif zmc==3.1:
          if set==1:   lp_data = lp310_1
          elif set==2: lp_data = lp310_2
          elif set==3: lp_data = lp310_3
          elif set==4: lp_data = lp310_4
          elif set==5: lp_data = lp310_5
          elif set==6: lp_data = lp310_6
        elif zmc==3.15:
          if set==1:   lp_data = lp315_1
          elif set==2: lp_data = lp315_2
          elif set==3: lp_data = lp315_3
          elif set==4: lp_data = lp315_4
          elif set==5: lp_data = lp315_5
          elif set==6: lp_data = lp315_6
  
        elif zmc==3.2:
          if set==1:   lp_data = lp320_1
          elif set==2: lp_data = lp320_2
          elif set==3: lp_data = lp320_3
          elif set==4: lp_data = lp320_4
          elif set==5: lp_data = lp320_5
          elif set==6: lp_data = lp320_6
        elif zmc==3.25:
          if set==1:   lp_data = lp325_1
          elif set==2: lp_data = lp325_2
          elif set==3: lp_data = lp325_3
          elif set==4: lp_data = lp325_4
          elif set==5: lp_data = lp325_5
          elif set==6: lp_data = lp325_6
  
        elif zmc==3.3:
          if set==1:   lp_data = lp330_1
          elif set==2: lp_data = lp330_2
          elif set==3: lp_data = lp330_3
          elif set==4: lp_data = lp330_4
          elif set==5: lp_data = lp330_5
          elif set==6: lp_data = lp330_6
        elif zmc==3.35:
          if set==1:   lp_data = lp335_1
          elif set==2: lp_data = lp335_2
          elif set==3: lp_data = lp335_3
          elif set==4: lp_data = lp335_4
          elif set==5: lp_data = lp335_5
          elif set==6: lp_data = lp335_6
  
        elif zmc==3.4:
          if set==1:   lp_data = lp340_1
          elif set==2: lp_data = lp340_2
          elif set==3: lp_data = lp340_3
          elif set==4: lp_data = lp340_4
          elif set==5: lp_data = lp340_5
          elif set==6: lp_data = lp340_6
        elif zmc==3.45:
          if set==1:   lp_data = lp345_1
          elif set==2: lp_data = lp345_2
          elif set==3: lp_data = lp345_3
          elif set==4: lp_data = lp345_4
          elif set==5: lp_data = lp345_5
          elif set==6: lp_data = lp345_6
  
        elif zmc==3.5:
          if set==1:   lp_data = lp350_1
          elif set==2: lp_data = lp350_2
          elif set==3: lp_data = lp350_3
          elif set==4: lp_data = lp350_4
          elif set==5: lp_data = lp350_5
          elif set==6: lp_data = lp350_6
        elif zmc==3.55:
          if set==1:   lp_data = lp355_1
          elif set==2: lp_data = lp355_2
          elif set==3: lp_data = lp355_3
          elif set==4: lp_data = lp355_4
          elif set==5: lp_data = lp355_5
          elif set==6: lp_data = lp355_6
  
        elif zmc==3.6:
          if set==1:   lp_data = lp360_1
          elif set==2: lp_data = lp360_2
          elif set==3: lp_data = lp360_3
          elif set==4: lp_data = lp360_4
          elif set==5: lp_data = lp360_5
          elif set==6: lp_data = lp360_6
        elif zmc==3.65:
          if set==1:   lp_data = lp365_1
          elif set==2: lp_data = lp365_2
          elif set==3: lp_data = lp365_3
          elif set==4: lp_data = lp365_4
          elif set==5: lp_data = lp365_5
          elif set==6: lp_data = lp365_6
  
        elif zmc==3.7:
          if set==1:   lp_data = lp370_1
          elif set==2: lp_data = lp370_2
          elif set==3: lp_data = lp370_3
          elif set==4: lp_data = lp370_4
          elif set==5: lp_data = lp370_5
          elif set==6: lp_data = lp370_6
  
        id_mc = lp_data[:,0].astype(int)
        mass_mc = lp_data[:,47]
        sfr_mc = lp_data[:,50]
        age_mc = lp_data[:,44]
        Mu_mc = lp_data[:,7]	# f2
        Mv_mc = lp_data[:,33]	# f28
        Mj_mc = lp_data[:,14]	# f9
        Mnuv_mc = lp_data[:,40]	# f35
        Mr_mc = lp_data[:,34]	# f29
  
        wi_mc = np.where( id_mc == id_c20[i] )[0]
        i_mass.append( mass_mc[wi_mc] )
        i_sfr.append( sfr_mc[wi_mc] )
        i_age.append( age_mc[wi_mc] )
        i_Mu.append( Mu_mc[wi_mc] )
        i_Mv.append( Mv_mc[wi_mc] )
        i_Mj.append( Mj_mc[wi_mc] )
        i_Mnuv.append( Mnuv_mc[wi_mc] )
        i_Mr.append( Mr_mc[wi_mc] )

    ### COMPILE 10000 OVERDENSITY VALUES / MEMBERSHIP    
    i_s, i_od, i_zod = get_density_mc(ra_c20[i], dec_c20[i], mc_z)
 #   i_mem = []		### HOW TO DO THIS? - SAVE FOR LATER
 #   for j in range(len(i_od)):
 #     if (i_zod[j]>3): i_mem.append(1)
 #     else: i_mem.append(0)

    ### WRITE TO FILE, ONE FOR EACH ITERATION
    for j in range(niter):
      file_ij = open(file_base + 'iter' + str(niter_print[j]).zfill(4) + '.dat', 'a')
      file_ij.write( str(id_c20[i]).rjust(10) + ('%.6f' % ra_c20[i]).rjust(13) +\
        ('%.6f' % dec_c20[i]).rjust(13) + ('%.4f' % zp_c20[i]).rjust(10) +\
        '   0  ' + ('%.4f' % mc_z[j]).rjust(8) +\
        ('%.3f' % i_mass[j]).rjust(8) + ('%.3f' % i_sfr[j]).rjust(8) +\
        ('%.3e' % i_age[j]).rjust(14) + ('%.3f' % i_Mu[j]).rjust(8) +\
        ('%.3f' % i_Mv[j]).rjust(8) + ('%.3f' % i_Mj[j]).rjust(8) +\
        ('%.3f' % i_Mnuv[j]).rjust(8) + ('%.3f' % i_Mr[j]).rjust(8) +\
        ('%.4f' % i_s[j]).rjust(10) + ('%.4f' % i_od[j]).rjust(10) +\
        ('%.4f' % i_zod[j]).rjust(10) + '\n')
      file_ij.close()

  if i%1000==0:
    print( str(i) + ' out of ' + str(len(id_c20)) + ' (' + '%.2f' % (i/len(id_c20)*100) + '%) phot. objects completed' )




#----------------------------------------------------
def monte_carlo_z(pz, niter=10000):

  zp_z = np.arange(1001)/100
  sim_z = []
  for i in range(len(zp_z)):
    for j in np.arange(int(pz[i]*1e3)):
      sim_z.append(zp_z[i])
    #print(i)
  sim_z = np.array(sim_z)
  mc_z = []
  for i in range(niter):
    ii = randint(0, len(sim_z)-1)
    mc_z.append( sim_z[ii] )

  return mc_z


#----------------------------------------------------
def get_set_lprun(gid):

  if gid<287340: set=1
  elif gid<593477: set=2
  elif gid<895167: set=3
  elif gid<1200167: set=4
  elif gid<1516388: set=5
  else: set=6

  return set


#----------------------------------------------------
def get_zmc_lprun(zi):

  z_opt = 3+0.05*np.arange(15)
  zn = z_opt.astype(str)
  if ((zi<2.95) | (zi>3.75)):
    return(-99)
  else:
    wz_close = np.where(np.abs(zi-z_opt)==np.min(np.abs(zi-z_opt)))[0][0]
    return z_opt[wz_close] #, zn[wz_close]


#----------------------------------------------------
def get_density_mc(ra_gal, dec_gal, z_gal, vmc_cube=vmc_cube):
  """ RETURNS OVERDENSITY=log(1+delta_gal) AND ASSOCIATED Z-SCORE """

  with fits.open(vmc_cube) as ff:
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

    ### FIND CORRECT COORDINATE INDICES
    ra_vmc = gridx0
    dec_vmc = gridy0
    if ( (ra_gal<np.min(ra_vmc)) | (ra_gal>np.max(ra_vmc)) |\
      (dec_gal<np.min(dec_vmc)) | (dec_gal>np.max(dec_vmc)) ):
      if ((type(z_gal)==numpy.float64) | (type(z_gal)==float)):
        Sigma_mc = [-99]
        od_mc = [-99]
        zscore_mc = [-99]
      else:
        Sigma_mc = np.ones(len(z_gal))*-99
        od_mc = np.ones(len(z_gal))*-99
        zscore_mc = np.ones(len(z_gal))*-99
    else:
      dra = np.abs(ra_vmc-ra_gal)
      ddec = np.abs(dec_vmc-dec_gal)
      rr = np.where(dra==np.min(dra))[0][0]
      dd = np.where(ddec==np.min(ddec))[0][0]

      ### CALCULATE OVERDENSITY AT ALL REDSHIFTS FOR GALAXY PIXEL
      Sigma_rd = dens_all[:,dd,rr]
      od_rd = np.log10( 1+ Sigma_rd / mean_dens_d_fromlog -1 )
  
      ### CALCULATE Z-SCORE OVERDENSITY AT ALL REDSHIFTS FOR GALAXY PIXEL
      zscore = (od_rd-eyfit_x0log_od5) / eyfit_sigmalog_od5
  
      ### GET ZSPEC ZSCORE
      if ((type(z_gal)==numpy.float64) | (type(z_gal)==float)):
        if ((z_gal<np.min(gridz0)) | (z_gal>np.max(gridz0))):
          Sigma_mc = [-99]
          od_mc = [-99]
          zscore_mc = [-99]
        else:
          dz = np.abs(gridz0-z_gal)
          zz = np.where(dz==np.min(dz))[0][0] 
          Sigma_mc = Sigma_rd[zz]
          od_mc = od_rd[zz]    
          zscore_mc = zscore[zz]
  
      ### FOR EACH ITERATION, FIND CORRECT SLICE AND RETURN ARRAY OF ZSCORES
      else:
        Sigma_mc = []
        od_mc = []
        zscore_mc = []
        for zi in z_gal:
          if ((zi<np.min(gridz0)) | (zi>np.max(gridz0))):
            Sigma_mc.append( -99 )
            od_mc.append( -99 )
            zscore_mc.append( -99 )
          else:
            dz = np.abs(gridz0-zi)
            zz = np.where(dz==np.min(dz))[0][0]
            Sigma_mc.append( Sigma_rd[zz] )
            od_mc.append( od_rd[zz] )   
            zscore_mc.append( zscore[zz] )

  return np.array(Sigma_mc), np.array(od_mc), np.array(zscore_mc)



