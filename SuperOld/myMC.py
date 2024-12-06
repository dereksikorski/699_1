import numpy as np
from scipy.stats import skewnorm
import matplotlib.pyplot as plt
from astropy.io import fits






def MCz(zs, ws, MC_fn,verbose=False, **kwargs):
    """
    Performs a Monte Carlo on the redshift distribution of input galaxies

    INPUTS:
        - zs (array)    = List of median redshift values
        - ws (array)    = weights for the MC (i.e. z is changed if random number >= ws)
        - MC_fun (fn)   = Python function used to generate the new redshift values for the galaxies

    OUTPUTS:
        - (array) List of redshifts
    """

    new_idxs = np.where( np.random.random(size=len(zs)) >= ws )     # Indices of zs that need to be replaced
    if verbose==True:
        print(kwargs)

    nz = MC_fn(*kwargs.values())        # Generate the set of redshifts

    zs[new_idxs] = nz    # Replace redshifts as dictated by the MC

    return zs


# -------------------------------------------------------------------------


def my_PDF(xs, u68, o68):
    """
    PDFs to draw the new redshifts from. Skewed-normal based on the confidence interval from COSMOS2020

    INPUTS:
        - xs (array)    = Median redshift values
        - l68 (array)   = Lower bound of the 68% confidence interval
        - u68 (array)   = Upper bound of the 68% confidence interval
    OUTPUTS:
        - (array)   = New redshift values. redshifts <0 or unavailable are marked -99
    """
    omega = np.sqrt((u68**2 + o68**2)/2)        # Scale of skew-normal  
    alpha = ((np.pi/2)**0.5 * (o68-u68)) / np.sqrt(2*u68**2 + 2*o68**2 - np.pi*(o68-u68)**2/2)  # Skewness
    bad_idxs = np.where((omega != omega) | (alpha != alpha) )   # Find nan's in either array

    # Replace NaNs with 1 temporarily for calculation
    alpha[bad_idxs] = 1     
    omega[bad_idxs] = 1

    z_vals = skewnorm.rvs(a=alpha, loc=xs, scale=omega) # Find new zs based on skew-normal
    z_vals[bad_idxs] = np.nan  # Replace NaNs with -99
    return z_vals


# -------------------------------------------------------------------------

def MC_weights(qf):
    """
    Get the weights for the MC iterations given a quality flag.
    For photometry, take qf == 0
    """
    weights = np.select( [(qf>0)&(qf%10==2), (qf>0)&(qf%10==9), (qf>0)&(qf%10==3), (qf>0)&(qf%10==4) ],
              [0.7, 0.7, 0.993, 0.993],
              default=0)
    
    return weights


# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
niter = 1000    # Number of MC iterations

## SET RANGE YOU'RE INTERESTED IN ##

ra_range = (149.4, 150.7)
dec_range = (1.5, 2.9)            # RA/Dec/z extended from Cucciati+ 18 to be more complete
z_range = (2., 3.)
IRAC_limit = 26.5



## PREP COSMOS DATA ##
cosmos_file = fits.open(r"C:/Users/sikor/OneDrive/Desktop/BigData/COSMOS2020/COSMOS2020_CLASSIC_R1_v2.0.fits")
cosmos = cosmos_file[1].data

phot_use_idx = np.where( (ra_range[0]<= cosmos["ALPHA_J2000"]) & (cosmos["ALPHA_J2000"] <= ra_range[1])          # RA check
                 & (dec_range[0] <= cosmos["DELTA_J2000"]) & (cosmos["DELTA_J2000"] <= dec_range[1])      # Dec check
                 & (cosmos["lp_type"]==0)       # Object type from lephare  (galaxies) 
                 & ((cosmos["IRAC_CH1_MAG"] <= IRAC_limit) | (cosmos["IRAC_CH2_MAG"] <= IRAC_limit) ))[0]   #  IRAC mag-cut        


c20p_all = cosmos[phot_use_idx]     # Trim the spec catalog to only include galaxies I care about


### PREP SPEC DATA ###
c20s = np.loadtxt("Data/Hyperion_C20_spec_noOD.txt", dtype=str, skiprows=1)

# Fix up the formatting for the spec data-file:
new_array = []
for idx in range(c20s.shape[1]):
    try:
        col = c20s[:,idx].astype(np.float32)
    except:
        col = c20s[:,idx]
    new_array.append(col)

c20s = np.array(new_array, dtype=object)
c20s = np.transpose(c20s)


### TRIM OUT THE SPECTRA FROM COSMOS ###
idxs = []
for i, c20_id in enumerate(c20p_all["ID"]):
    if c20_id in c20s[:,0]: idxs.append(i)

c20p_all = np.delete(c20p_all, idxs, axis=0)



### RUN FOR PHOTOZs ###

med_zs = c20p_all["lp_zPDF"]        # Redshifts
ws = np.zeros(len(med_zs))          # Weights (all set to 0)

l68_c20 = c20p_all["lp_zPDF_l68"]    # l68
u68_c20 = c20p_all["lp_zPDF_u68"]    # u68



# Setup to add columns to existing C20p array
old_ds = c20p_all.dtype.descr    # Old dtypes
new_ds = []
new_zs = np.zeros(shape=(niter, c20p_all.shape[0]))



for n in range(niter):

    if n%10==0: print(n)

    zs_in = np.copy(med_zs)     # Copy initial redshifts

    # Find new redshifts from MC
    new_z = MCz(zs_in, ws, my_PDF, xs=med_zs, l68=med_zs -l68_c20, u68=u68_c20-med_zs)
    new_zs[n] = new_z


    ## Add the new column for the MC iteration
    new_ds.append((f"MC_iter_{n}", ">f8"))

    
    ### PLOT ###
    fig, ax = plt.subplots()
    bbox = dict(boxstyle='round', fc = "white", ec='k', alpha=0.5)

    ax.hist(new_z, bins=np.arange(0,8,0.05))
    ax.set_title(f"Redshift Distribution of C20 -- (MC_iter {n})")
    ax.text(0.7,0.9, f"# of Galaxies = {len(np.where((0 <=new_z) & (8>= new_z))[0])}", fontsize=7, bbox=bbox, transform=ax.transAxes)
    ax.set_xlabel("z")
    ax.set_ylabel("N")
    plt.savefig(f"./MC_iterations/DistPlots/run_{n}")
    plt.clf()

    fig, ax = plt.subplots()
    ax.hist(new_z, bins=np.arange(2,3,0.01))
    ax.set_title(f"Redshift Distribution of Field -- (MC_iter {n})")
    ax.text(0.7, 0.9, f"# of Galaxies = {len(np.where((2 <=new_z) & (3>= new_z))[0])}", fontsize=7, bbox=bbox, transform=ax.transAxes)
    ax.set_xlabel("z")
    ax.set_ylabel("N")
    plt.savefig(f"./MC_iterations/FieldPlots/run_{n}")



new_zs = np.array(new_zs)

new_ds = np.dtype(old_ds + new_ds)      # New columns for the numpy array

## Keep only galaxies that fell within correct z-range at least once ###
z_bool = ((2< new_zs) & (new_zs < 3)).any(axis=0)
good_idxs = np.where(z_bool)[0]     # Where the condition is met

c20p_all = np.take(c20p_all, good_idxs)     # keep galaxies in c20p that meet condition
new_zs = np.take(new_zs, good_idxs, axis=1) # keep mc-iterations for galaxies that meet condition

c20p = np.zeros(c20p_all.shape, dtype=new_ds)    # generate an array of zeros to fill

## Fill up the array
for d in old_ds:    # Refill everything that was in original file
    c20p[d[0]] = c20p_all[d[0]]

for n in range(niter):  # Refill with new iterations
    c20p[f"MC_iter_{n}"] = new_zs[n]

np.save(r"C:\Users\sikor\OneDrive\Desktop\BigData\Hyperion\CosmosMC.npy", c20p)