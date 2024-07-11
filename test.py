import numpy as np
import gzip
import pickle
from time import time
from astropy.io import fits



data = np.random.random(size=(100000, 1000))


t1 = time()
# # Save the array to a compressed .dat file using gzip
np.save(r"C:\Users\sikor\OneDrive\Desktop\BigData\test\test.npy", data)
with gzip.open(r"C:\Users\sikor\OneDrive\Desktop\BigData\test\test.npy.gz", 'wb') as f:
    pickle.dump(data, f)

# t2 = time()
# with gzip.open(r"C:\Users\sikor\OneDrive\Desktop\BigData\test\test.cat.gz", 'wb') as f:
#     pickle.dump(data, f)

# t3 = time()
# with gzip.open(r"C:\Users\sikor\OneDrive\Desktop\BigData\test\test.npy.gz", 'wb') as f:
#     pickle.dump(data, f)


# Create a PrimaryHDU object
hdu = fits.PrimaryHDU(data)

# Use the HDUList to write to a compressed FITS file
# with gzip.open(r"C:\Users\sikor\OneDrive\Desktop\BigData\test\test.fits.gz", 'wb') as f:
#     hdu.writeto(f, overwrite=True)

t4=time()


# print(t2-t1)
# print(t3-t2)
# print(t4-t3)
print(t4-t1)