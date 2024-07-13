from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from time import time
from astropy.cosmology import FlatLambdaCDM
import os
cosmo = FlatLambdaCDM(H0=70, Om0=0.3)
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit



# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------

class GalPop:


    def __init__(self, IDs=[], coords=[], ms=[], obs_type = [], n_sigmas=[], ODs=[], pks={},  mags={}, verbose=False) -> None:
        """
        Create a population of galaxies:
        ATTRIBUTES:
            - IDs   (array)     = array of IDs for the galaxies
            - coords (array)    = 2D array of [ra, dec, z] for each galaxy
                - shape = (# of galaxies, 3)
            - ms    (array)     = array of galaxy masses
            - obs_type (array)  = Specify if galaxy is photoz (0) or specz (1)
            - n_sigmas (array)   = The number of sigma above the mean overdensity in the redshift slice for each galaxy
                        (how find_peaks defines a peak)
                - shape = (# of Galaxies,)
            - ODs   (array)    = The overdensity values for each galaxy 
                - shape = (# of Galaxies,)
            - pks (dict)    = The peak number that a given galaxy is located in for each sigma-threshold. Form {sig_number: array_of_pks}
                - Note this is the peak number in the peak-summary file in which the first peak is n=1 (rather than n=0).
                  If not in a peak, then peak number = -99
            - mags  (dict)      = Dictionary of magnitudes with form {band_name : array_of_mags}
                - Can be added after initialization with "addMag"
            - verbose (bool)    = If changed to True, each method call prints updates to track progress
        """
        ## Optional Variables
        self.IDs    = IDs       # Galaxy IDs
        self.coords = coords        # Galaxy coords [ [ra1, dec1, z1], [ra2, dec2, z2], ...]
        self.ms = ms  # Mass
        self.obs_type = obs_type        # photz or specz

        self.n_sigmas = n_sigmas  # Sigma above mean overdensity
        self.ODs = ODs  # Overdensity values
        self.pks = pks  # Peak number dictionary
        self.mags = mags    # Magnitude dictionary
        self.verbose = verbose  # verbose option for printing

        ## Interior Variables
        self.voxels = []    # Voxels of each galaxy
        self.subpops = {}   # Create samples based on sigma-cuts
        self.vols = {}      # volumes of the sigma cuts
        self.smfs = {}      # smf information stored as [[m1, N1, n1, n_error1], ...] for each value
        self.fits = {}      # Fits to the smfs. Data is stored depending on type of fit (see self.fit_SMF)

        
    # ====================================================================


    def addMag(self, mag_name : str, mag_vals : list[float]) -> None:
        """
        Add an array of magnitudes for the galaxy population. Adds to self.mags as {mag_name : mag_vals}

        INPUTS:
            - mag_name  (str)   = Name of the band. Referenced in functions for mag-cuts
            - mag_vals  (array) = array of
        """
        if self.verbose: print(f"Updating self.mags with new band: {mag_name}")
        self.mags[mag_name] = mag_vals


    # ====================================================================


    def assignPeaks(self, sigs, sig_cube, pk_path, pk_sums, pk_folders, overwrite = False) -> None:
        """
        Assign each galaxy in the galaxy population to a peak given a sigma-threshold to define peaks. Updates the 
        voxel information for each galaxy if needed and adds an array of peak numbers to the pks dictionary attribute.

        INPUTS:
            - sigs  (float)     - The sigma-thresholds. Purely for bookkeeping as these are the keys in self.pks
            - sig_cube (.fits)  - The output of 'make_cube_final_overdense.py'.
            - pk_path (str)    - The path to the directory containing all of the find_peaks information 
                        (i.e. summary files and peak-folders)
            - pk_sums (array)      - List of the file names of the peak summary files in pk_path to use
            - pk_folders (array)   - List of the folder names in pk_path to use
            - overwrite (bool)      - Whether or not to overwrite current peak assignements for threshold
                - Meant to stop any accidental replacement of peak info
        """

        if len(self.voxels) == 0:   # Assign each galaxy to a voxel if needed
            self.update_Voxels(sig_cube)


        ## For each sigma, test membership and return peak numbers
        for idx, s in enumerate(pk_sums):
            if (sigs[idx] in self.pks) and (overwrite==False):  # Check if there is existing peak info
                print(f"Sigma = {sigs[idx]} is already has peak info in self.pks. Specify overwrite=True to change peak info")
                pass
            if self.verbose: print(f"Finding peaks for sigma = {sigs[idx]}")

            pk_sum = np.genfromtxt(pk_path + s, dtype=float, skip_header=0)    # Peak summary file
            pk_dict = {}        # Store info for used peaks to reduce number of file-reads needed
            pk_numbers = []    # Store peak number for each galaxy

            ## Find peak number for each gal
            for v in self.voxels:
                
                pk = -99    # Initially assign the galaxy to no peak

                ## Loop through each peak until galaxy is placed in one
                for p in pk_sum:    

                    # Test if gal is within min-max range of peak
                    if (p[-6]<=v[0] <= p[-5]) and (p[-4]<=v[1]<=p[-3]) and (p[-2]<=v[2]<=p[-1]):
                        
                        try:    # Use peak data from dictionary if it's already been loaded
                            p_data = pk_dict[p[0]]  
                        except: # Otherwise, load in the peak data
                            try:    # If non-single-digit peak (e.g., 10, 25, 102, etc)
                                p_data = np.genfromtxt(pk_path+pk_folders[idx]+"\\"+f"pixels_{int(p[0])}.dat", comments = '#')
                            except: # If single digit peak (e.g., 1, 2, 3, etc)
                                p_data = np.genfromtxt(pk_path+pk_folders[idx]+"\\"+f"pixels_0{int(p[0])}.dat", comments = '#')

                            pk_dict[p[0]] = p_data  # Add to dictionary

                        # Check if voxel is actually in the peak
                        good_voxels = np.where((p_data[:,0] == v[0]) & (p_data[:,1]==v[1]) & (p_data[:,2] == v[2]))[0]

                        if len(good_voxels) != 0:
                            # Match found! Save the peak number for this galaxy and pass to next galaxy
                            pk = p[0]   
                            break
                
                # Add the peak number for the galaxy
                pk_numbers.append(pk)
                        
            # Add to the peak dictionary under with the sigma-threshold as the key
            self.pks[sigs[idx]] = pk_numbers


    # ====================================================================



    def subPop(self, key_name, sig_range, min_mass, z_range, pk_path, pk_sum, sig_cube, cosmo = None, plot="None"):
        """
        INPUTS:
            - key_name (str/float)  = key for self.subpops dictionary and self.vols dictionary where info from this is stored
            - sig_range  (array) = Define the range of overdensities to test [sig_min , sig_max). Note if sig_min != -99 or
                sig_max != np.inf, they should correspond to a key in the self.pks dict set via self.assignPeaks
                - If sig_max == np.inf, report all galaxies above sig_min. 
                - If sig_min == -99, report all galaxies outside of peaks with a voxel above sig_max
            - min_mass (float)  =  Mass-cutoff for peaks
            - z_range    (array) = Define the redshift range to look at [z_min, z_max]
            - pk_path (str)    - The path to the directory containing all of the find_peaks information 
                        (i.e. summary files and peak-folders)
            - pk_sum (str)    - File containing the peak summary data from find_peaks
            - sig_cube (.fits)  - The fits file containing the sigma-cube from find_peaks
            - cosmo (astropy cosmology) = Cosmology to calculate field volume if applicable
            - plot (str)    - Plotting option -- Plots via helper method
                - "None" = No plots
                - "show" = Show the plot in the terminal
                - directory = If you want to save, put the directory name to save to here
        """
        # Check if relevant peak information has been found
        if (sig_range[1] not in self.pks) and (sig_range[1] != np.inf):
                print(f"self.pks has no key {sig_range[1]}. Run self.assignPeaks with this key")
                return
        
        if len(self.n_sigmas) == 0:   # Update n-sigmas if needed
            self.update_n_sigmas(sig_cube)

        if self.verbose:
            print(f"Finding the subpopulation in the peak {key_name}")


        pk_sum = np.genfromtxt(pk_path + pk_sum, dtype=float, skip_header=0)    # Peak summary file
        
        ## Find interpolated position of each peak barycenter
        bRAs = np.interp(x = pk_sum[:,4], xp = np.arange(np.shape(sig_cube[1])[0]), fp = sig_cube[1].data)
        bdecs = np.interp(x = pk_sum[:,5], xp = np.arange(np.shape(sig_cube[2])[0]), fp = sig_cube[2].data)
        bzs = np.interp(x = pk_sum[:,6], xp = np.arange(np.shape(sig_cube[3])[0]), fp = sig_cube[3].data)

        b_coords = np.c_[bRAs, np.c_[bdecs, bzs]]   # Pack peak coordinates into array

        # Make cuts to find relevant peaks in the summary file
        g_idxs = np.where((b_coords[:,2] >= z_range[0]) & (b_coords[:,2] <= z_range[1])  # redshifts
                    & (pk_sum[:,11] >= min_mass) )                # Masses
        
        g_pks = pk_sum[g_idxs]      # List of relevant peak information
        g_coords = b_coords[g_idxs] # list of barycenter coordinates of relevant peaks


        ### DEFINE GAL SAMPLE IN SIGMA-RANGE

        ## FIELD (no lower limit)
        if sig_range[0] == -99: 

            bad_Vol = 0  # Volume of potential structures in field
            test_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1]) )[0] # Gals in redshift range

            # Loop through each peak, find max mass, and remove gals from field if they're in a protostructure
            for pk in g_pks:
                pk_gal_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1])    # In relevant redshift
                            & (self.pks[sig_range[1]] == pk[0]) )[0]       # In the peak
                
                # If max n-sigma is greater than max sigma entered, remove all these galaxies from the field
                if len(pk_gal_idxs) != 0:
                    if np.max(self.n_sigmas[pk_gal_idxs]) >= sig_range[1]:
                        bad_Vol += pk[10]   # Add volume of peak to bad volume
                        # Remove all the indices from the list of good indices
                        bad_ids = np.in1d(test_idxs, pk_gal_idxs)
                        test_idxs = np.delete(test_idxs, bad_ids)
        
        ## UPPER BOUND PROVIDED (finite sigma-interval)
        elif sig_range[1] != np.inf: 
            mask = np.in1d(self.pks[sig_range[1]], g_pks[:,0])   # Make mask for if peak number is in relevant peak
            test_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1])    # In relevant redshift
                            & (self.n_sigmas >= sig_range[0]) & (self.n_sigmas < sig_range[1]) # In relevant overdensity regime
                            & mask )    # In relevant peak
            
        ## NO UPPER BOUND (half-open sigma-interval)
        else:   
            mask = np.in1d(self.pks[sig_range[0]], g_pks[:,0])   # Make mask for if peak number is in relevant peak
            test_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1])    # In relevant redshift
                            & (self.n_sigmas >= sig_range[0])   # In relevant overdensity
                            & mask )    # In relevant 
            
        ### SAVE SAMPLE
        self.subpops[key_name] = test_idxs  # Saves the indices of the galaxies that lay in this sample

        ### FIND VOLUMES

        ## Not the field
        if sig_range[0] != -99:
            rel_sig = int(not(sig_range[1] == np.inf))  # Check if upper bound or not, and find volume accordingly
            unique_peaks = np.unique(self.pks[sig_range[rel_sig]][test_idxs]) # Find peak numbers that have a galaxy in them
            Vol = np.sum(pk_sum[np.argmin(np.abs(unique_peaks[:,np.newaxis] - pk_sum[:,0]), axis=1)][:,10])     # sum the volumes of the peaks



        ## Field sample
        else:
            if cosmo == None: 
                print("Unable to calculate Volume due to missing Cosmology")
                return
            ## Subtract volume of protostructures from the cube
            theta_dec = np.abs(np.max(self.coords[test_idxs][:,1])-np.min(self.coords[test_idxs][:,1]))*np.pi/180 
            theta_RA  = np.abs(np.max(self.coords[test_idxs][:,0])-np.min(self.coords[test_idxs][:,0]))*np.pi/180  * np.cos(theta_dec)
            Omega     = theta_RA * theta_dec # get rid of unit
            V_cube  = Omega/(4*np.pi) *(cosmo.comoving_volume(z_range[1]) - cosmo.comoving_volume(z_range[0]))
            Vol = V_cube.value - bad_Vol


        self.vols[key_name] = Vol

        ## Plot results
        if plot != "None":
            good_gals = self.coords[test_idxs]     # Trim down to galaxies which may be in a peak
            bad_gals = np.delete(self.coords, test_idxs, axis=0)   # Galaxies which aren't used.
            bad_gals = bad_gals[np.where((bad_gals[:,2] >= z_range[0]) & (bad_gals[:,2] <= z_range[1]))]    # In relevant redshift

            self.subPop_plot(sig_range, z_range, g_coords, g_pks, good_gals, bad_gals, plot)


    # ====================================================================



    def subPop_plot(self, sig_range, z_range, g_coords, g_pks, good_gals, bad_gals, plot):
        """
        Helper method for plotting data from the subPop
        """

        ra_range = [np.min(self.coords[:,0]), np.max(self.coords[:,0])]
        dec_range = [np.min(self.coords[:,1]), np.max(self.coords[:,1])]


        style_dict = {          # Dictionary of all the styles
            12: ['*', 'red', "[12,12.5)"], 12.5: ['h', 'gold', "[12.5,13)"], 13: ['o', 'deepskyblue', "[13,13.5)"],
            13.5: ['X', 'darkorange', "[13.5,14)"], 14 :['>', 'forestgreen', "[14,14.5)"], 14.5 : ['s', 'royalblue', "[14.5,15)"], 
            15 :['D', 'maroon', r"M $\geq15$"]  }
        
        ## Set up gridspec plot
        fig = plt.figure(figsize=(14,10))
        gs = gridspec.GridSpec(2, 1, wspace=0)

        # Row 1 -- plotting peak locations
        row = gs[0].subgridspec(1,2, width_ratios=[1,1.5])
        ax00, ax01 = fig.add_subplot(row[0]), fig.add_subplot(row[1], projection='3d')

        round_ms = np.array([min(15,m//0.5/2) for m in g_pks[:,11]])   # Rounded masses that dictate the plotting style

        for k in list(style_dict.keys()):
            peaks = np.where(k == round_ms)[0]      # Find which peaks are in the given mass-bin

            if len(peaks) != 0: # If there are some peaks in the mass-bin:

                # 2D plot
                ax00.scatter(g_coords[peaks][:,0], g_coords[peaks][:,1], marker=style_dict[k][0],
                            c=style_dict[k][1], s=100, label=style_dict[k][2])  
                # 3D plot
                ax01.scatter(g_coords[peaks][:,2], g_coords[peaks][:,0], g_coords[peaks][:,1],
                    marker=style_dict[k][0], c=style_dict[k][1], s=100, label=style_dict[k][2])
                
        # Clean up plots after all points are plotted
        ax00.set(title="Peak Locations", xlim=ra_range, ylim=dec_range, xlabel="RA (deg)", ylabel="Dec (deg)")
        ax00.legend(title = r"$\log_{10}(M_*)\in$")
        ax00.invert_xaxis()
        ax01.set(title="Peak Locations", xlim=z_range, ylim=ra_range, zlim=dec_range, xlabel="z", ylabel="RA (deg)", zlabel="Dec (deg)")
        ax01.legend(title = r"$\log_{10}(M_*)\in$")
        ax01.set_box_aspect((5,5,3), zoom=1.2)
        ax01.view_init(25)
        ax01.invert_yaxis()
        ax01.invert_xaxis()

        # Row 2 -- Plotting good vs bad galaxies
        row = gs[1].subgridspec(1,2, width_ratios=[1,1.5])
        ax10, ax11 = fig.add_subplot(row[0]), fig.add_subplot(row[1], projection='3d')

        # 2D
        ax10.scatter(good_gals[:,0], good_gals[:,1], marker='.', c='g') # Usable galaxies
        ax10.scatter(bad_gals[:,0], bad_gals[:,1], marker='.', c='r', alpha=0.25, label=f"{len(bad_gals)} unsuable gals") # Unusable galaxies

        # 3D
        ax11.scatter(good_gals[:,2], good_gals[:,0], good_gals[:,1], marker='.', c='g', label=f"{len(good_gals)} usable galaxies")

        # Clean up plots
        ax10.set(title="Usability of Galaxies", xlim=ra_range, ylim=dec_range, xlabel="RA (deg)", ylabel="Dec (deg)")
        ax10.invert_xaxis()
        ax10.legend()
        ax11.set(title="Usable Galaxies", xlim=z_range, ylim=ra_range, zlim=dec_range, xlabel="z", ylabel="RA (deg)", zlabel="Dec (deg)")
        ax11.set_box_aspect((5,5,3), zoom=1.2)
        ax11.view_init(25)
        ax11.legend()
        ax11.invert_yaxis()
        ax11.invert_xaxis()
        if sig_range[1] > 0:
            fig.suptitle(rf"$\sigma \in$ [{sig_range[0]}, {sig_range[1]})", fontsize=18)
        else:
            fig.suptitle(rf"$\sigma \geq$ {sig_range[0]}", fontsize=18)
    

        # Show/save plots
        if plot in ("show", "Show"): plt.show()
        else:
            if sig_range[1] >0:
                try: plt.savefig(plot + f"\Sigma_{sig_range[0]}_{sig_range[1]}_plot.png")
                except:
                    try:
                        os.mkdir(plot)
                        plt.savefig(plot + f"\Sigma_{sig_range[0]}_{sig_range[1]}_plot.png")
                    except: print("Unable to make plot")
            else:
                try: plt.savefig(plot + f"\Sigma_{sig_range[0]}_plot.png")
                except:
                    try:
                        os.mkdir(plot)
                        plt.savefig(plot + f"\Sigma_{sig_range[0]}_plot.png")
                    except: print("Unable to make plot")
        plt.close()

    # ====================================================================

    def make_SMF(self, subPop_keys, smf_keys ,m_range):
        """
        Generates SMFs based on the key-names associated with different subpops. 

        INPUTS:
            - subPop_keys (array) =  keys for self.subpops dictionary. These are the sub pops which will have an SMF generated for them
            - smf_keys  (array) = key names in the self.smfs dictionary to store the smf info
                - NOTE: Info is stored as an array of [[m, N, n, n_error], ...] for each key.
            - m_range (array) = mass-range to generate the SMF for
                - [min_mass, max_mass, m_step]
        """
        for sk, k in zip(subPop_keys, smf_keys):

            smf_info = []       # Store smf info for this key as [[m, N, n, n_err], ...]
            mass = m_range[0]   # keep track of mass

            while mass <= m_range[1]:
                smf_mbin = []   # [m, N, n, n_error] for this mass bin

                # Find galaxies in the mass bin
                gals = np.where( (self.ms[self.subpops[k]] >= mass) & (self.ms[self.subpops[k]] < mass + m_range[2]) )[0]

                if len(gals) != 0:  # If there are galaxies in the mass bin
                    smf_mbin.append(np.median(self.ms[self.subpops[k]][gals]))  # mass
                    smf_mbin.append(len(gals))                  # N
                    smf_mbin.append(len(gals) / self.vols[k] / m_range[2])  # n
                    smf_mbin.append(np.sqrt(len(gals)) / self.vols[k] / m_range[2])  # n_error
                else:   # No galaxies in the mass bin
                    smf_mbin.append(mass)       
                    smf_mbin.append(np.nan)
                    smf_mbin.append(np.nan)
                    smf_mbin.append(np.nan)

                mass += m_range[2]  # Step up to new mass bin
                smf_info.append(smf_mbin)

            self.smfs[sk] = np.array(smf_info)     # Add the info for the current key


    # ====================================================================


    def SMF_plot(self, smf_keys, smf_labels, fit_keys, fit_labels, title="", plot=""):
        """
        Plot different SMFs together

        INPUTS:
            - smf_keys (array)  = Keys in self.smfs to plot
            - smf_labels    (array) = Plot labels for the smf_keys
            - fit_keys  (array) = Keys in self.fits to plot
            - fit_labels    (array) = Plot labels for the fit_keys
            - title (str)   = Title of the plot. If '', title it "SMF"
            - plot (str)    = Path to save the string at. If '', then it shows the plot.
        """
        colors = ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200']  
        shapes = ["o", "H", "P", "s", "D", (5,1,0)]
        min_m, max_m = 99, -99  # For plot limits
        for i, k in enumerate(smf_keys):
            try:
                plt.errorbar(self.smfs[k][:,0], self.smfs[k][:,2], self.smfs[k][:,3], 
                             label=smf_labels[i], marker=shapes[i], color=colors[i], ls='')
                min_m = min(min_m, min(self.smfs[k][:,0]))
                max_m = max(max_m, max(self.smfs[k][:,0]))
            except: print(f"The key {k} is not in self.smfs. Generate the smf and run again")
        # Fix up mass limits
        min_m = 8.5 if min_m==99 else min_m-0.5
        max_m = 12 if max_m==-99 else max_m+0.5

        for i, k in enumerate(fit_keys):
            try:    
                params = self.fits[fit_keys]
            except: print(f"The key {k} is not in self.fits. Generate the fit and run again")
            m_vals = np.linspace(min_m, max_m, 1000)

            if len(params) == 3:    # Single Schechter fit
                plt.plot(m_vals, self.schechter(m_vals, *params), color=colors[i], marker='', label=fit_labels[i])
            else:    # Double Schechter fit
                plt.plot(m_vals, self.Dschechter(m_vals, *params), color=colors[i], marker='', label=fit_labels[i])
      
        
        plt.yscale("log")
        if (len(smf_keys)==0) or (len(fit_keys==0)): plt.legend(loc='lower left')
        else: plt.legend(loc='lower left', ncol=2)
        if title=="": plt.title("SMF") 
        else: plt.title(title)
        plt.ylabel(r"$\rm \phi \quad [N\, cMpc^{-3}\,dex^{-1}]$", fontsize=15)
        plt.xlabel(r"$\rm \log_{10}(M_*/M_\odot)$", fontsize=15)    
        plt.xlim(min_m, max_m)

        if plot != "":
            try:
                plt.savefig(plot)
            except:
                os.mkdir(plot)
                plt.savefig(plot)
        else:
            plt.show()


    # ====================================================================

    def SMF_relPlot(self, base_key, base_label, smf_keys, smf_labels, title="", plot=""):
        """
        Plot SMFs relative to a single one (i.e. the ratio of the smfs to the base-smf)

        INPUTS:
            - base_key (key)    = Key in self.smfs to use as the normalizing smf
            - base_label (str)  = label for the base smf in the y_axis (i.e. \phi_{base_label})
            - smf_keys (array)  = Keys in self.smfs to plot normalized by the base-smf
            - smf_labels    (array) = Plot labels for the smf_keys
            - title (str)   = Title of the plot. If '', title it "SMF"
            - plot (str)    = Path to save the string at. If '', then it shows the plot.
        """
        colors = ['006BA4', 'FF800E', 'ABABAB', '595959', '5F9ED1', 'C85200']  
        shapes = ["o", "H", "P", "s", "D", (5,1,0)]
        min_m, max_m = 99, -99  # For plot limits
        try:
            norms = self.smfs[base_key][:,2]
        except:
            print(f"The key {base_key} is not in self.smfs. Generate the smf and run again")
            return
        
        for idx, k in enumerate(smf_keys):
            try:
                # Set length of normalizing array
                if len(norms) > len(smf_keys[k]): 
                    n = norms[:len(smf_keys[k])]   # Shorten normalizing array
                    data = smf_keys[k] / n
                else:   # Shorten data
                    data = smf_keys[k][:len(norms)] / norms
                plt.errorbar(data[:,0], data[:,2], data[:,3], 
                             label=smf_labels[idx], marker=shapes[idx], color=colors[idx])
                min_m = min(min_m, min(self.smfs[k][:,0]))
                max_m = max(max_m, max(self.smfs[k][:,0]))
            except: print(f"The key {k} is not in self.smfs. Generate the smf and run again")

        plt.yscale("log")
        plt.legend(loc='lower left')
        if title=="": plt.title("SMF") 
        else: plt.title(title)
        plt.ylabel(rf"$\rm \phi/\phi_{base_label}$", fontsize=15)
        plt.xlabel(r"$\rm \log_{10}(M_*/M_\odot)$", fontsize=15)    
        plt.xlim(min_m, max_m)

        if plot != "":
            try:
                plt.savefig(plot)
            except:
                os.mkdir(plot)
                plt.savefig(plot)
        else:
            plt.show()

    # ====================================================================


    def fit_SMF(self, smf_keys, fit_keys, N_schechters, **kwargs):
        """
        Fits either a single or double schechter function to a given SMF. Fits are done with scipy.optimize.curve_fit


        INPUTS:
            - smf_key (array)   = Key names for SMF in self.smfs to fit (see self.make_SMF)
            - fit_key (array)   = Key names for this fit in the self.fits dict. The fit is stored as:
                - 1 (single)      --> [M_star, phi_star, alpha]
                - 2 (double)      --> [M_star, phi_s1, phi_s2, alpha_1, alpha_2]
            - N_schechter (array) = Either 1 (single) or 2 (double) schechter.
            - **kwargs         = Optional variables for curve_fit
        """
        for s_key, f_key, Ns in zip(smf_keys, fit_keys, N_schechters):


            if Ns == 1:     fit_fn = self.schechter
            elif Ns == 2:   fit_fn = self.Dschechter
            else:   
                print("Elements of N_schechter must be either 1 or 2")
                return

            f_err, f_params = curve_fit(fit_fn, self.smfs[s_key][:,0], self.smfs[s_key][:,2], sigma=self.smfs[s_key][:,3],
                                        *kwargs.values())
            self.fits[f_key] = f_params





    
    def schechter(M, M_star, phi_star, alpha):
        """
        Single-Schecter Function
        """
        return np.log(10)*phi_star*10**((M-M_star)*(1+alpha))*np.exp(-10**(M-M_star))
    
    def Dschechter(M, M_star, phi_s1, phi_s2, alpha_1, alpha_2):
        """
        Double-Schechter function
        """
        return np.log(10)*np.exp(-10**(M-M_star))*(phi_s1*(10**(M-M_star))**(alpha_1+1) \
        +phi_s2*(10**(M-M_star))**(alpha_2+1) )




    # ====================================================================




    def update_Voxels(self, sig_cube) -> None:
        """
        Place each galaxy of the galaxy population into a voxel within the overdensity cube. The voxel coordinates are assigned to
        the 'voxel' attribute which speeds up future computations

        INPUTS:
            - sig_cube (.fits)  - The output of 'make_cube_final_overdense.py'.
        """
        if self.verbose:    print("Updating voxel assignments (self.voxels)")

        ## For each galaxy, find the nearest voxel from the data cube
        self.voxels = np.c_[
            np.argmin(np.abs(self.coords[:,0][:, np.newaxis] - sig_cube[1].data), axis=1),  # ra
            np.argmin(np.abs(self.coords[:,1][:, np.newaxis] - sig_cube[2].data), axis=1),  # dec
            np.argmin(np.abs(self.coords[:,2][:, np.newaxis] - sig_cube[3].data), axis=1)   # z
        ]


    # ====================================================================

    def update_ODs(self, sig_cube):
        """
        Update the overdensities values for each galaxy based on the voxels stored in self.voxels

        INPUTS:
            - sig_cube (.fits)  - The output of 'make_cube_final_overdense.py'.
        """
        ## Update voxels if there are none
        if len(self.voxels) == 0:
            self.update_Voxels(sig_cube)
        
        if self.verbose: print("Updating overdensity values (self.ODs)")

        ## Calculate Overdensities and assign to attribute
        self.ODs = np.log10(sig_cube[0].data[self.voxels[:,2],self.voxels[:,1],self.voxels[:,0]] / sig_cube[6].data[self.voxels[:,2]] )
        

    # ====================================================================


    def update_n_sigmas(self, sig_cube):
        """
        Update the number of sigma above the mean overdensity in the redshift slice for each galaxy

        INPUTS:
            - sig_cube (.fits)  - The output of 'make_cube_final_overdense.py'.
        """
        ## Update voxels if there are none
        if len(self.voxels) == 0:
            self.update_Voxels(sig_cube)

        ## Update overdensities if needed
        if len(self.ODs) == 0:
            self.update_ODs(sig_cube)
        
        if self.verbose: print("Updating n_sigmas values (self.n_sigmas)")

        ## Calculate Overdensities and assign to attribute
        self.n_sigmas  = (self.ODs - sig_cube[11].data[self.voxels[:,2]]) / sig_cube[12].data[self.voxels[:,2]]    # (od - mean) / sigma
        

    # ====================================================================

    def combine(self, other):
        """
        Combine this instance with another GalPops instance to create a new object. 
        Verbose for new object is set to True if both have Verbose=True
            - NOTE: This method is currently is a bit messy with SMFs (and related attributes). It's recommended to 
            remake the SMFs after combining

        INPUTS:
            - other (GalPop)    = Other GalPop object to combine with this one

        OUTPUTS:
            - (GalPoP) = returns a *new* object. This combination does not happen in place
        """
        combined_attrs = {}     # where to store the new attributes

        ## Loop through the attributes and combine where possible
        for attr in self.__dict__:

            self_attr = getattr(self, attr)     # Attrs of this obj
            other_attr = getattr(other, attr)   # other attrs

            if isinstance(self_attr, np.ndarray): # attr is a list
                combined_attrs[attr] = np.concatenate((self_attr, other_attr))   
            elif isinstance(self_attr, dict):   # attr is a dict
                combined_attrs[attr] = self._combine_dicts(self_attr, other_attr)
        combined_attrs["verbose"] = bool(self_attr and other_attr)  # set verbose option

        combined_instance = GalPop()
        for attr, value in combined_attrs.items():
            setattr(combined_instance, attr, value)
        return combined_instance
    
    def _combine_dicts(self, dict1, dict2):
        """
        Helper for self.combine. This combines two dictionary attributes for the two objects
        """
        combined_dict = {}
        for key in set(dict1.keys()).union(dict2.keys()):
            # Pull the lists from the dictionaries
            ls1 = dict1.get(key, np.array([]))
            ls2 = dict2.get(key, np.array([]))
            # Turn into numpy arrays if needed
            ls1 = np.array(ls1) if not isinstance(ls1, np.ndarray) else ls1
            ls2 = np.array(ls2) if not isinstance(ls2, np.ndarray) else ls2
            combined_dict[key] =  np.concatenate((ls1, ls2))
        return combined_dict

    # ====================================================================


    def saveFile(self, path):
        """
        Saves most attributes at the path using np.save --> file type must be .npy! Note that all attributes will be saved. The
        columns of the resulting .npy file are:
            - 0-2 = ra, dec, z
            - 3 = mass
            - 4-6 = voxel-x, voxel-y, voxel-z
            - 7 = n_sigmas
            - 8 = ODs
            - 9+ = peak number at given sigma-threshold

        Note that any column 5-9 that does not have data will be marked with NaNs in the file. This file structure can be read
        back in to other objects later
        """
        # Make inital headers
        header = ["ID", "ra", "dec", "z", "ms", "vx", "vy", "vz", "ODs", "n_sigmas"]

        # Make a NaN array to fill the gaps
        save_array = np.empty((len(self.coords), len(header)+len(self.pks))) 
        save_array.fill(np.nan)


        if len(self.IDs) != 0:
            save_array[:,0] = self.IDs
        if len(self.coords) != 0:
            save_array[:, 1:4] = self.coords
        if len(self.ms) != 0:
            save_array[:,4] = self.ms
        if len(self.voxels) != 0:
            save_array[:, 5:8] = self.voxels
        if len(self.ODs) != 0:
            save_array[:,8] = self.ODs
        if len(self.n_sigmas) != 0:
            save_array[:,9] = self.n_sigmas

        for sig in self.pks:
            # Note this updates length of header, so column number also updates
            save_array[:,len(header)] = self.pks[sig]
            header.append(str(sig))
        
        # Save to structured array
        dtype = [(h, "f8") for h in header]
        structured = np.zeros(save_array.shape[0], dtype=dtype)

        for i, h in enumerate(header):
            structured[h] = save_array[:,i]

        np.save(path, structured)  # Save to path

    # ====================================================================
    
    def loadFile(self, path):
        """
        Loads files assuming the same structure as saveFile. Fills in attributes where appropriate.
        """
        try:
            data = np.load(path, allow_pickle=True)
        except:
            print("There were problems loading the following file:\n ", path)
            return
        
        self.IDs = data["ID"]
        self.coords = np.c_[data["ra"], data["dec"], data["z"]]    # Galaxy coordinates
        self.ms = data["ms"]     # Galaxy masses

        # Check voxels
        vx = np.where(data["vx"] != data["vx"])[0]    # Check for how many NaNs
        vy = np.where(data["vy"] != data["vy"])[0] 
        vz = np.where(data["vz"] != data["vz"])[0] 
        if (len(vx) != len(self.ms)) | (len(vy) != len(self.ms)) | (len(vz) != len(self.ms)):   # If voxels were reported
            self.voxels = np.c_[data["vx"], data["vy"], data["vz"]]

        # Check Overdensities and n_sigmas
        if len(np.where(data["ODs"] != data["ODs"])[0]) != len(self.ms):
            self.ODs = data["ODs"]
        if len(np.where(data["n_sigmas"] != data["n_sigmas"])[0]) != len(self.ms):
            self.n_sigmas = data["n_sigmas"]

        # Load peak numbers if reported
        if len(data.dtype.names) > 10:
            for idx in range(len(data.dtype.names) - 10):
                self.pks[data.dtype.names[idx+10]] = data[data.dtype.names[idx+10]]


    # ====================================================================
    # ====================================================================



