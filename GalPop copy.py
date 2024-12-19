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
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm
from scipy.spatial import KDTree
from multiprocessing import Pool
from typing import *
from collections.abc import Iterable, Mapping




# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------

class GalPop:


    def __init__(self, 
                 IDs: Iterable[object], coords: Iterable[Iterable[float]], 
                 mags: Mapping[str, Iterable[float]] = None, verbose: bool = False, misc: dict = None) -> None:
        """
        Create a Galaxy Population object. This requires a number of miscellaneous inputs based on what the user wants to do.
        
        INPUTS:
            - IDs    = An array of unique IDs for the galaxies in the population
            - coords = A 2D array of coordinates for each galaxy in the form:
                - [ [RA_1, Dec_1, z_1], [RA_2, Dec_2, z_2], ...]
            - mags (optional)   = An array of magnitudes
            - verbose (optional)    = Whether or not to print updates 
            - misc (optional) = Dictionary of optional parameters depending on functions used            
        """
        # Required
        self.IDs = IDs 
        self.coords = coords

        # Optional
        self.mags = mags if mags is not None else {}
        self.verbose = verbose
        self.misc = misc if misc is not None else {}


        ## Internal Variables
        self.voxels = []    # Voxels of each galaxy
        self.subpops = {}   # Create samples based on sigma-cuts
        self.vols = {}      # volumes of the sigma cuts
        self.smfs = {}      # smf information stored as [[m1, N1, n1, n_error1], ...] for each value
        self.fits = {}      # Fits to the smfs. Data is stored depending on type of fit (see self.fit_SMF)
        
    # ====================================================================


    def addMag(self, mag_name : str, mag_vals : Iterable[float]) -> None:
        """
        Add an array of magnitudes for the galaxy population. Adds to self.mags as {mag_name : mag_vals}

        INPUTS:
            - mag_name   = Name of the band. Referenced in functions for mag-cuts
            - mag_vals   = The array of magnitudes
        """
        if self.verbose: print(f"Updating self.mags with new band: {mag_name}")
        self.mags[mag_name] = mag_vals



    # ====================================================================

    def assignPeaks(self, sigs, sig_cube, sig_summary_files, sig_data_folders, overwrite=False):
        """
        Assign galaxies to overdensity peaks.

        Parameters:
            sigs (list): List of sigma thresholds.
            sig_summary_files (list): List of summary files for each sigma threshold.
            sig_data_folders (list): List of directories containing peak voxel data files for each sigma.
            overwrite (bool): Whether to overwrite existing assignments in self.pks.
        """
        if not overwrite and self.pks:
            raise ValueError("Peak assignments already exist. Use overwrite=True to reassign.")
        
        if len(self.n_sigmas) == 0:  # Assign each galaxy to a voxel if needed
            self.update_n_sigmas(sig_cube)


        # Initialize pks for each sigma
        self.pks = {sig: [-99] * len(self.voxels) for sig in sigs}

        # Process each sigma level
        for sig, summary_file, data_folder in zip(sigs, sig_summary_files, sig_data_folders):
            # Load the peak summary file
            pk_sum = np.genfromtxt(summary_file, comments="#")
            
            # Extract bounds and peak centers
            peak_centers = pk_sum[:, 1:4].astype(int)  # Columns 1, 2, 3 are peak center voxels
            peak_bounds = pk_sum[:, -6:].reshape(-1, 3, 2).astype(int)  # Min/max bounds for x, y, z
            
            # Preload peak voxel data into sets for fast lookup
            peak_voxel_sets = {}
            for row in pk_sum:
                peak_id = int(row[0])  # Peak number starts at 1
                try:
                    data = np.genfromtxt(f"{data_folder}/pixels_{peak_id:02d}.dat", comments="#")
                    peak_voxel_sets[peak_id] = set(map(tuple, data[:, :3].astype(int)))
                except FileNotFoundError:
                    continue
            
            # Create KDTree for peak centers
            tree = KDTree(peak_centers)
            max_radius = np.sqrt(np.max((peak_bounds[:, :, 1] - peak_bounds[:, :, 0]) ** 2).sum())

            # Assign galaxies to peaks
            assignments = [-99] * len(self.voxels)  # Initialize with -1 (unassigned)
            for voxel_idx, voxel in enumerate(self.voxels):
                # Find nearby peaks using KDTree
                nearby_indices = tree.query_ball_point(voxel, max_radius)
                for idx in nearby_indices:
                    bounds = peak_bounds[idx]
                    # Check if voxel is within the bounds
                    if all(bounds[dim, 0] <= voxel[dim] <= bounds[dim, 1] for dim in range(3)):
                        # Check if voxel is in the voxel set for this peak
                        peak_id = int(pk_sum[idx, 0])  # Map back to the correct peak number
                        if tuple(voxel) in peak_voxel_sets.get(peak_id, set()):
                            assignments[voxel_idx] = peak_id
                            break

            self.pks[sig] = assignments




    def subPop(self, key_name, sig_range, min_mass, z_range, pk_path, pk_sum, sig_cube, pk_nums = [], pk_folder='None', cosmo = None, plot="None"):
        """
        INPUTS:
            - key_name (str/float)  = key for self.subpops dictionary and self.vols dictionary where info from this is stored
            - sig_range  (array) = Define the range of overdensities to test [sig_min , sig_max). Note if sig_min != -99 or
                - sig_max != np.inf, they should correspond to a key in the self.pks dict set via self.assignPeaks
                - If sig_max == np.inf, report all galaxies above sig_min. 
                - If field (i.e. pk_folder != 'None'), sig_min defines potential protostuctures and sig_max confirms a protostructure
            - min_mass (float)  =  Mass-cutoff for peaks
            - z_range    (array) = Define the redshift range to look at [z_min, z_max]
            - pk_path (str)    - The path to the directory containing all of the find_peaks information 
                        (i.e. summary files and peak-folders)
            - pk_sum (str)    - File containing the peak summary data from find_peaks
            - sig_cube (.fits)  - The fits file containing the sigma-cube from find_peaks
            - pk_nums (list)    = List of the peak numbers to test for (note not the peak index, but peak number)
                    - If [], it tests all peaks
            - pk_folder (str)   - Path the folder containing peak info related to the pk_sum (only needed for field)
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
        
        if (len(self.n_sigmas) == 0) or (np.all(np.isnan(self.n_sigmas))):   # Update n-sigmas if needed
            self.update_n_sigmas(sig_cube)

        if self.verbose:
            print(f"Finding the subpopulation in the peak {key_name}")


        pk_sum = np.genfromtxt(pk_path + pk_sum, dtype=float, skip_header=0)    # Peak summary file
        
        ## Find interpolated position of each peak barycenter
        bRAs = np.interp(x = pk_sum[:,4], xp = np.arange(np.shape(sig_cube[1])[0]), fp = sig_cube[1].data)
        bdecs = np.interp(x = pk_sum[:,5], xp = np.arange(np.shape(sig_cube[2])[0]), fp = sig_cube[2].data)
        bzs = np.interp(x = pk_sum[:,6], xp = np.arange(np.shape(sig_cube[3])[0]), fp = sig_cube[3].data)

        b_coords = np.c_[bRAs, np.c_[bdecs, bzs]]   # Pack peak coordinates into array

        ## Make cut on relevant peaks based on mass, redshift, and specified pk numbers
        if len(pk_nums) == 0: pk_nums = pk_sum[:,0]     # If no pks specified, assume all are ok

        g_idxs = np.where((b_coords[:,2] >= z_range[0]) & (b_coords[:,2] <= z_range[1])  # redshifts
                    & (pk_sum[:,11] >= min_mass)                 # Masses
                    & (np.isin(pk_sum[:,0], pk_nums))  )    # Make sure the peak number is one that is wanted

        g_pks = pk_sum[g_idxs]      # List of relevant peak information
        g_coords = b_coords[g_idxs] # list of barycenter coordinates of relevant peaks


        ### DEFINE GAL SAMPLE IN SIGMA-RANGE

        ## FIELD (no lower limit)
        if pk_folder != 'None': 

            bad_Vol = 0  # Volume of potential structures in field
            test_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1]) )[0] # Gals in redshift range

            # Loop through each peak, find max mass, and remove gals from field if they're in a protostructure
            for pk in g_pks:
                pk_gal_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1])    # In relevant redshift
                            & (self.pks[sig_range[0]] == pk[0]) )[0]   
                
                # If max n-sigma is greater than max sigma entered, remove all these galaxies from the field
                if len(pk_gal_idxs) != 0:
                    try:    # If non-single-digit peak (e.g., 10, 25, 102, etc)
                        p_data = np.genfromtxt(pk_path+pk_folder+"\\"+f"pixels_{int(pk[0])}.dat", comments = '#')
                    except: # If single digit peak (e.g., 1, 2, 3, etc)
                        p_data = np.genfromtxt(pk_path+pk_folder+"\\"+f"pixels_0{int(pk[0])}.dat", comments = '#')

                    if np.max(p_data[:,3]) >= sig_range[1]:
                        bad_Vol += pk[10]   # Add volume of peak to bad volume
                        # Remove all the indices from the list of good indices
                        bad_ids = np.in1d(test_idxs, pk_gal_idxs)
                        test_idxs = np.delete(test_idxs, bad_ids)
        
        ## UPPER BOUND PROVIDED (finite sigma-interval)
        elif sig_range[1] != np.inf: 
            mask = np.in1d(self.pks[sig_range[0]], g_pks[:,0])   # Make mask for if peak number is in relevant peak
            test_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1])    # In relevant redshift
                            & (self.n_sigmas >= sig_range[0]) & (self.n_sigmas < sig_range[1]) # In relevant overdensity regime
                            & mask )[0]    # In relevant peak
            
        ## NO UPPER BOUND (half-open sigma-interval)
        else:   
            mask = np.in1d(self.pks[sig_range[0]], g_pks[:,0])   # Make mask for if peak number is in relevant peak

            test_idxs = np.where((self.coords[:,2] >= z_range[0]) & (self.coords[:,2] <= z_range[1])    # In relevant redshift
                            & (self.n_sigmas >= sig_range[0])   # In relevant overdensity
                            & mask )[0]    # In relevant peak
            
        ### SAVE SAMPLE
        self.subpops[key_name] = np.isin(range(len(self.IDs)), test_idxs )   # Saves the bools for which indices are in the subpop

        ### FIND VOLUMES

        ## Not the field
        if sig_range[0] != -99:
            rel_sig = int(not(sig_range[1] == np.inf))  # 0 if upper limit is inf, 1 otherwise
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
            g_gal_otype = self.obs_type[test_idxs]
            bad_gals = np.delete(self.coords, test_idxs, axis=0)   # Galaxies which aren't used.
            bad_gals = bad_gals[np.where((bad_gals[:,2] >= z_range[0]) & (bad_gals[:,2] <= z_range[1]))]    # In relevant redshift

            ## Find plot peaks
            rel_sig = int(not(sig_range[1] == np.inf))
            unique_peaks = np.unique(self.pks[sig_range[rel_sig]][test_idxs]) # Find peak numbers that have a galaxy in them
            p_ids = np.in1d(g_pks[:,0], unique_peaks)


            self.subPop_plot(key_name, sig_range, z_range, g_coords[p_ids], g_pks[p_ids], good_gals, bad_gals, g_gal_otype, plot)


    # ====================================================================



    def subPop_plot(self, key_name, sig_range, z_range, g_coords, g_pks, good_gals, bad_gals, otype, plot):
        """
        Helper method for plotting data from the subPop
        """
        ra_range = [np.min(self.coords[:,0]), np.max(self.coords[:,0])]
        dec_range = [np.min(self.coords[:,1]), np.max(self.coords[:,1])]
        # ra_range = (149.5, 150.6)  
        # dec_range = (1.7, 2.8)


        style_dict = {          # Dictionary of all the styles
            12: ['*', 'gold', "[12,12.5)"], 12.5: ['h', 'deepskyblue', "[12.5,13)"], 13: ['o', 'royalblue', "[13,13.5)"],
            13.5: ['X', 'forestgreen', "[13.5,14)"], 14 :['>', 'darkorange', "[14,14.5)"], 14.5 : ['s', 'red', "[14.5,15)"], 
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
        ax00.set(xlim=ra_range, ylim=dec_range, xlabel="RA (deg)", ylabel="Dec (deg)")
        ax00.xaxis.label.set_fontsize(15)
        ax00.yaxis.label.set_fontsize(15)
        for axis in ax00.get_xticklabels() + ax00.get_yticklabels(): axis.set_fontsize(12)
      
        ax00.legend(title = r"$\log_{10}(M_*)\in$", title_fontsize=12)
        ax00.invert_xaxis()
        ax01.set(xlim=z_range, ylim=ra_range, zlim=dec_range, xlabel="Redshift", ylabel="RA (deg)", zlabel="Dec (deg)")
        ax01.set_xticks(np.arange(z_range[0], z_range[1], 0.05))
        ax01.xaxis.label.set_fontsize(15)
        ax01.yaxis.label.set_fontsize(15)
        ax01.zaxis.label.set_fontsize(15)
        for axis in ax01.get_xticklabels() + ax01.get_yticklabels() + ax01.get_zticklabels(): axis.set_fontsize(12)
        ax01.legend(title = r"$\log_{10}(M_*)\in$", title_fontsize=12)
        ax01.set_box_aspect((5,5,3), zoom=1.2)
        ax01.view_init(25)
        ax01.invert_yaxis()
        ax01.invert_xaxis()

        # Row 2 -- Plotting good vs bad galaxies
        row = gs[1].subgridspec(1,2, width_ratios=[1,1.5])
        ax10, ax11 = fig.add_subplot(row[0]), fig.add_subplot(row[1], projection='3d')

        # 2D
        ots = [0,1,2]
        ot_cs = ['tab:blue', 'tab:orange', 'tab:green']
        ot_label = ['COSMOS2020', 'Spectroscopy', 'HST Grism']
        s_gals = 40
        # Field:
        if sig_range[0] == -99: 
            ot_cs = ['tab:green', 'tab:green', 'tab:green']
            s_gals = 20
            ax10.scatter(good_gals[:,0], good_gals[:,1], marker='.', c='g', alpha=0.15) # Usable galaxies
            ax10.scatter(bad_gals[:,0], bad_gals[:,1], marker='.', c='r', label=f"{len(bad_gals)} Excluded Galaxies") # Unusable galaxies
        else:
            ax10.scatter(bad_gals[:,0], bad_gals[:,1], marker='.', c='r', alpha=0.15, label=f"{len(bad_gals)} Excluded Galaxies") # Unusable galaxies
            for idx in range(len(ots)):
                plot_ids = np.where(otype==ots[idx])
                if len(plot_ids[0])>0:
                    plt_coords = good_gals[plot_ids]
                    ax10.scatter(plt_coords[:,0], plt_coords[:,1], marker='.', s=s_gals, 
                                c=ot_cs[idx], label=f"{ot_label[idx]} : {len(plot_ids[0])}")

        # 3D
        for idx in range(len(ots)):
            plot_ids = np.where(otype==ots[idx])
            if len(plot_ids[0])>0:
                plt_coords = good_gals[plot_ids]
                ax11.scatter(plt_coords[:,2], plt_coords[:,0], plt_coords[:,1], marker='.', s=s_gals,
                            c=ot_cs[idx], label=f"{ot_label[idx]} : {len(plot_ids[0])}")

        # Clean up plots
        ax10.set(xlim=ra_range, ylim=dec_range, xlabel="RA (deg)", ylabel="Dec (deg)")
        ax10.xaxis.label.set_fontsize(15)
        ax10.yaxis.label.set_fontsize(15)
        for axis in ax10.get_xticklabels() + ax10.get_yticklabels(): axis.set_fontsize(12)
        ax10.invert_xaxis()
        ax10.legend()
        ax11.set(xlim=z_range, ylim=ra_range, zlim=dec_range, xlabel="Redshift", ylabel="RA (deg)", zlabel="Dec (deg)")
        ax11.set_xticks(np.arange(z_range[0], z_range[1], 0.05))
        ax11.xaxis.label.set_fontsize(15)
        ax11.yaxis.label.set_fontsize(15)
        ax11.zaxis.label.set_fontsize(15)
        for axis in ax11.get_xticklabels() + ax11.get_yticklabels() + ax11.get_zticklabels(): axis.set_fontsize(12)
        ax11.set_box_aspect((5,5,3), zoom=1.2)
        ax11.view_init(25)
        ax11.legend(title="Source of Redshift", title_fontsize=12, loc= 'upper right')

        ax11.invert_yaxis()
        ax11.invert_xaxis()
    

        # Show/save plots
        if plot in ("show", "Show"): plt.show()
        else:
            if sig_range[1] >0:
                if sig_range[0]==-99:
                    try: plt.savefig(plot + f"\Sigma_{sig_range[0]}_{sig_range[1]}_plot{key_name}.png")
                    except:
                        try:
                            os.mkdir(plot)
                            plt.savefig(plot + f"\Sigma_{sig_range[0]}_{sig_range[1]}_plot{key_name}.png")
                        except: print("Unable to make plot")

                else:
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
                - NOTE: each element of this array can be either a single key, or a list of keys if combining subPops for one SMF
            - smf_keys  (array) = key names in the self.smfs dictionary to store the smf info
                - NOTE: Info is stored as an array of [[m, N, n, n_error], ...] for each key.
            - m_range (array) = mass-range to generate the SMF for
                - [min_mass, max_mass, m_step]
        """
        for sk, k in zip(subPop_keys, smf_keys):

            # If combining multiple subpops for one SMF, combine those
            if self.isIt(sk):
                masses = self.ms[self.subpops[sk[0]]]   # initialize list of masses
                sfrs = self.SFRs[self.subpops[sk[0]]]   # initialize list of masses
                vol = self.vols[sk[0]]      # initialize volume
                for sk_sub in sk[1:]:   # loop through remaining keys and combine masses and volume
                    masses = np.concatenate((masses, self.ms[self.subpops[sk_sub]] ))
                    sfrs = np.concatenate((sfrs, self.SFRs[self.subpops[sk_sub]] ))
                    vol += self.vols[sk_sub]

            else:   # if just looking at one key
                masses = self.ms[self.subpops[sk]] 
                sfrs = self.SFRs[self.subpops[sk]] 
                vol = self.vols[sk]

            smf_info = []       # Store smf info for this key as [[m, N, n, n_err], ...]
            mass = m_range[0]   # keep track of mass

            while mass <= m_range[1]:
                smf_mbin = []   # [m, N, n, n_error] for this mass bin

                # Find galaxies in the mass bin
                gals = np.where( (masses >= mass) & (masses < mass + m_range[2]) )[0]

                if len(gals) != 0:  # If there are galaxies in the mass bin
                    smf_mbin.append(np.median(masses[gals]))  # mass
                    smf_mbin.append(len(gals))                  # N
                    smf_mbin.append(len(gals) / vol / m_range[2])  # n
                    smf_mbin.append(np.sqrt(len(gals)) / len(gals) / vol / m_range[2])  # n_error
                    smf_mbin.append(np.median(sfrs[gals] / 10**masses[gals]))
                else:   # No galaxies in the mass bin
                    smf_mbin.append(mass)       
                    smf_mbin.append(0)
                    smf_mbin.append(0)
                    smf_mbin.append(np.nan)
                    smf_mbin.append(np.nan)

                mass += m_range[2]  # Step up to new mass bin
                smf_info.append(smf_mbin)

            self.smfs[k] = np.array(smf_info)     # Add the info for the current key


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
        colors = ['tab:purple', 'tab:blue','tab:green', 'tab:orange', 'tab:red']  
        shapes = ["o", "H", "s", "D", (5,1,0)]
        min_m, max_m = 99, -99  # For plot limits
        if len(smf_keys) > 0:
            for i, k in enumerate(smf_keys):
                try:
                    plt.errorbar(self.smfs[k][:,0], self.smfs[k][:,2], self.smfs[k][:,3], 
                                label=smf_labels[i], marker=shapes[i], color=colors[i], ls='')
                    min_m = np.min(min_m, min(self.smfs[k][:,0]))
                    max_m = np.max(max_m, max(self.smfs[k][:,0]))
                except: pass
                # except: print(f"The key {k} is not in self.smfs. Generate the smf and run again")
            # Fix up mass limits
            min_m = 9.5 if min_m==99 else min_m-0.2
            max_m = 12 if max_m==-99 else max_m+0.5
        if len(fit_keys) > 0:
            for i, k in enumerate(fit_keys):
                try:    
                    params = self.fits[k]
                except: print(f"The key {k} is not in self.fits. Generate the fit and run again")
                m_vals = np.linspace(min_m, max_m, 1000)

                if len(params) == 3:    # Single Schechter fit
                    plt.plot(m_vals, self.schechter(m_vals, params[0], params[1], params[2]), color=colors[i], marker='', label=fit_labels[i])
                else:    # Double Schechter fit
                    plt.plot(m_vals, self.Dschechter(m_vals, *params), color=colors[i], marker='', label=fit_labels[i])
        
        
        plt.yscale("log")
        plt.ylim(10**-6, 10**-1)
        if (len(smf_keys)==0) or (len(fit_keys)==0): plt.legend(loc='lower left')
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
        colors = ['tab:purple', 'tab:blue','tab:green', 'tab:orange', 'tab:red']  
        shapes = ["o", "H", "P", "s", "D", (5,1,0)]
        min_m, max_m = 99, -99  # For plot limits
        min_n, max_n = 0.2, 1
        try:
            norms = self.smfs[base_key][:,2]
        except:
            print(f"The key {base_key} is not in self.smfs. Generate the smf and run again")
            return
        
        for idx, k in enumerate(smf_keys):
            try:
                data = self.smfs[k]
                # Set length of normalizing array
                if len(norms) > len(data): 
                    norms = norms[:len(data)]   # Shorten normalizing array
                else:   # Shorten data
                    data = data[:len(norms)]
                plt.errorbar(data[:,0], data[:,2] / norms, data[:,3], 
                                label=smf_labels[idx], marker=shapes[idx], color=colors[idx])
                
                min_m = min(min_m, min(data[:,0]))
                max_m = max(max_m, max(data[:,0]))
                min_n = min(min_n, min(data[:,2]/norms))
                max_n = max(max_n, max(data[:,2]/norms))
            except: print(f"The key {k} is not in self.smfs. Generate the smf and run again")



        plt.legend(loc='lower left')
        if title=="": plt.title("SMF") 
        else: plt.title(title)
        plt.ylabel(rf"$\rm \phi/\phi_{base_label}$", fontsize=15)
        plt.xlabel(r"$\rm \log_{10}(M_*/M_\odot)$", fontsize=15)    
        plt.xlim(min_m, max_m)
        plt.ylim(min_n-0.2, max_n+ 0.5)

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
            
            # Find valid data points in the SMFs
            g_idxs = np.where(np.isnan(self.smfs[s_key][:,2]) == False)
            smf_data =  self.smfs[s_key][g_idxs]
            
            if Ns == 1: fit_fn = self.schechter
            elif Ns == 2:  fit_fn = self.Dschechter
            else:   
                print("Elements of N_schechter must be either 1 or 2")
                return

            f_params, f_err = curve_fit(fit_fn, smf_data[:,0], smf_data[:,2], sigma=smf_data[:,3], **kwargs)
            self.fits[f_key] = f_params





    
    def schechter(self, M, M_star, phi_star, alpha):
        """
        Single-Schecter Function
        """
        return np.log(10)*phi_star*10**((M-M_star)*(1+alpha))*np.exp(-10**(M-M_star))
    
    def Dschechter(self, M, M_star, phi_s1, phi_s2, alpha_1, alpha_2):
        """
        Double-Schechter function
        """
        return np.log(10)*np.exp(-10**(M-M_star))*(phi_s1*(10**(M-M_star))**(alpha_1+1) \
        +phi_s2*(10**(M-M_star))**(alpha_2+1) )




    # ====================================================================


    def popPlot(self, sp_key, xlims, ylims, zlims):

        ots = [0,1,2]
        ot_cs = ['tab:blue', 'tab:orange', 'tab:green']
        ot_label = ['COSMOS2020', 'Spectroscopy', 'HST Grism']
        s_gals = 40

        g_ids = self.subpops[sp_key]

        # Sample data: replace with your actual data
        RA = self.coords[:,0][g_ids]  # Right Ascension
        Dec = self.coords[:,1][g_ids]  # Declination
        redshift =self.coords[:,2][g_ids]  # Redshift

        fig = plt.figure(figsize=(8,18))
        ax = fig.add_subplot(111, projection='3d')


        for idx in range(len(ots)):
            plot_ids = np.where(self.obs_type[g_ids]==ots[idx])
            if len(plot_ids[0])>0:
                ax.scatter(RA[plot_ids], Dec[plot_ids], redshift[plot_ids], marker='.', s=s_gals,
                            c=ot_cs[idx], label=f"{ot_label[idx]} : {len(plot_ids[0])}")

                ax.scatter(RA[plot_ids], Dec[plot_ids],zlims[0], marker='.', s=s_gals/2,
                            c=ot_cs[idx],alpha=0.35)
                
                ax.scatter(RA[plot_ids], ylims[0], redshift[plot_ids], marker='.', s=s_gals/2,
                
                            c=ot_cs[idx],alpha=0.35)
        # Plot the projection (shadow) on the x-y plane at z=0

        # Set labels
        ax.set_xlabel('RA', fontsize=15)
        ax.set_ylabel('Dec', fontsize=15)
        ax.set_zlabel('Redshift', fontsize=15)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        ax.legend(title="Source of Redshift", title_fontsize=12)
        # ax.invert_xaxis()

        # Adjust the viewing angle: set the elevation (elev) close to 0
        ax.set_box_aspect(aspect=(1,1,2), zoom=1.2)
        ax.view_init(elev=25, azim=30)  # Adjust azim for a better view

        plt.show()


    def popPlot2(self, sp_key, xlims, ylims, zlims, az, el):
        ots = [0,1,2]
        ot_cs = ['tab:blue', 'tab:orange', 'tab:green']
        ot_label = ['COSMOS2020', 'Spectroscopy', 'HST Grism']
        s_gals = 40

        g_ids = self.subpops[sp_key]

        # Sample data: replace with your actual data
        RA = self.coords[:,0][g_ids]  # Right Ascension
        Dec = self.coords[:,1][g_ids]  # Declination
        redshift =self.coords[:,2][g_ids]  # Redshift

        fig = plt.figure(figsize=(8,18))
        ax = fig.add_subplot(111, projection='3d')

        for idx in range(len(ots)):
            plot_ids = np.where(self.obs_type[g_ids]==ots[idx])
            if len(plot_ids[0])>0:
                ax.scatter(RA[plot_ids], Dec[plot_ids], redshift[plot_ids], marker='.', s=s_gals,
                            c=ot_cs[idx], label=f"{ot_label[idx]} : {len(plot_ids[0])}")

        # Contour on x-y plane
        xy = np.vstack([RA, Dec])
        kde = gaussian_kde(xy)
        xgrid = np.linspace(xlims[0], xlims[1], 500)
        ygrid = np.linspace(ylims[0], ylims[1], 500)
        X, Y = np.meshgrid(xgrid, ygrid)
        Z = kde(np.vstack([X.ravel(), Y.ravel()])).reshape(X.shape)

        ax.contour(X, Y, Z, levels=10, cmap='spring', offset=zlims[0], zdir='z')  # Ensure the offset aligns with the z-axis limit

        # Contour on x-z plane
        xz = np.vstack([RA, redshift])
        kde = gaussian_kde(xz)
        xgrid = np.linspace(xlims[0], xlims[1], 500)
        zgrid = np.linspace(zlims[0], zlims[1], 500)
        X, Z = np.meshgrid(xgrid, zgrid)
        Y = kde(np.vstack([X.ravel(), Z.ravel()])).reshape(X.shape)

        ax.contour(X, Y, Z, levels=10, cmap='autumn', offset=ylims[0], zdir='y')  # Ensure the offset aligns with the y-axis limit
        ax.set_xlabel('\nRA', fontsize=15)
        ax.set_ylabel('\nDec', fontsize=15)
        ax.set_zlabel('\nRedshift', fontsize=15)
        ax.set_xlim(xlims)
        ax.set_ylim(ylims)
        ax.set_zlim(zlims)
        ax.set_xticks(np.linspace(xlims[0], xlims[1], num=5))

        ax.legend(title="Source of Redshift", title_fontsize=12)

        # Adjust the viewing angle
        ax.set_box_aspect(aspect=(1,1,2), zoom=1.2)
        ax.invert_xaxis()
        ax.view_init(elev=el, azim=az)  # Experiment with different elevation and azimuth angles

        plt.show()






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
        if (len(self.voxels) == 0) or (np.all(np.isnan(self.voxels))):
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
        if (len(self.voxels) == 0) or (np.all(np.isnan(self.voxels))):
            self.update_Voxels(sig_cube)

        ## Update overdensities if needed
        if (len(self.ODs) == 0) or (np.all(np.isnan(self.ODs))):
            self.update_ODs(sig_cube)
        
        if self.verbose: print("Updating n_sigmas values (self.n_sigmas)")

        ## Calculate Overdensities and assign to attribute
        self.n_sigmas  = (self.ODs - sig_cube[11].data[self.voxels[:,2]]) / sig_cube[12].data[self.voxels[:,2]]    # (od - mean) / sigma
        

    # ====================================================================

    def del_objs(self, IDs):
        """
        Given some IDs in self.IDs, this deletes the corresponding element of all applicable attributes 
            - NOTE: This can make some things no longer applicable (such as stored SMFs, volumes, etc)
        """
        idxs = np.where(np.in1d(self.IDs,IDs))     # Which indices to delete.

        ## Loop through the attributes and combine where possible
        for attr in self.__dict__:

            if attr in ['vols', 'smfs', 'fits']:
                continue

            self_attr = getattr(self, attr)     # Attrs of this obj

            # Delete in non-empty iterables (excluding dictionaries)
            if self.isIt(self_attr) and (len(self_attr)!=0):

                setattr(self, attr, np.delete(self_attr,idxs, axis=0) )

            elif isinstance(self_attr, dict):   # attr is a dict

                temp_dict = {}
                for k in self_attr:
                    temp_dict[k] = np.delete(self_attr[k], idxs)
                setattr(self, attr, temp_dict)


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

            if self.isIt(self_attr): # attr is a list
                combined_attrs[attr] = np.concatenate((self_attr, other_attr))   
            elif isinstance(self_attr, dict):   # attr is a dict
                combined_attrs[attr] = self._combine_dicts(self_attr, other_attr)
        combined_attrs["verbose"] = bool(self_attr and other_attr)  # set verbose option

        combined_instance = GalPop()
        for attr, value in combined_attrs.items():
            setattr(combined_instance, attr, value)
        return combined_instance
    
    # ====================================================================

    
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

    def create_LP_cat(self):

        cat = np.zeros(shape=(len(self.IDs), len(self.mags) + 3))

        cat[:,0] = self.IDs # Update IDs
        
        for i, m in enumerate(self.mags):
            cat[:,1+i] = self.mags[m]   # Update mags and errors

        cat[:,-2] = np.zeros(len(cat))  # Context
        cat[:,-1] = self.coords[:,2]    # Redshift
        
        return cat



    # ====================================================================


    def saveFile(self, path, ex=[]):
        """Save the object to a .npy file at the specified path, excluding specified attributes."""
        data = {key: value for key, value in self.__dict__.items() if key not in ex}
        np.save(path, data)
        if self.verbose:
            print(f"Saved GalPop object to {path}, excluding {ex}")

    # ====================================================================


    @classmethod
    def loadFile(cls, path):
        """Load the object from a .npy file at the specified path."""
        data = np.load(path, allow_pickle=True).item()
        if not isinstance(data, dict):
            raise ValueError("Loaded data is not a dictionary. Ensure the file contains the correct data structure.")
        
        obj = cls()
        obj.__dict__.update(data)
        if obj.verbose:
            print(f"Loaded GalPop object from {path}")
        return obj
 
    # ====================================================================


    def isIt(self, it, ex=[dict, str]):
        """
        Given some object "it", check if it is an Iterable, excluding types in the list "ex"
        """
        result = isinstance(it, Iterable)   # First check if it's iterable
        for t in ex:
            result = result and not isinstance(it, t)       # Check if it is any of the excluded types
        return result



    # ====================================================================
    # ====================================================================



