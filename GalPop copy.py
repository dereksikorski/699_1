from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from astropy.io import fits
from time import time
import astropy.cosmology
from astropy.cosmology import FlatLambdaCDM
import os
cosmo = FlatLambdaCDM(H0=70, Om0=0.27)
import matplotlib.gridspec as gridspec
from scipy.optimize import curve_fit
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import gaussian_kde
from tqdm.notebook import tqdm
from scipy.spatial import KDTree
from multiprocessing import Pool
from typing import *
import warnings




# --------------------------------------------------------
# --------------------------------------------------------
# --------------------------------------------------------

class GalPop:


    def __init__(self,   IDs: Iterable,   coords: Iterable[Iterable[int]],     ms: Iterable[float],
        *,
        mags: dict[str, Iterable[float]] = None,     verbose: bool = False,    misc: dict = None ) -> None:
        """
        Initialize a population of galaxies.

        Parameters
        ----------
        IDs : Iterable
            Array of unique IDs for each galaxy in the population.
        coords : Iterable[Iterable[int]]
            2D array of [ra, dec, z] for each galaxy. Shape: (# of galaxies, 3).
        ms : Iterable[float]
            Array of galaxy masses.

        Optional Parameters
        ---------

        mags : dict[str, Iterable[float]], optional
            Dictionary of magnitudes with {band_name: array_of_mags}. Defaults to None.
        verbose : bool, optional
            If True, print progress during method calls. Defaults to False.
        misc : dict, optional
            Additional miscellaneous information. Defaults to None.

        Attributes
        ----------
        IDs : Iterable
            Unique galaxy identifiers.
        coords : Iterable[Iterable[int]]
            3D galaxy coordinates.
        ms : Iterable[float]
            Galaxy masses.
        mags : dict[str, Iterable[float]]
            Magnitudes for different bands.
        verbose : bool
            Verbosity of logging.
        misc : dict
            Miscellaneous attributes.

        Internal Attributes
        --------

        voxels : Iterable[Iterable[float]]  --> Derived in update_voxels()
            3D voxel coordinates [x,y,z] for each derived from a given VMC map
        ODs : Iterable[float]   --> Derived in update_ODs()
            Overdensity values for each galaxy
        n_sigmas : Iterable[float]  --> Derived in update_n_sigmas()
            Number of standard deviations above the average overdensity for a given galaxy
        pks : Dict[ (int|float)  ,  Iterable ]  --> Derived in assign_peaks()
            Dictionary of peak numbers for each galaxy given a set of results from find_peaks.py
        subpops : Dict[ (str|float) , Iterable[bool]] --> Derived in make_field() or make_sp()
            Dictionary of bool for each galaxy for if it is in a defined subpopulation or not
        vols : Dict[ (str|float), float]    --> Derived in make_field() or make_sp()
            Dictionary of volume (cMpc^3) a given subpopulation is located within
        """

        ## Mandatory attributes
        self.IDs = IDs  # Galaxy IDs
        self.coords = coords  # Galaxy coords [ [ra1, dec1, z1], [ra2, dec2, z2], ...]
        self.ms = ms  # Mass

        ## Optional attributes
        self.mags = mags or {}  # Magnitude dictionary
        self.verbose = verbose  # Verbose option for printing
        self.misc = misc or {}

        ## Interior Variables
        self.voxels = []  # Voxels of each galaxy
        self.ODs =  []  # Overdensity values
        self.n_sigmas = []  # Sigma above mean overdensity
        self.pks =  {}  # Peak number dictionary
        self.subpops = {}  # Create samples based on sigma-cuts
        self.vols = {}  # Volumes of the sigma cuts
        self.smfs = {}  # SMF information stored as [[m1, N1, n1, n_error1], ...] for each value
        self.fits = {}  # Fits to the SMFs. Data is stored depending on type of fit

        ## Check uniqueness
        if len(np.unique(IDs)) != len(IDs):
            raise ValueError("Non-unique IDs provided. The IDs input must be an array of unique identifiers.")

        if self.verbose:
            print(f"Initialized GalPop object with {len(self.IDs)} galaxies.")

    # ====================================================================
    # ====================================================================
    # ====================================================================



    def assign_peaks(self, sigs : Iterable[float], sig_summary_files : Iterable[str],
                     sig_data_folders : Iterable[str], overwrite : bool = False) -> None:
        """
        Assign galaxies to a peak number given a set of find_peaks outputs. This updates the self.pks dictionary with keys given
        by the sigs input.

        Parameters
        -----------
        sigs : Iterable[float]
            List of sigma values used to defined peaks in the find_peaks outputs. These are used as the keys in self.pks
            - Example: [2, 2.5, 3] would assign every galaxy to peaks defined with thresholds 2, 2.5, and 3 sigma. self.pks
                would then be populated with lists corresponding to the keys 2, 2.5, and 3

        sig_summary_files : Iterable[str]
            Paths to the find_peaks summary files corresponding to each sigma-value in sigs.
        
        sig_data_folders : Iterable[str]
            Paths to the folders containing individual peak informaiton for each sigma-value
        
        overwrite : bool, optional
            If True, will overwrite the contents of self.pks for the given keys in sigs. This method can take sometime, so this is
            purely to avoid accidental deletion of peak information.

        Notes
        -----
        - The method uses spatial KDTree lookups and voxel data to efficiently assign galaxies to peaks.
        - Peak assignment relies on summary files, voxel data, and bounds for each peak.
        - Requires `self.update_n_sigmas()` to be run beforehand to ensure galaxy overdensity data is up to date.
        """

        ## Check existance of sigmas and verify overwrite
        existing_sigmas = [s for s in sigs if s in self.pks ]

        if existing_sigmas and not overwrite:
            existing_str = ", ".join(map(str, existing_sigmas))
            raise ValueError(
                f"Peak assignments already exist for sigma = {existing_str}. "
                f"Use `overwrite=True` to reassign.")

        ## Make sure n-sigmas have been defined
        if len(self.n_sigmas) == 0:  
            raise ValueError("Missing galaxy overdensity information. Run self.update_n_sigmas() first, then re-run.")


        ## Process each sigma level
        for sig, summary_file, data_folder in zip(sigs, sig_summary_files, sig_data_folders):

            # Fill the pk-info with filler data
            self.pks[sig] = [-99] * len(self.voxels)

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

            self.pks[sig] = np.array(assignments)


    # ====================================================================
    # ====================================================================
    # ====================================================================




    def make_subpop(self, 
            subpop_name: str,  field : bool, pk_def: Iterable[float, float, float],
            sig_summary_file: str, sig_data_folder: str, 
            *,
            boundaries: Iterable[Iterable[float]] = None,
            cosmo: astropy.cosmology = None, 
            gal_sample: Iterable[bool] = None):
        """
        Create a subpopulation of galaxies based on peak inclusion or exclusion criteria.

        Parameters
        ----------
        subpop_name : str
            The name of the subpopulation. This will be the key in self.subpops and self.vols.

        field : bool
            Specify whether this is a field sample (i.e. peaks are excluded) or a structure sample (i.e. peaks are included)

        pk_def : Iterable[float, float, float]
            Array defining the peaks ([sigma_def, req_sig, min_mass]):
            - sigma_def = Sigma threshold defining peak (should be a key in self.pks).
            - req_sig = Sigma requirement for a peak to be considered valid.
            - min_mass = Minimum mass requirement [log(M)] for a peak to be valid.

        sig_summary_file : str
            Path to the relevant summary file from `find_peaks.py`.

        sig_data_folder : str
            Path to the folder containing peak information from `find_peaks.py`.

        Optional Parameters
        -------------------
        boundaries : Iterable[Iterable[float]], optional
            Boundaries for the volume calculation ([RA_min, RA_max], [Dec_min, Dec_max], [z_min, z_max]).

        cosmo : astropy.cosmology, optional
            Cosmology for volume calculations. Defaults to FlatLambdaCDM(H0=70.0, Om0=0.27).

        gal_sample : Iterable[bool], optional
            Array specifying which galaxies to consider. Defaults to all galaxies.
        """
        ## Validate inputs
        cosmo = cosmo or FlatLambdaCDM(H0=70.0, Om0=0.27)

        if gal_sample is None:
            gal_sample = np.arange(len(self.IDs))

        sigma_def = pk_def[0]
        if sigma_def not in self.pks:
            raise ValueError(f"Sigma = {sigma_def} not found in self.pks. Run self.assign_peaks with this sigma.")
        

        ## Initialize subpopulation mask
        subpop_mask = np.full(len(self.IDs), False)
        subpop_mask[gal_sample] = True

        # Identify peaks to include or exclude
        peaks, peak_vol = self._identify_peaks(pk_def, sig_summary_file, sig_data_folder, gal_sample)

        # Update subpopulation mask
        subpop_mask = np.logical_and(subpop_mask, ~np.isin(self.pks[sigma_def], peaks) if field  else np.isin(self.pks[sigma_def], peaks))

        # Calculate volume
        volume = self._calculate_volume(boundaries, gal_sample, cosmo, peak_vol) if field else peak_vol

        # Store results
        self.subpops[subpop_name] = subpop_mask
        self.vols[subpop_name] = volume

    # ====================================================================


    def _identify_peaks(self, pk_def: Iterable[float, float, float], 
                        sig_summary_file: str, sig_data_folder: str, 
                        gal_sample: Iterable[int]):
        """
        Helper function for self.make_subpop()\n
        Identify peaks to include or exclude and calculate associated volume.
        """
        sigma_def, req_sig, min_mass = pk_def
        peak_summary = np.genfromtxt(sig_summary_file, dtype=float, skip_header=0)

        # Find peaks with galaxies assigned to them
        peaks = np.unique(self.pks[sigma_def][gal_sample])
        peaks = peaks[peaks >= 0]

        valid_peaks = []
        total_volume = 0

        for peak_id in peaks:
            if peak_summary[peak_id - 1, 11] >= min_mass:  # Check mass of the peak
                try:
                    voxel_data = np.genfromtxt(f"{sig_data_folder}/pixels_{int(peak_id):02d}.dat", comments="#")
                except FileNotFoundError:
                    continue

                if np.max(voxel_data[:, 3]) >= req_sig:  # Check sigma requirement
                    valid_peaks.append(peak_id)
                    total_volume += peak_summary[peak_id - 1, 10]

        return (valid_peaks, total_volume)
    
    # ====================================================================


    def _calculate_volume(self, boundaries: Iterable[Iterable[Iterable[float]]], gal_sample: Iterable[int], cosmo: astropy.cosmology, bad_volume: float):
        """
        Helper function for make_field. \n
        Calculate the total volume for the field, accounting for excluded peaks.
        """
        if boundaries:      # If user has specified boundaries
            ra_min, ra_max,  dec_min, dec_max, z_min, z_max   = boundaries[0], boundaries[1], boundaries[2]

        else:       # If user has not specified boundaries, use relevant galaxies to infer boundaries
            coords = self.coords[gal_sample]
            ra_min, ra_max, dec_min, dec_max, z_min, z_max = np.min(coords[:, 0]), np.max(coords[:, 0]),  np.min(coords[:, 1]), np.max(coords[:, 1]),  np.min(coords[:, 2]), np.max(coords[:, 2])

        # Find Volume
        theta_dec = np.radians(np.abs(dec_max - dec_min))
        theta_ra = np.radians(np.abs(ra_max - ra_min)) * np.cos(theta_dec / 2)
        solid_angle = theta_ra * theta_dec

        volume_cube = (solid_angle / (4 * np.pi)) * (cosmo.comoving_volume(z_max) - cosmo.comoving_volume(z_min))
        return volume_cube.value - bad_volume



    # ====================================================================
    # ====================================================================
    # ====================================================================


    def modify_subpops(self, new_sp : str, base_sp : str, other_sps : Iterable[str], include : bool = True):
            """
            Creates a new subpopulation based on an existing one and a list of other subpopulations. This either combines the populations, or excludes the other 
            subpopulations from the base subpopulation. \n
            Caution! This uses a simple addition or subtraction of volumes. If the subpops are not exclusive in space, the resulting volume may not be reliable

            Parameters:
            ------
            new_name : str
                Name of the new subpopulation to add to self.subpops

            base_name : str
                Name of the subpopulation in self.subpops to serve as a basis for the new subpopulation

            other_sps : Iterable[str] : 
                Names of the subpopulations to include or exclude.

            include : bool
                Whether to include (True) or exclude (False) the other subpopulations from the base subpopulation
            """
            ## Validate inputs

            if base_sp not in self.subpops:
                raise ValueError(f"Base subpopulation '{base_sp}' not found.")
            if not all(name in self.subpops for name in other_sps):
                raise ValueError("One or more specified subpopulations are not found.")
            if not other_sps:
                raise ValueError("No subpopulations specified for inclusion or exclusion.")


            # Start with the base subpopulation
            new_galaxies = self.subpops[base_sp].copy()
            new_volume = self.vols[base_sp]

            # Inclusion:
            if include:
                for name in other_sps:
                    new_galaxies = np.logical_or(new_galaxies, self.subpops[name])      # Check galaxies are in either of the populations
                    new_volume += self.vols[name]

            # Exclusion
            else:
                for name in other_sps:
                    new_galaxies = np.logical_and(new_galaxies, ~self.subpops[name])    # Check galaxies are in base and not in the excluded pop
                    new_volume -= self.vols[name]

            # Caution user if Volume is negative
            if new_volume <= 0:
                warnings.warn("Negative volume found. The resulting volume may not be reliable.")
            
            # Add the new subpopulation
            self.subpops[new_sp] = new_galaxies
            self.vols[new_sp] = new_volume
            



    def make_SMF(self, subPop_keys, smf_keys, m_range):
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







    def update_voxels(self, sig_cube) -> None:
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
            self.update_voxels(sig_cube)
        
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
            self.update_voxels(sig_cube)

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


    def save_file(self, path, ex=[]):
        """Save the object to a .npy file at the specified path, excluding specified attributes."""
        data = {key: value for key, value in self.__dict__.items() if key not in ex}
        np.save(path, data)
        if self.verbose:
            print(f"Saved GalPop object to {path}, excluding {ex}")

    # ====================================================================


    @classmethod
    def load_file(cls, path):
        """
        Load the object from a .npy file at the specified path.
        """
        data = np.load(path, allow_pickle=True).item()
        if not isinstance(data, dict):
            raise ValueError("Loaded data is not a dictionary. Ensure the file contains the correct data structure.")

        # Extract mandatory parameters
        IDs = data.pop("IDs", [])
        coords = data.pop("coords", [])
        ms = data.pop("ms", [])

        # Create a new instance with the mandatory parameters
        obj = cls(IDs, coords, ms)

        # Update remaining attributes
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



