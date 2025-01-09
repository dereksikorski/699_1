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
            


    # ==========================================================================================
    # ==========================================================================================
    # ==========================================================================================




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
    # ====================================================================



