"""
Example 5: Single-Component Matching with Minimum Power Spectral Density (PSD) Compliance
=======================================================================================

This script demonstrates advanced single-component spectral matching, ensuring the 
resulting acceleration history complies with both a Target Response Spectrum (PSA) 
and a minimum Target Power Spectral Density (PSD) function.

Methodological Context:
-----------------------
Standard spectral matching algorithms (like the one used in Example 1) can sometimes 
achieve a tight PSA match by suppressing energy at intermediate frequencies. This 
"energy depletion" can lead to unconservative responses in secondary systems or 
equipment mounted within the structure. 

This script implements the methodology proposed in Montejo (2025), utilizing an 
iterative Continuous Wavelet Transform (CWT) to first match the PSA, followed by a 
targeted injection of wavelet energy to satisfy the PSD threshold. 

*Note on Biaxial Input:* While this script demonstrates the PSD-adjustment algorithm 
on a single component for illustrative purposes, current state-of-practice for 3D 
structural analysis prefers simultaneous biaxial matching (RotDnn). 

Workflow:
---------
1. Data Ingestion: Loads the seed record (AT2) and the frequency-domain targets (PSA, PSD).
2. Advanced Matching: Sequentially matches the PSA (Stage I) and the PSD (Stage III).
3. Verification Plotting: Uses `plot_psa_psd_fas_results` to generate two primary figures:
   - Spectral Comparison (Figure 1): A dynamically sized plot featuring two panels. 
     The top panel tracks the PSA matching against the target bounds, while the bottom 
     panel verifies that the one-sided PSD of the final record rests above the 
     specified minimum reduction limit.
   - Time-Domain Comparison (Figure 2): Displays the acceleration, velocity, and 
     displacement histories. It overlays the scaled seed and final matched record, 
     plotting normalized energy buildups on secondary axes to quantify temporal alteration.
4. Output Generation: Saves the generated figures and exports the matched data to disk.

References:
---------
Montejo, L. A. (2025). Generation of Response Spectrum Compatible Records Satisfying 
a Minimum Power Spectral Density Function. Earthquake Engineering and Resilience, 
4(2), 215-228.  https://doi.org/10.1002/eer2.70008Digital

Montejo, L. A. (2024). Strong-Motion-Duration-Dependent Power Spectral Density
Functions Compatible with Design Response Spectra. Geotechnics, 4(4), 1048-1064.
 https://doi.org/10.3390/geotechnics4040053
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings

from reqpy_M import (
    generate_single_component_psa_fas_psd_compatible_record, 
    load_PEERNGA_record,
    plot_psa_psd_fas_results,
    save_results_as_at2,
    save_results_as_2col,
    save_generation_spectral_outputs
)

warnings.filterwarnings("ignore")
plt.close('all')

# ---------------------------------------------------------------------------
# --- 1. Configuration Parameters
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Original Exact File Names
seed_file = 'RSN933_BIGBEAR_SEA000.AT2'
target_psa_file = 'WUS_M7.5_R75_Frequencies.txt'
target_psd_file = 'WUS_M7.5_R75_SD575_12.50s.txt' 

output_base_name = f"Example5_{seed_file[:-4]}_PSA_PSD"

# Matching Constraints
damping = 0.05
target_pga = 1.0        
psa_limits = (0.9, 1.3) 
psd_reduction = 0.7     
units_label = 'g'


# ---------------------------------------------------------------------------
# --- 2. Data Ingestion
# ---------------------------------------------------------------------------
logging.info("Loading input data...")

s, dt, nt, eqname = load_PEERNGA_record(seed_file)
fs = 1 / dt

# Load PSA targets (Frequency Domain)
target_psa_data = np.loadtxt(target_psa_file)
f_psa = target_psa_data[:, 0]
t_psa = target_psa_data[:, 1]

# Load PSD targets 
target_psd_data = np.loadtxt(target_psd_file)
f_psd = target_psd_data[:, 0]
t_psd = target_psd_data[:, 2] # Extract PSD from column 2


# ---------------------------------------------------------------------------
# --- 3. Advanced Spectral Matching (PSA + PSD Only)
# ---------------------------------------------------------------------------
logging.info("Starting generation process (PSA + PSD)...")

results = generate_single_component_psa_fas_psd_compatible_record(
    s=s,
    fs=fs,
    f_PSA=f_psa,             # Using frequency vector from the target file
    targetPSA=t_psa,
    f_PSD=f_psd,
    targetPSD=t_psd,
    f_FAS=f_psd,             # Required by signature, but ignored by 'psd' mode
    targetFAS=t_psd,         # Required by signature, but ignored by 'psd' mode
    adjustment_mode='psd',   # CRITICAL: Adjust for PSD only
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    targetPGA=target_pga,
    F1PSA=0.2, F2PSA=50.0,
    F1Check=0.3, F2Check=30.0, 
    zi=damping,
    baseline_method='sixth_order'
)

logging.info("Generation complete.")


# ---------------------------------------------------------------------------
# --- 4. Plot Verification & Data Export
# ---------------------------------------------------------------------------
logging.info("Generating verification plots...")

fig_spec, fig_hist, max_deltas = plot_psa_psd_fas_results(
    results=results,
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    F1PSA=0.2, F2PSA=50.0,
    F1Check=0.3, F2Check=30.0,
    zi=damping,
    units=units_label
)

fig_spec.savefig(f"{output_base_name}_Verification_Spectra.png", dpi=300)
fig_hist.savefig(f"{output_base_name}_Verification_TimeHistories.png", dpi=300)

logging.info("Saving output data...")
save_results_as_at2(results, f'{output_base_name}_Matched.AT2', comp_key='sca')
save_results_as_2col(results, f'{output_base_name}_Matched_2Col.txt', comp_key='sca')
save_generation_spectral_outputs(results, f'{output_base_name}_Spectra')

logging.info("Data saved successfully.")
plt.show()