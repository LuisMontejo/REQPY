"""
Example: Generating Records Compatible with PSA and Minimum PSD

This script demonstrates the complete workflow:
1.  Loading a seed record and target spectra (PSA and PSD).
2.  Generating a compatible record using 'generate_psa_psd_compatible_record'.
3.  Automatically generating verification plots using 'plot_psa_psd_results'.

Based on:
Montejo, L.A. (2025). "Generation of Response Spectrum Compatible Records 
Satisfying a Minimum Power Spectral Density Function." EER.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

# Import from the unified library
from reqpy_M import (
    generate_psa_psd_compatible_record, 
    load_PEERNGA_record,
    plot_psa_psd_results,
    save_results_as_at2,
    save_results_as_2col,
    save_generation_spectral_outputs # New function
)

import warnings
warnings.filterwarnings("ignore")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
log = logging.getLogger()

plt.close('all')

# =============================================================================
# 1. CONFIGURATION & INPUTS
# =============================================================================

seed_file = 'RSN1546_CHICHI_TCU122-N.AT2'
target_psa_file = 'WUS_M7.5_R75_Frequencies.txt'
target_psd_file = 'WUS_M7.5_R75_SD575_12.50s.txt'
units_label = 'g' # Define units for labeling

target_pga = 1.0        
psd_reduction = 0.7     
psa_limits = (0.9, 1.3) 
damping = 0.05

# =============================================================================
# 2. LOAD DATA
# =============================================================================

log.info("Loading input data...")
s, dt, nt, eqname = load_PEERNGA_record(seed_file)
fs = 1/dt

target_psa_data = np.loadtxt(target_psa_file)
f_psa = target_psa_data[:, 0]
t_psa = target_psa_data[:, 1]

target_psd_data = np.loadtxt(target_psd_file)
f_psd = target_psd_data[:, 0]
t_psd = target_psd_data[:, 2] # PSD column

# =============================================================================
# 3. GENERATE COMPATIBLE RECORD
# =============================================================================

log.info("Starting generation process...")

results = generate_psa_psd_compatible_record(
    s=s,
    fs=fs,
    f_PSA=f_psa,
    targetPSA=t_psa,
    f_PSD=f_psd,
    targetPSD=t_psd,
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    targetPGA=target_pga,
    F1PSA=0.2, F2PSA=50.0,
    F1PSD=0.3, F2PSD=30.0,
    zi=damping,
    BLcorr=True,
    localized=True,
    smoothing_method='konno_ohmachi',
    smoothing_coeff=20.0        
)

log.info("Generation complete.")

# =============================================================================
# 4. PLOTTING & SAVING
# =============================================================================

log.info("Generating verification plots...")

fig_spec, fig_hist = plot_psa_psd_results(
    results=results,
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    F1PSA=0.2, F2PSA=50.0,
    F1PSD=0.3, F2PSD=30.0,
    zi=damping,
    units=units_label
)

fig_spec.savefig("Verification_Spectra.png", dpi=300)
fig_hist.savefig("Verification_TimeHistories.png", dpi=300)
log.info("Saved plots.")

log.info("Saving output data...")
# Save time histories
save_results_as_at2(results, 'Output_Matched_Record.AT2', comp_key='sca')
save_results_as_2col(results, 'Output_Matched_Record_2Col.txt', comp_key='sca')
# Save spectral data (PSA, PSD)
save_generation_spectral_outputs(results, 'Output_Spectra')

log.info("Data saved successfully.")
plt.show()