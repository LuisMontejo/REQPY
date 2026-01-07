"""
Example: Generating Records Compatible with PSA, Minimum FAS, and Minimum PSD.

This script demonstrates the new functionality for FAS compliance.
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

# Import from the unified library
from reqpy_M import (
    generate_psa_psd_fas_compatible_record, 
    load_PEERNGA_record,
    plot_psa_psd_fas_results,
    save_results_as_at2,
    save_results_as_2col,
    save_generation_spectral_outputs
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
target_psd_file = 'WUS_M7.5_R75_SD575_12.50s.txt' # Contains FAS in col 1, PSD in col 2
units_label = 'g' 

target_pga = 1.0        
psd_reduction = 0.7     
fas_reduction = 0.84 # ~sqrt(0.7)
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
t_fas = target_psd_data[:, 1] # FAS column
t_psd = target_psd_data[:, 2] # PSD column

# =============================================================================
# 3. GENERATE COMPATIBLE RECORD
# =============================================================================

log.info("Starting generation process...")

results = generate_psa_psd_fas_compatible_record(
    s=s,
    fs=fs,
    f_PSA=f_psa,
    targetPSA=t_psa,
    f_PSD=f_psd,
    targetPSD=t_psd,
    f_FAS=f_psd, # Frequencies for FAS are same as PSD
    targetFAS=t_fas,
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    FASreduction=fas_reduction,
    targetPGA=target_pga,
    F1PSA=0.2, F2PSA=50.0,
    F1Check=0.3, F2Check=30.0, # Range for FAS and PSD checks
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

fig_spec, fig_hist = plot_psa_psd_fas_results(
    results=results,
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    FASreduction=fas_reduction,
    F1PSA=0.2, F2PSA=50.0,
    F1Check=0.3, F2Check=30.0,
    zi=damping,
    units=units_label
)

fig_spec.savefig("Verification_Spectra_FAS.png", dpi=300)
fig_hist.savefig("Verification_TimeHistories_FAS.png", dpi=300)
log.info("Saved plots.")

log.info("Saving output data...")
save_results_as_at2(results, 'Output_Matched_Record_FAS_Adj.AT2', comp_key='sca')
save_results_as_2col(results, 'Output_Matched_Record_FAS_Adj_2Col.txt', comp_key='sca')
save_generation_spectral_outputs(results, 'Output_Spectra_FAS_Adj')

log.info("Data saved successfully.")
plt.show()