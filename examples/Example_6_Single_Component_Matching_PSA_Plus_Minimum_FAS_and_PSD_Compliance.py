"""
Example 6: Single-Component Matching with Minimum Fourier Amplitude (FAS) and PSD Compliance
============================================================================================

This script extends the methodology from Example 5 by introducing an intermediate 
matching stage to enforce a minimum Fourier Amplitude Spectrum (FAS) before applying 
the final Power Spectral Density (PSD) compliance check.

Methodological Context:
-----------------------
While injecting energy directly to satisfy a target PSD (as seen in Example 5) is 
effective, sudden or large energy injections in the frequency domain can sometimes 
introduce localized disruptions in the phase angle and temporal envelope of the 
seed record. 

To mitigate this we introduce a three-stage approach:
    Stage I:   Standard PSA match.
    Stage II:  Intermediate minimum FAS adjustment (to smoothly raise the 
               baseline frequency content without altering phase violently).
    Stage III: Final minimum PSD adjustment (which now requires much smaller, 
               less disruptive energy injections).

Workflow:
---------
1. Data Ingestion: Loads the seed record and extracts the PSA, FAS, and PSD targets.
2. Advanced Matching: Triggers all three stages sequentially (`adjustment_mode='both'`).
3. Verification Plotting: Uses `plot_psa_psd_fas_results` to generate figures documenting 
   all three stages of the modification process:
   - Spectral Comparison (Figure 1): Dynamically expands to three panels. The top panel 
     tracks the PSA match. The middle panel shows the intermediate Stage II record being 
     lifted to meet the FAS threshold. The bottom panel shows the final Stage III record 
     satisfying the final PSD threshold.
   - Time-Domain Comparison (Figure 2): Plots acceleration, velocity, and displacement, 
     overlaying the original scaled record, the Stage II FAS-adjusted record, and the 
     final Stage III PSD-adjusted record to track the kinematic evolution step-by-step.
4. Output Generation: Saves the generated figures and exports the matched data to disk.

References:
---------
Montejo, L. A. (2025). Generation of Response Spectrum Compatible Records Satisfying 
a Minimum Power Spectral Density Function. Earthquake Engineering and Resilience, 
4(2), 215-228.  https://doi.org/10.1002/eer2.70008Digital

Montejo, L. A. (2024). Strong-Motion-Duration-Dependent Power Spectral Density
Functions Compatible with Design Response Spectra. Geotechnics, 4(4), 1048-1064.
https://doi.org/10.3390/geotechnics4040053
 
Montejo, L. A. (2026). Generation of Orientation-Independent Response Spectrum 
Matched Records Satisfying Minimum Fourier Amplitude and Power Spectral Density 
Requirements

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

output_base_name = f"Example6_{seed_file[:-4]}_PSA_FAS_PSD"

# Matching Constraints
damping = 0.05
target_pga = 1.0        
psa_limits = (0.9, 1.3) 
psd_reduction = 0.7     
fas_reduction = 0.84        # ~sqrt(0.7)
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

# Load FAS and PSD targets from the combined file
target_psd_data = np.loadtxt(target_psd_file)
f_psd = target_psd_data[:, 0]
t_fas = target_psd_data[:, 1] # Extract FAS from column 1
t_psd = target_psd_data[:, 2] # Extract PSD from column 2


# ---------------------------------------------------------------------------
# --- 3. Advanced Spectral Matching (PSA + FAS + PSD)
# ---------------------------------------------------------------------------
logging.info("Starting generation process (PSA + FAS + PSD)...")

results = generate_single_component_psa_fas_psd_compatible_record(
    s=s,
    fs=fs,
    f_PSA=f_psa,
    targetPSA=t_psa,
    f_PSD=f_psd,
    targetPSD=t_psd,
    f_FAS=f_psd,             # Frequencies for FAS are same as PSD
    targetFAS=t_fas,
    adjustment_mode='both',  # CRITICAL: Trigger the 3-stage process (FAS then PSD)
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    FASreduction=fas_reduction,
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
    FASreduction=fas_reduction,
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