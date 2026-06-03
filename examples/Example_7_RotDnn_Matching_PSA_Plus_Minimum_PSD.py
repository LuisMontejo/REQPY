"""
Example 7: Biaxial RotDnn Matching with Minimum Power Spectral Density (PSD) Compliance
=======================================================================================

This script demonstrates the integration of simultaneous biaxial matching with targeted 
frequency-domain energy injection, ensuring the resulting orthogonal acceleration histories 
comply with both an orientation-independent Target Response Spectrum (RotD100) and a 
minimum Target Power Spectral Density (PSD) function.

Methodological Context:
-----------------------
This approach represents the modern standard for developing 3D non-linear structural 
input motions. By adjusting both components concurrently using the Continuous Wavelet 
Transform (CWT), the algorithm achieves a precise RotDnn match while preserving the 
original polarization and directionality of the seed record. 

Subsequently, it enforces PSD compliance (Stage III) to prevent the "energy depletion" 
often caused by strict PSA matching, ensuring adequate power is maintained for secondary 
systems without artificially inflating the target envelope.

Workflow:
---------
1. Data Ingestion: Loads the biaxial seed components and extracts the RotD100 targets 
   (PSA and PSD) from their respective frequency-domain text files.
2. Advanced RotDnn Matching: Sequentially matches the RotD100 PSA (Stage I) and 
   the RotDnn PSD (Stage III) using `adjustment_mode='psd'`.
3. Verification Plotting: Uses `plot_rotdnn_psa_psd_fas_results` to generate the 
   spectral and kinematic comparisons, along with the optional Directionality 
   Spectrum of Acceleration (DSA) and polar plots to verify that natural directionality 
   was maintained throughout the PSD adjustment process.
4. Output Generation: Saves the generated figures and exports the final matched 
   acceleration histories for both orthogonal components.
   
References:
-----------
Montejo, L.A. (2026). "Generation of Orientation-Independent Response Spectrum 
Matched Records Satisfying Minimum Fourier Amplitude and Power Spectral Density 
Requirements." https://doi.org/10.31223/X5Z49W

Montejo, L. A. (2026). Generation of Fourier Amplitude Spectra and Power Spectral 
Density Functions Compatible with Orientation-Independent Design Spectra for 
Bidirectional Seismic Analyses of Nuclear Facilities. Nuclear Engineering and 
Technology, 104136. https://doi.org/10.1016/j.net.2026.104136
    
"""

import numpy as np
import matplotlib.pyplot as plt
import logging
import warnings

from reqpy_M import (
    generate_rotdnn_psa_fas_psd_compatible_record, 
    load_PEERNGA_record,
    plot_rotdnn_psa_psd_fas_results,
    save_results_as_at2,
    save_generation_spectral_outputs
)

warnings.filterwarnings("ignore")
plt.close('all')

# ---------------------------------------------------------------------------
# --- 1. Configuration Parameters
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# File Paths
seed_file_1 = 'RSN933_BIGBEAR_SEA000.AT2'
seed_file_2 = 'RSN933_BIGBEAR_SEA090.AT2'

target_psa_file = 'BSSA14_M7_VS400_RJB50_RotD100.txt' 
target_psd_file = 'BSSA14_M7_VS400_RJB50_RotD100_SD10.4s_TargetFAS_PSD.txt'

output_base_name = f"Example7_{seed_file_1[:-10]}_RotD100_PSD"

# Matching Constraints
damping = 0.05
rotd_percentile = 100             # Target RotD100
psa_limits = (0.9, 1.3)           # US-NRC allows -10% to +30% for PSA
psd_reduction = 0.7               # Minimum required PSD threshold
units_label = 'g'


# ---------------------------------------------------------------------------
# --- 2. Data Ingestion
# ---------------------------------------------------------------------------
logging.info("Loading biaxial seed records and target spectra...")

s1, dt, _, name1 = load_PEERNGA_record(seed_file_1)
s2, _, _, name2 = load_PEERNGA_record(seed_file_2)
fs = 1 / dt

# Truncate to match lengths for simultaneous processing
n = min(len(s1), len(s2))
s1, s2 = s1[:n], s2[:n]

# Load PSA targets (Column 0: Freq, Column 1: PSA)
t_psa_data = np.loadtxt(target_psa_file)
f_psa = t_psa_data[:, 0]
t_psa = t_psa_data[:, 1]

# Load PSD targets (Column 0: Freq, Column 4: PSD_Mean)
t_psd_data = np.loadtxt(target_psd_file)
f_psd = t_psd_data[:, 0]
t_psd = t_psd_data[:, 4]


# ---------------------------------------------------------------------------
# --- 3. Advanced RotDnn Spectral Matching (PSA + PSD Only)
# ---------------------------------------------------------------------------
logging.info(f"Initiating sequential biaxial matching (Stage I: RotD{rotd_percentile} PSA, Stage III: PSD)...")

results = generate_rotdnn_psa_fas_psd_compatible_record(
    s1=s1, s2=s2, fs=fs,
    f_PSA=f_psa, targetPSA=t_psa,
    f_PSD=f_psd, targetPSD=t_psd,
    f_FAS=f_psd, targetFAS=t_psd,   # Ignored by 'psd' adjustment mode
    nn=rotd_percentile,
    adjustment_mode='psd',          # CRITICAL: Adjust for PSD only
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    F1PSA=0.2, F2PSA=50.0,
    F1Check=0.3, F2Check=30.0,
    zi=damping,
    baseline_method='sixth_order'
)

logging.info("Generation complete.")


# ---------------------------------------------------------------------------
# --- 4. Plot Verification & Data Export
# ---------------------------------------------------------------------------
logging.info("Generating verification and directionality plots...")

figures = plot_rotdnn_psa_psd_fas_results(
    results=results,
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction,
    F1PSA=0.2, F2PSA=50.0,
    F1Check=0.3, F2Check=30.0,
    zi=damping,
    units=units_label,
    plot_directionality=True,
    polar_freqs=[0.5, 1, 2, 4, 8, 12, 16, 20]
)

# Save figures (Spectra, Time Histories, Polar, Directionality)
figures[0].savefig(f'{output_base_name}_Spectra.png', dpi=300)
figures[1].savefig(f'{output_base_name}_TimeHistories.png', dpi=300)
figures[2].savefig(f'{output_base_name}_Polar.png', dpi=300)
figures[3].savefig(f'{output_base_name}_Directionality.png', dpi=300)

logging.info("Exporting matched spectral data and time histories...")
save_generation_spectral_outputs(results, prefix=output_base_name)

at2_header1 = {
    'title': f'Matched record from {seed_file_1} (Target: {target_psa_file})',
    'station': name1.split('_comp_')[0] if '_comp_' in name1 else name1,
    'component': f"{name1.split('_comp_')[-1]}-RotD100-PSD-Matched"
}
save_results_as_at2(results, f'{output_base_name}_H1_Matched.AT2', comp_key='sca1', header_details=at2_header1)

at2_header2 = {
    'title': f'Matched record from {seed_file_2} (Target: {target_psa_file})',
    'station': name2.split('_comp_')[0] if '_comp_' in name2 else name2,
    'component': f"{name2.split('_comp_')[-1]}-RotD100-PSD-Matched"
}
save_results_as_at2(results, f'{output_base_name}_H2_Matched.AT2', comp_key='sca2', header_details=at2_header2)

logging.info("Script finished successfully.")
plt.show()
