"""
Example 8: Biaxial RotDnn Matching with Minimum FAS and PSD Compliance
=======================================================================================

This script demonstrates the complete, state-of-the-art methodology for generating 
orientation-independent response spectrum matched records, incorporating an intermediate 
Fourier Amplitude Spectrum (FAS) adjustment stage.

Methodological Context:
-----------------------
As proposed in Montejo (2026), satisfying a target PSD through direct energy injection 
can occasionally cause localized disruptions in the temporal envelope of the seed record. 
To preserve the natural phase and temporal evolution of the earthquake, this script 
implements a three-stage correction process on the orthogonal components simultaneously:

    Stage I:   Standard Biaxial RotDnn PSA match.
    Stage II:  Intermediate minimum FAS adjustment (to smoothly raise the baseline 
               frequency content and limit violent phase alterations).
    Stage III: Final minimum PSD adjustment (closing the remaining compliance gap).

   
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

output_base_name = f"Example8_{seed_file_1[:-10]}_RotD100_FAS_PSD"

# Matching Constraints
damping = 0.05
rotd_percentile = 100             # Target RotD100
psa_limits = (0.9, 1.3)           # US-NRC allows -10% to +30% for PSA
fas_reduction = 0.84              # Minimum required FAS threshold (~sqrt(0.7))
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

# Load PSA targets 
t_psa_data = np.loadtxt(target_psa_file)
f_psa = t_psa_data[:, 0]
t_psa = t_psa_data[:, 1]

# Load FAS and PSD targets from the combined file
t_psd_data = np.loadtxt(target_psd_file)
f_psd = t_psd_data[:, 0]
t_fas = t_psd_data[:, 1]  # Column 1: FAS_Mean
t_psd = t_psd_data[:, 4]  # Column 4: PSD_Mean


# ---------------------------------------------------------------------------
# --- 3. Advanced RotDnn Spectral Matching (PSA + FAS + PSD)
# ---------------------------------------------------------------------------
logging.info(f"Initiating 3-stage biaxial matching (I: RotD{rotd_percentile} PSA, II: FAS, III: PSD)...")

results = generate_rotdnn_psa_fas_psd_compatible_record(
    s1=s1, s2=s2, fs=fs,
    f_PSA=f_psa, targetPSA=t_psa,
    f_PSD=f_psd, targetPSD=t_psd,
    f_FAS=f_psd, targetFAS=t_fas,
    nn=rotd_percentile,
    adjustment_mode='both',         # CRITICAL: Trigger the full 3-stage process
    targetPSAlimits=psa_limits,
    PSDreduction=psd_reduction, 
    FASreduction=fas_reduction,
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
    FASreduction=fas_reduction,
    F1PSA=0.2, F2PSA=50.0,
    F1Check=0.3, F2Check=30.0,
    zi=damping,
    units=units_label,
    plot_directionality=True,
    polar_freqs=[0.5, 1, 2, 4, 8, 12, 16, 20]
)

# Save figures
figures[0].savefig(f'{output_base_name}_Spectra.png', dpi=300)
figures[1].savefig(f'{output_base_name}_TimeHistories.png', dpi=300)
figures[2].savefig(f'{output_base_name}_Polar.png', dpi=300)
figures[3].savefig(f'{output_base_name}_Directionality.png', dpi=300)

logging.info("Exporting matched spectral data and time histories...")
save_generation_spectral_outputs(results, prefix=output_base_name)

at2_header1 = {
    'title': f'Matched record from {seed_file_1} (Target: {target_psa_file})',
    'station': name1.split('_comp_')[0] if '_comp_' in name1 else name1,
    'component': f"{name1.split('_comp_')[-1]}-RotD100-FAS-PSD-Matched"
}
save_results_as_at2(results, f'{output_base_name}_H1_Matched.AT2', comp_key='sca1', header_details=at2_header1)

at2_header2 = {
    'title': f'Matched record from {seed_file_2} (Target: {target_psa_file})',
    'station': name2.split('_comp_')[0] if '_comp_' in name2 else name2,
    'component': f"{name2.split('_comp_')[-1]}-RotD100-FAS-PSD-Matched"
}
save_results_as_at2(results, f'{output_base_name}_H2_Matched.AT2', comp_key='sca2', header_details=at2_header2)

logging.info("Script finished successfully.")
plt.show()