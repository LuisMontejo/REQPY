"""
Example 3: Direct Biaxial RotDnn Spectral Matching
========================================================================

This script demonstrates the recommended, state-of-the-art approach for 
developing biaxial seismic input for 3D non-linear response history analysis (NRHA).

Methodological Context:
-----------------------
Unlike independent component matching (demonstrated and critiqued in Examples 1 
and 2), this algorithm simultaneously modifies both horizontal components to 
directly target an orientation-independent response spectrum (e.g., RotD100). 
By adjusting the components concurrently using the Continuous Wavelet Transform (CWT), 
this method achieves a precise match to the target envelope without artificially 
inflating the composite intensity. Provided the seed in prpery selected, 
it preserves the original polarization, phase correlation, and directionality 
characteristics of the historic seed record.

Workflow:
---------
1. Data Ingestion: Loads two orthogonal seed components and the target RotD100 spectrum.
2. Biaxial Matching: Calls `generate_rotdnn_compatible_record` to simultaneously 
   adjust both traces using the default 6th-order polynomial baseline correction.
3. Verification Plotting: Generates comprehensive verification plots using 
   `plot_rotdnn_results`. This includes the Directionality Spectrum of Acceleration 
   (DSA) and polar trajectories to definitively confirm that the natural 
   directionality of the record remains intact.
4. Frequency Domain Analysis: Computes and compares the RotDnn Fourier Amplitude 
   Spectra (FAS) and Power Spectral Density (PSD), alongside the Effective FAS (EAS) 
   and Effective PSD, utilizing the robust "smooth last" computational workflow.
5. Output Generation: Saves the frequency/directionality comparison plots and exports 
   the finalized acceleration time histories for both orthogonal components into standard 
   formats (AT2 and single-column text).
   
References:
---------

Montejo, L. A. (2021). "Response spectral matching of horizontal ground 
motion components to an orientation-independent spectrum (RotDnn)."
Earthquake Spectra, 37(2), 1127-1144.https://doi.org/10.1177/8755293020970981
"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from reqpy_M import (
    generate_rotdnn_compatible_record, 
    load_PEERNGA_record, 
    plot_rotdnn_results,
    save_results_as_at2, 
    save_results_as_1col,
    calculate_fas_rotDnn, 
    calculate_psd_rotDnn,
    calculate_eas, 
    calculate_epsd, 
    get_log_freqs,
    plot_rotdnn_fas_psd_comparison, 
    plot_effective_fas_psd_comparison
)

plt.close('all')

# ---------------------------------------------------------------------------
# --- 1. Configuration Parameters
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# File Paths
seed_file_1 = 'RSN175_IMPVALL.H_H-E12140.AT2'             # Seed record comp 1 [g]
seed_file_2 = 'RSN175_IMPVALL.H_H-E12230.AT2'             # Seed record comp 2 [g]
target_file = 'Multi-Period_MCER Spectrum_UPRM_III_C.txt' # Target PSA spectrum 

# Matching Constraints
dampratio = 0.05                             # SDOF damping ratio (5%)
TL1 = 0.05                                   # Lower bound of matching period domain (s)
TL2 = 6.0                                    # Upper bound of matching period domain (s)
nn = 100                                     # Percentile for RotD calculation (100 = RotD100)
targetPSAlimits = (0.9, 1.1)                 # Allowable ratio limits for final match

output_base_name = f"{seed_file_1[:-10]}_{target_file[:-4]}_RotD{nn}"


# ---------------------------------------------------------------------------
# --- 2. Data Ingestion
# ---------------------------------------------------------------------------
logging.info("Loading seed components and target spectrum...")

s1, dt, n1, name1 = load_PEERNGA_record(seed_file_1)
s2, _, n2, name2 = load_PEERNGA_record(seed_file_2)
fs = 1 / dt
nyquist_freq = fs / 2

target_spectrum = np.loadtxt(target_file)
sort_idx = np.argsort(target_spectrum[:, 0])
To = target_spectrum[sort_idx, 0]  
dso = target_spectrum[sort_idx, 1] 
    

# ---------------------------------------------------------------------------
# --- 3. Direct RotDnn Spectral Matching
# ---------------------------------------------------------------------------
logging.info(f"Initiating simultaneous biaxial matching targeting RotD{nn}...")

results = generate_rotdnn_compatible_record(
    s1=s1,
    s2=s2,
    fs=fs,
    T_PSA=To,
    targetPSA=dso,
    nn=nn,
    targetPSAlimits=targetPSAlimits,
    T1PSA=TL1,
    T2PSA=TL2,
    zi=dampratio,
    baseline_method='sixth_order'
)

logging.info("Spectral matching complete.")


# ---------------------------------------------------------------------------
# --- 4. Plot Verification and Directionality
# ---------------------------------------------------------------------------
logging.info("Generating standard verification and directionality plots...")

fig_spec, fig_hist, fig_polar, fig_dir, _ = plot_rotdnn_results(
    results=results,
    targetPSAlimits=targetPSAlimits,
    T1PSA=TL1,
    T2PSA=TL2,
    zi=dampratio,
    plot_directionality=True,
    units='g'
)

hist_filename = f"{output_base_name}_TimeHistories.png"
spec_filename = f"{output_base_name}_Spectra.png"
polar_filename = f"{output_base_name}_Polar.png"
dir_filename = f"{output_base_name}_Directionality.png"

fig_hist.savefig(hist_filename, dpi=300)
fig_spec.savefig(spec_filename, dpi=300)
fig_polar.savefig(polar_filename, dpi=300)
fig_dir.savefig(dir_filename, dpi=300)

logging.info(f"Saved plots to disk.")


# ---------------------------------------------------------------------------
# --- 5. Extract Results for Frequency Domain Analysis
# ---------------------------------------------------------------------------
n = len(results['sc1'])
s1_orig_trunc = s1[:n]
s2_orig_trunc = s2[:n]

s1_scaled = results['s1_scaled']
s2_scaled = results['s2_scaled']
s1_matched = results['sc1']
s2_matched = results['sc2']


# ---------------------------------------------------------------------------
# --- 6. Calculate RotDnn and Effective Spectra (FAS/PSD)
# ---------------------------------------------------------------------------
logging.info("Calculating RotDnn and Effective FAS/PSD for comparison...")

output_freq_vector = get_log_freqs(fmin=0.1, fmax=nyquist_freq, pts_per_decade=50)

# Define common analysis parameters using the "smooth last" workflow
analysis_params = {
    'sample_rate': fs,
    'smoothing_method': 'konno_ohmachi',
    'smoothing_coeff': 20.0,
    'downsample_freqs': output_freq_vector, 
    'smooth_last': True 
}
percentiles_to_calc = [nn] 

# Calculate RotDnn FAS
_, fas_rotd_smooth_orig, fas_rotd_raw_orig = calculate_fas_rotDnn(s1_orig_trunc, s2_orig_trunc, percentiles=percentiles_to_calc, **analysis_params)
_, fas_rotd_smooth_scaled, fas_rotd_raw_scaled = calculate_fas_rotDnn(s1_scaled, s2_scaled, percentiles=percentiles_to_calc, **analysis_params)
_, fas_rotd_smooth_matched, fas_rotd_raw_matched = calculate_fas_rotDnn(s1_matched, s2_matched, percentiles=percentiles_to_calc, **analysis_params)

# Calculate RotDnn PSD
_, psd_rotd_smooth_orig, psd_rotd_raw_orig = calculate_psd_rotDnn(s1_orig_trunc, s2_orig_trunc, percentiles=percentiles_to_calc, **analysis_params)
_, psd_rotd_smooth_scaled, psd_rotd_raw_scaled = calculate_psd_rotDnn(s1_scaled, s2_scaled, percentiles=percentiles_to_calc, **analysis_params)
_, psd_rotd_smooth_matched, psd_rotd_raw_matched = calculate_psd_rotDnn(s1_matched, s2_matched, percentiles=percentiles_to_calc, **analysis_params)

# Calculate Effective FAS (EAS)
_, eas_smooth_orig, eas_raw_orig = calculate_eas(s1_orig_trunc, s2_orig_trunc, **analysis_params)
_, eas_smooth_scaled, eas_raw_scaled = calculate_eas(s1_scaled, s2_scaled, **analysis_params)
_, eas_smooth_matched, eas_raw_matched = calculate_eas(s1_matched, s2_matched, **analysis_params)

# Calculate Effective PSD (EPSD)
_, epsd_smooth_orig, epsd_raw_orig = calculate_epsd(s1_orig_trunc, s2_orig_trunc, **analysis_params)
_, epsd_smooth_scaled, epsd_raw_scaled = calculate_epsd(s1_scaled, s2_scaled, **analysis_params)
_, epsd_smooth_matched, epsd_raw_matched = calculate_epsd(s1_matched, s2_matched, **analysis_params)


# ---------------------------------------------------------------------------
# --- 7. Plot RotDnn FAS/PSD Comparison
# ---------------------------------------------------------------------------
logging.info("Plotting RotDnn FAS/PSD comparison...")

fas_rotd_raw_all = (fas_rotd_raw_orig, fas_rotd_raw_scaled, fas_rotd_raw_matched)
fas_rotd_smooth_all = (fas_rotd_smooth_orig, fas_rotd_smooth_scaled, fas_rotd_smooth_matched)
psd_rotd_raw_all = (psd_rotd_raw_orig, psd_rotd_raw_scaled, psd_rotd_raw_matched)
psd_rotd_smooth_all = (psd_rotd_smooth_orig, psd_rotd_smooth_scaled, psd_rotd_smooth_matched)

fig_rotd = plot_rotdnn_fas_psd_comparison(
    output_freq_vector, nn,
    fas_rotd_raw_all, fas_rotd_smooth_all,
    psd_rotd_raw_all, psd_rotd_smooth_all
)

rotd_filename = f"{output_base_name}_RotD_FAS_PSD_Comparison.png"
fig_rotd.savefig(rotd_filename, dpi=300)


# ---------------------------------------------------------------------------
# --- 8. Plot Effective FAS/PSD Comparison
# ---------------------------------------------------------------------------
logging.info("Plotting Effective FAS/PSD comparison...")

eas_raw_all = (eas_raw_orig, eas_raw_scaled, eas_raw_matched)
eas_smooth_all = (eas_smooth_orig, eas_smooth_scaled, eas_smooth_matched)
epsd_raw_all = (epsd_raw_orig, epsd_raw_scaled, epsd_raw_matched)
epsd_smooth_all = (epsd_smooth_orig, epsd_smooth_scaled, epsd_smooth_matched)

fig_eff = plot_effective_fas_psd_comparison(
    output_freq_vector,
    eas_raw_all, eas_smooth_all,
    epsd_raw_all, epsd_smooth_all
)

eff_filename = f"{output_base_name}_Effective_FAS_PSD_Comparison.png"
fig_eff.savefig(eff_filename, dpi=300)


# ---------------------------------------------------------------------------
# --- 9. Data Export
# ---------------------------------------------------------------------------
logging.info("Exporting matched records to disk...")

# Save Component 1
at2_filepath1 = f"{output_base_name}_Comp1_Matched.AT2"
at2_header1 = {
    'title': f'Matched record from {seed_file_1} (Target: {target_file})',
    'station': name1.split('_comp_')[0] if '_comp_' in name1 else name1,
    'component': f"{name1.split('_comp_')[-1]}-Matched"
}
save_results_as_at2(results, at2_filepath1, comp_key='sc1', header_details=at2_header1)

txt_1col_filepath1 = f"{output_base_name}_Comp1_Matched_1col.txt"
header_1col_1 = (f"Matched acceleration (g), dt={results.get('dt', 0.0):.8f}s\n"
                 f"Original Seed: {name1}\n"
                 f"Target Spectrum: {target_file}\n"
                 f"Data points follow:")
save_results_as_1col(results, txt_1col_filepath1, comp_key='sc1', header_str=header_1col_1)

# Save Component 2
at2_filepath2 = f"{output_base_name}_Comp2_Matched.AT2"
at2_header2 = {
    'title': f'Matched record from {seed_file_2} (Target: {target_file})',
    'station': name2.split('_comp_')[0] if '_comp_' in name2 else name2,
    'component': f"{name2.split('_comp_')[-1]}-Matched"
}
save_results_as_at2(results, at2_filepath2, comp_key='sc2', header_details=at2_header2)

txt_1col_filepath2 = f"{output_base_name}_Comp2_Matched_1col.txt"
header_1col_2 = (f"Matched acceleration (g), dt={results.get('dt', 0.0):.8f}s\n"
                 f"Original Seed: {name2}\n"
                 f"Target Spectrum: {target_file}\n"
                 f"Data points follow:")
save_results_as_1col(results, txt_1col_filepath2, comp_key='sc2', header_str=header_1col_2)

logging.info(f"Script finished. All files successfully exported.")
plt.show()
