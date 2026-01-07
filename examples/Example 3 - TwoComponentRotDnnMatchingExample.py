"""
Example 3: Direct RotDnn Component Matching 

Modifies two horizontal components from a historic record simultaneously so that
the resulting RotD100 response spectrum (computed from the pair)
matches the specified RotD100 design/target spectrum.

This updated version also computes and plots the RotDnn FAS, RotDnn PSD,
Effective FAS, and Effective PSD for the original, scaled, and matched pairs
using the recommended "smooth last" workflow.

It plots both the "Raw Percentile" (thin, transparent) and the
"Smoothed Percentile" (thick, solid) to show the effect of smoothing.
"""

# Import necessary functions 
from reqpy_M import (
    REQPYrotdnn, load_PEERNGA_record, plot_rotdnn_results,
    save_results_as_at2, save_results_as_2col, save_results_as_1col,
    calculate_fas_rotDnn, calculate_psd_rotDnn,
    calculate_eas, calculate_epsd, get_log_freqs,
    plot_rotdnn_fas_psd_comparison, plot_effective_fas_psd_comparison)

import numpy as np
import matplotlib.pyplot as plt
import logging
plt.close('all')


# --- 1. Configuration ---
# Setup basic logging to see output from the module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

seed_file_1 = 'RSN175_IMPVALL.H_H-E12140.AT2' # Seed record comp1 [g]
seed_file_2 = 'RSN175_IMPVALL.H_H-E12230.AT2' # Seed record comp2 [g]
target_file = 'ASCE7.txt'                    # Target spectrum (T, PSA)
dampratio = 0.02                             # Damping ratio for spectra
TL1 = 0.05                                   # Lower period limit for matching (s)
TL2 = 6.0                                    # Upper period limit for matching (s)
nit_match = 15
nn = 100                                     # Percentile for RotD (100 = RotD100)
baseline_correct = True
p_order = -1
output_base_name = seed_file_1[:-10]+'_'+target_file[:-4]+'_RotD'+str(nn) # Base name for output files

# --- 2. Load target spectrum and seed record ---

s1, dt, n1, name1 = load_PEERNGA_record(seed_file_1)
s2, _, n2, name2 = load_PEERNGA_record(seed_file_2)
fs = 1 / dt
nyquist_freq = fs / 2

target_spectrum = np.loadtxt(target_file)
sort_idx = np.argsort(target_spectrum[:, 0])
To = target_spectrum[sort_idx, 0]  # Target spectrum periods
dso = target_spectrum[sort_idx, 1] # Target spectrum PSA
    

# --- 3. Perform Direct RotDnn Spectral Matching ---
# Call the REQPYrotdnn function
results = REQPYrotdnn(
    s1=s1,
    s2=s2,
    fs=fs,
    dso=dso,
    To=To,
    nn=nn,
    T1=TL1,
    T2=TL2,
    zi=dampratio,
    nit=nit_match,
    baseline=baseline_correct,
    porder=p_order)

print("Spectral matching complete.")
print(f"Final RMSE (pre-BC): {results.get('rmsefin', 'N/A'):.2f}%")
print(f"Final Misfit (pre-BC): {results.get('meanefin', 'N/A'):.2f}%")

# --- 4. Extract Results & Create Scaled Records ---
n = len(results['scc1'])
s1_orig_trunc = s1[:n]
s2_orig_trunc = s2[:n]
sf = results['sf']
s1_scaled = s1_orig_trunc * sf
s2_scaled = s2_orig_trunc * sf
s1_matched = results['scc1']
s2_matched = results['scc2']


# --- 5. Plot Original Time History and Response Spectra ---
# Call the plotting function for RotDnn results
fig_hist, fig_spec = plot_rotdnn_results(
    results=results,
    s1_orig=s1_orig_trunc, # Pass original unscaled record 1
    s2_orig=s2_orig_trunc, # Pass original unscaled record 2
    target_spec=(To, dso),
    T1=TL1,
    T2=TL2,
    xlim_min=None,
    xlim_max=None)

# Save and show plots
hist_filename = f"{output_base_name}_TimeHistories.png"
spec_filename = f"{output_base_name}_Spectra.png"
fig_hist.savefig(hist_filename, dpi=300)
fig_spec.savefig(spec_filename, dpi=300)
print(f"Saved plots to {hist_filename} and {spec_filename}")
plt.show()

# --- 6. NEW: Calculate RotDnn and Effective Spectra ---
print("Calculating RotDnn and Effective FAS/PSD for comparison...")

# --- NEW: Define a log-spaced frequency vector for smoothed output ---
output_freq_vector = get_log_freqs(fmin=0.1, fmax=nyquist_freq, pts_per_decade=50)

# Define common analysis parameters
analysis_params = {
    'sample_rate': fs,
    'smoothing_method': 'konno_ohmachi',
    'smoothing_coeff': 20.0,
    'downsample_freqs': output_freq_vector, # <-- Pass the output vector
    'smooth_last': True # <-- *** USE "SMOOTH LAST" WORKFLOW ***
}
percentiles_to_calc = [nn] # e.g., [100]

# --- Calculate RotDnn FAS ---
_, fas_rotd_smooth_orig, fas_rotd_raw_orig = calculate_fas_rotDnn(
    s1_orig_trunc, s2_orig_trunc, percentiles=percentiles_to_calc, **analysis_params
)
_, fas_rotd_smooth_scaled, fas_rotd_raw_scaled = calculate_fas_rotDnn(
    s1_scaled, s2_scaled, percentiles=percentiles_to_calc, **analysis_params
)
_, fas_rotd_smooth_matched, fas_rotd_raw_matched = calculate_fas_rotDnn(
    s1_matched, s2_matched, percentiles=percentiles_to_calc, **analysis_params
)

# --- Calculate RotDnn PSD ---
_, psd_rotd_smooth_orig, psd_rotd_raw_orig = calculate_psd_rotDnn(
    s1_orig_trunc, s2_orig_trunc, percentiles=percentiles_to_calc, **analysis_params
)
_, psd_rotd_smooth_scaled, psd_rotd_raw_scaled = calculate_psd_rotDnn(
    s1_scaled, s2_scaled, percentiles=percentiles_to_calc, **analysis_params
)
_, psd_rotd_smooth_matched, psd_rotd_raw_matched = calculate_psd_rotDnn(
    s1_matched, s2_matched, percentiles=percentiles_to_calc, **analysis_params
)

# --- Calculate Effective FAS (EAS) ---
_, eas_smooth_orig, eas_raw_orig = calculate_eas(s1_orig_trunc, s2_orig_trunc, **analysis_params)
_, eas_smooth_scaled, eas_raw_scaled = calculate_eas(s1_scaled, s2_scaled, **analysis_params)
_, eas_smooth_matched, eas_raw_matched = calculate_eas(s1_matched, s2_matched, **analysis_params)

# --- Calculate Effective PSD (EPSD) ---
_, epsd_smooth_orig, epsd_raw_orig = calculate_epsd(s1_orig_trunc, s2_orig_trunc, **analysis_params)
_, epsd_smooth_scaled, epsd_raw_scaled = calculate_epsd(s1_scaled, s2_scaled, **analysis_params)
_, epsd_smooth_matched, epsd_raw_matched = calculate_epsd(s1_matched, s2_matched, **analysis_params)


# --- 7. NEW: Plot RotDnn FAS/PSD Comparison ---
print("Plotting RotDnn FAS/PSD comparison...")

# Package data for the plotting function
fas_rotd_raw_all = (fas_rotd_raw_orig, fas_rotd_raw_scaled, fas_rotd_raw_matched)
fas_rotd_smooth_all = (fas_rotd_smooth_orig, fas_rotd_smooth_scaled, fas_rotd_smooth_matched)
psd_rotd_raw_all = (psd_rotd_raw_orig, psd_rotd_raw_scaled, psd_rotd_raw_matched)
psd_rotd_smooth_all = (psd_rotd_smooth_orig, psd_rotd_smooth_scaled, psd_rotd_smooth_matched)

# Call the new helper function
fig_rotd = plot_rotdnn_fas_psd_comparison(
    output_freq_vector, nn,
    fas_rotd_raw_all, fas_rotd_smooth_all,
    psd_rotd_raw_all, psd_rotd_smooth_all
)

rotd_filename = f"{output_base_name}_RotD_FAS_PSD_Comparison.png"
fig_rotd.savefig(rotd_filename, dpi=300)
print(f"Saved RotD FAS/PSD plot to {rotd_filename}")


# --- 8. NEW: Plot Effective FAS/PSD Comparison ---
print("Plotting Effective FAS/PSD comparison...")

# Package data for the plotting function
eas_raw_all = (eas_raw_orig, eas_raw_scaled, eas_raw_matched)
eas_smooth_all = (eas_smooth_orig, eas_smooth_scaled, eas_smooth_matched)
epsd_raw_all = (epsd_raw_orig, epsd_raw_scaled, epsd_raw_matched)
epsd_smooth_all = (epsd_smooth_orig, epsd_smooth_scaled, epsd_smooth_matched)

# Call the new helper function
fig_eff = plot_effective_fas_psd_comparison(
    output_freq_vector,
    eas_raw_all, eas_smooth_all,
    epsd_raw_all, epsd_smooth_all
)

eff_filename = f"{output_base_name}_Effective_FAS_PSD_Comparison.png"
fig_eff.savefig(eff_filename, dpi=300)
print(f"Saved Effective FAS/PSD plot to {eff_filename}")

plt.show()

# --- 9. Save Matched Records ---
# ... (Saving code remains unchanged) ...
print("Saving matched record files...")

# --- Save Component 1 ---
at2_filepath1 = f"{output_base_name}_Comp1_Matched.AT2"
at2_header1 = {
    'title': f'Matched record from {seed_file_1} (Target: {target_file})',
    'station': name1.split('_comp_')[0] if '_comp_' in name1 else name1,
    'component': f"{name1.split('_comp_')[-1]}-Matched"
}
save_results_as_at2(results, at2_filepath1, comp_key='scc1', header_details=at2_header1)

txt_1col_filepath1 = f"{output_base_name}_Comp1_Matched_1col.txt"
header_1col_1 = (f"Matched acceleration (g), dt={results.get('dt', 0.0):.8f}s\n"
                 f"Original Seed: {name1}\n"
                 f"Target Spectrum: {target_file}\n"
                 f"Data points follow:")
save_results_as_1col(results, txt_1col_filepath1, comp_key='scc1', header_str=header_1col_1)

# --- Save Component 2 ---
at2_filepath2 = f"{output_base_name}_Comp2_Matched.AT2"
at2_header2 = {
    'title': f'Matched record from {seed_file_2} (Target: {target_file})',
    'station': name2.split('_comp_')[0] if '_comp_' in name2 else name2,
    'component': f"{name2.split('_comp_')[-1]}-Matched"
}
save_results_as_at2(results, at2_filepath2, comp_key='scc2', header_details=at2_header2)

txt_1col_filepath2 = f"{output_base_name}_Comp2_Matched_1col.txt"
header_1col_2 = (f"Matched acceleration (g), dt={results.get('dt', 0.0):.8f}s\n"
                 f"Original Seed: {name2}\n"
                 f"Target Spectrum: {target_file}\n"
                 f"Data points follow:")
save_results_as_1col(results, txt_1col_filepath2, comp_key='scc2', header_str=header_1col_2)

print(f"Saved {at2_filepath1}, {txt_1col_filepath1}, {at2_filepath2}, {txt_1col_filepath2}")

print("\nScript finished.")