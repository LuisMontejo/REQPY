"""
Example 1: Single Component Spectral Matching 

Matches a single component to a target spectrum.
This updated version also computes and plots the FAS and PSD of the
original, scaled, and matched records for comparison, demonstrating the
new analysis functions.

The smoothed spectra are downsampled to a log-spaced vector for plotting.
"""

# Import necessary functions
from reqpy_M import (
    REQPY_single, load_PEERNGA_record, plot_single_results,
    save_results_as_at2, save_results_as_2col, save_results_as_1col,
    calculate_earthquake_fas, calculate_earthquake_psd, get_log_freqs,
    plot_fas_psd_comparison)

import numpy as np
import matplotlib.pyplot as plt
import logging

plt.close('all')
# --- Configuration ---
# Setup basic logging to see output from the module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

seed_file = 'RSN175_IMPVALL.H_H-E12140.AT2'    # Seed record [g]
target_file = 'ASCE7.txt'                     # Target spectrum (T, PSA)
dampratio = 0.05                             # Damping ratio for spectra
TL1 = 0.05                                    # Lower period limit for matching (s)
TL2 = 6.0                                     # Upper period limit for matching (s)
nit_match = 15                                # Number of matching iterations
baseline_correct = True                       # Perform baseline correction?
p_order = -1                                  # Detrending order for baseline (-1 = none)
output_base_name = seed_file[:-4]+'_'+target_file[:-4] # Base name for output files

# --- 1. Load target spectrum and seed record ---

s_orig, dt, npts, eqname = load_PEERNGA_record(seed_file)
fs = 1 / dt
nyquist_freq = fs / 2
   
target_spectrum = np.loadtxt(target_file)
    
sort_idx = np.argsort(target_spectrum[:, 0])
To = target_spectrum[sort_idx, 0]  # Target spectrum periods
dso = target_spectrum[sort_idx, 1] # Target spectrum PSA

# --- 2. Perform Spectral Matching ---
results = REQPY_single(
    s=s_orig,
    fs=fs,
    dso=dso,
    To=To,
    T1=TL1,
    T2=TL2,
    zi=dampratio,
    nit=nit_match,
    baseline=baseline_correct,
    porder=p_order)

print("Spectral matching complete.")
print(f"Final RMSE (pre-BC): {results['rmsefin']:.2f}%")
print(f"Final Misfit (pre-BC): {results['meanefin']:.2f}%")

# --- 3. Extract Results & Create Scaled Record ---
ccs = results['ccs']
cvel = results['cvel']
cdespl = results['cdespl']
sf = results['sf']
s_scaled = s_orig[:len(ccs)] * sf # Create scaled record for comparison

# --- 4. Plot Original Time History and Response Spectra ---

fig_hist, fig_spec = plot_single_results(
    results=results,
    s_orig=s_orig,
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
plt.show() # Display plots

# --- 5. NEW: Calculate FAS and PSD for all records ---
print("Calculating FAS and PSD for comparison...")

# --- Define a log-spaced frequency vector for smoothed output ---
output_freq_vector = get_log_freqs(fmin=0.1, fmax=nyquist_freq, pts_per_decade=100)

# Define common analysis parameters
analysis_params = {
    'sample_rate': fs,
    'smoothing_method': 'konno_ohmachi', # 'none', 'konno_ohmachi', 'variable_window'
    'smoothing_coeff': 20.0,
    'downsample_freqs': output_freq_vector 
}

# Calculate FAS (capturing both raw and smooth outputs)
freqs_fas_raw, fas_orig_raw, _, fas_orig_smooth = calculate_earthquake_fas(s_orig, **analysis_params)
_, fas_scaled_raw, _, fas_scaled_smooth = calculate_earthquake_fas(s_scaled, **analysis_params)
_, fas_matched_raw, _, fas_matched_smooth = calculate_earthquake_fas(ccs, **analysis_params)

# Calculate PSD (capturing both raw and smooth outputs)
freqs_psd_raw, psd_orig_raw, _, _, _, psd_orig_smooth = calculate_earthquake_psd(s_orig, **analysis_params)
_, psd_scaled_raw, _, _, _, psd_scaled_smooth = calculate_earthquake_psd(s_scaled, **analysis_params)
_, psd_matched_raw, _, _, _, psd_matched_smooth = calculate_earthquake_psd(ccs, **analysis_params)

# --- 6. NEW: Plot FAS and PSD Comparison ---
print("Plotting FAS/PSD comparison...")

fig_fas, fig_psd = plot_fas_psd_comparison(
    # Raw FAS data
    freqs_fas_raw=freqs_fas_raw,
    fas_orig_raw=fas_orig_raw,
    fas_scaled_raw=fas_scaled_raw,
    fas_matched_raw=fas_matched_raw,
    # Raw PSD data
    freqs_psd_raw=freqs_psd_raw,
    psd_orig_raw=psd_orig_raw,
    psd_scaled_raw=psd_scaled_raw,
    psd_matched_raw=psd_matched_raw,
    # Smoothed data
    output_freq_vector=output_freq_vector,
    fas_orig_smooth=fas_orig_smooth,
    fas_scaled_smooth=fas_scaled_smooth,
    fas_matched_smooth=fas_matched_smooth,
    psd_orig_smooth=psd_orig_smooth,
    psd_scaled_smooth=psd_scaled_smooth,
    psd_matched_smooth=psd_matched_smooth
)

# Save the new plots
fas_filename = f"{output_base_name}_FAS_Comparison.png"
psd_filename = f"{output_base_name}_PSD_Comparison.png"
fig_fas.savefig(fas_filename, dpi=300)
fig_psd.savefig(psd_filename, dpi=300)
print(f"Saved FAS plot to {fas_filename}")
print(f"Saved PSD plot to {psd_filename}")

plt.show()

# --- 7. Save Matched Record ---
print("Saving matched record files...")

# --- Option 1: Save as .AT2 format ---
at2_filepath = f"{output_base_name}_Matched.AT2"
at2_header_details = {
    'title': f'Matched record from {seed_file} (Target: {target_file})',
    'date': '01/01/2025', # Placeholder date
    'station': eqname.split('_comp_')[0] if '_comp_' in eqname else eqname,
    'component': f"{eqname.split('_comp_')[-1]}-Matched"
}
save_results_as_at2(results, at2_filepath, comp_key='ccs', header_details=at2_header_details)

# --- Option 2: Save as 2-column (Time, Accel) .txt file ---
txt_2col_filepath = f"{output_base_name}_Matched_2col.txt"
header_2col = (f"Matched acceleration (g) vs. Time (s)\n"
               f"Original Seed: {eqname}\n"
               f"Target Spectrum: {target_file}\n"
               f"Time (s), Acceleration (g)")
save_results_as_2col(results, txt_2col_filepath, comp_key='ccs', header_str=header_2col)

# --- Option 3: Save as 1-column (Accel) .txt file ---
txt_1col_filepath = f"{output_base_name}_Matched_1col.txt"
header_1col = (f"Matched acceleration (g), dt={results.get('dt', 0.0):.8f}s\n"
               f"Original Seed: {eqname}\n"
               f"Target Spectrum: {target_file}\n"
               f"Data points follow:")
save_results_as_1col(results, txt_1col_filepath, comp_key='ccs', header_str=header_1col)
    
print(f"Saved {at2_filepath}, {txt_2col_filepath}, {txt_1col_filepath}")
print("\nScript finished.")