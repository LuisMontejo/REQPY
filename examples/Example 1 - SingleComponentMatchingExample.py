"""
Example 1: Single Component Spectral Matching (Concise)

Matches a single component to a target spectrum using the refactored module.
Removes logging and defensive error handling around core functions for brevity.
"""

# Import necessary functions f

from reqpy_M import (REQPY_single, load_PEERNGA_record, plot_single_results,
        save_results_as_at2, save_results_as_2col, save_results_as_1col)

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

# --- Load target spectrum and seed record ---

s_orig, dt, npts, eqname = load_PEERNGA_record(seed_file)
fs = 1 / dt
   
target_spectrum = np.loadtxt(target_file)
if target_spectrum.ndim != 2 or target_spectrum.shape[1] != 2:
    raise ValueError("Target file should have two columns (Period, PSA).")
    
sort_idx = np.argsort(target_spectrum[:, 0])
To = target_spectrum[sort_idx, 0]  # Target spectrum periods
dso = target_spectrum[sort_idx, 1] # Target spectrum PSA

# --- Perform Spectral Matching ---
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

# --- Extract Results ---
ccs = results['ccs']
cvel = results['cvel']
cdespl = results['cdespl']

# --- Plot Results ---

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

plt.show() # Display plots

# --- Save Matched Record ---

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
    


print("\nScript finished.")

