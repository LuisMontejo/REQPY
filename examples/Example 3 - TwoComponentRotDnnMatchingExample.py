"""
Example 3: Direct RotDnn Component Matching (Concise)

Modifies two horizontal components from a historic record simultaneously so that
the resulting RotD100 response spectrum (computed from the pair)
matches the specified RotD100 design/target spectrum.

This is the recommended approach for matching two components.

"""

# Import necessary functions 

from reqpy_M import (REQPYrotdnn, load_PEERNGA_record, plot_rotdnn_results,
    save_results_as_at2, save_results_as_2col, save_results_as_1col)

import numpy as np
import matplotlib.pyplot as plt
import logging
plt.close('all')


# --- Configuration ---
# Setup basic logging to see output from the module
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

seed_file_1 = 'RSN175_IMPVALL.H_H-E12140.AT2' # Seed record comp1 [g]
seed_file_2 = 'RSN175_IMPVALL.H_H-E12230.AT2' # Seed record comp2 [g]
target_file = 'ASCE7.txt'                    # Target spectrum (T, PSA)
dampratio = 0.0299                             # Damping ratio for spectra
TL1 = 0.05                                   # Lower period limit for matching (s)
TL2 = 6.0                                    # Upper period limit for matching (s)
nit_match = 15
nn = 100                                     # Percentile for RotD (100 = RotD100)
baseline_correct = True
p_order = -1
output_base_name = seed_file_1[:-10]+'_'+target_file[:-4]+'_RotD'+str(nn) # Base name for output files

# --- Load target spectrum and seed record ---

s1, dt, n1, name1 = load_PEERNGA_record(seed_file_1)
s2, _, n2, name2 = load_PEERNGA_record(seed_file_2)

fs = 1 / dt

target_spectrum = np.loadtxt(target_file)
sort_idx = np.argsort(target_spectrum[:, 0])
To = target_spectrum[sort_idx, 0]  # Target spectrum periods
dso = target_spectrum[sort_idx, 1] # Target spectrum PSA
    

# --- Perform Direct RotDnn Spectral Matching ---
# Call the  REQPYrotdnn function
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


# --- Plot Results ---
# Call the plotting function for RotDnn results
fig_hist, fig_spec = plot_rotdnn_results(
    results=results,
    s1_orig=s1, # Pass original unscaled record 1
    s2_orig=s2, # Pass original unscaled record 2
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

# --- Save Matched Records ---

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

