"""
Example 1: Single-Component Pseudo-Spectral Acceleration (PSA) Matching
========================================================================

This script demonstrates the simplest application of the `reqpy_plus` module: 
adjusting a single horizontal ground motion component to match a target PSA 
response spectrum.

Methodological Context:
-----------------------
While modifying a single component independently is historically common practice, 
it is no longer the recommended approach for defining seismic input, especially 
for 3D non-linear response history analysis. 

If this method is applied sequentially to both orthogonal components to match a 
RotD100 target spectrum, the resulting biaxial input will significantly 
overestimate the target intensity. Furthermore, even if the target is not a RotD100
independent component matching would alter the natural correlation between the 
horizontal traces, artificially altering the original directionality and 
polarization of the seed record. 

This example is provided primarily as a methodological baseline and a historical 
reference to demonstrate the evolution of spectral matching algorithms. For 
modern biaxial matching (RotDnn), refer to the advanced examples.

Workflow:
---------
1. Data Ingestion: Loads an acceleration seed record from an AT2 file and reads 
   the target response spectrum (period vs. spectral acceleration).
2. Spectral Matching: Calls `generate_single_component_compatible_record` to 
   adjust the seed trace using the Continuous Wavelet Transform (CWT) method, 
   employing the default 6th-order polynomial baseline correction.
3. Frequency Domain Verification: Explicitly calculates and plots the Fourier 
   Amplitude Spectra (FAS) and Power Spectral Density (PSD) for the original, 
   scaled, and matched records. This highlights how PSA-only matching affects 
   the underlying frequency composition of the record.
4. Output Generation: Saves verification plots and exports the finalized 
   acceleration time history in multiple standard formats (AT2, single-column, 
   and two-column text).

References:
---------
Montejo, L. A., & Suarez, L. E. (2013). "An improved CWT-based algorithm 
for the generation of spectrum-compatible records."
International Journal of Advanced Structural Engineering, 5(1), 26.
https://doi.org/10.1186/2008-6695-5-26

Suarez, L. E., & Montejo, L. A. (2007). "Applications of the wavelet 
transform in the generation and analysis of spectrum-compatible records."
Structural Engineering and Mechanics, 27(2), 173-197.
https://doi.org/10.12989/sem.2007.27.2.173

Suarez, L. E., & Montejo, L. A. (2005). "Generation of artificial
earthquakes via the wavelet transform." 
Int. Journal of Solids and Structures, 42(21-22), 5905-5919.
https://doi.org/10.1016/j.ijsolstr.2005.03.025

"""

import numpy as np
import matplotlib.pyplot as plt
import logging

from reqp_M import (
    generate_single_component_compatible_record, 
    load_PEERNGA_record, 
    plot_single_component_results,
    save_results_as_at2, 
    save_results_as_2col, 
    save_results_as_1col,
    calculate_earthquake_fas, 
    calculate_earthquake_psd, 
    get_log_freqs,
    plot_fas_psd_comparison
)

plt.close('all')

# ---------------------------------------------------------------------------
# --- Configuration Parameters
# ---------------------------------------------------------------------------

# Set up logging to monitor the iterative matching process
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# File Paths
seed_file = 'RSN175_IMPVALL.H_H-E12140.AT2'                # Seed acceleration record
target_file = 'Multi-Period_MCER Spectrum_UPRM_III_C.txt' # Target PSA spectrum

# Matching Constraints
dampratio = 0.05                 # SDOF damping ratio (5%)
TL1 = 0.05                       # Lower bound of matching period domain (s)
TL2 = 6.0                        # Upper bound of matching period domain (s)
nit_match = 16                   # Number of CWT adjustment iterations
targetPSAlimits = (0.9, 1.1)     # Allowable ratio limits for final match

output_base_name = f"{seed_file[:-4]}_{target_file[:-4]}"


# ---------------------------------------------------------------------------
# --- 1. Data Ingestion
# ---------------------------------------------------------------------------
logging.info("Loading seed record and target spectrum...")

s_orig, dt, npts, eqname = load_PEERNGA_record(seed_file)
fs = 1 / dt
nyquist_freq = fs / 2
   
target_spectrum = np.loadtxt(target_file)
sort_idx = np.argsort(target_spectrum[:, 0])
T_PSA = target_spectrum[sort_idx, 0]  
targetPSA = target_spectrum[sort_idx, 1] 


# ---------------------------------------------------------------------------
# --- 2. Spectral Matching (Single Component)
# ---------------------------------------------------------------------------
logging.info("Initiating single-component PSA matching...")

results = generate_single_component_compatible_record(
    s=s_orig,
    fs=fs,
    T_PSA=T_PSA,
    targetPSA=targetPSA,
    targetPSAlimits=targetPSAlimits,
    T1PSA=TL1,
    T2PSA=TL2,
    zi=dampratio,
    baseline_method='sixth_order', 
    nit=nit_match
)


# ---------------------------------------------------------------------------
# --- 3. Results Extraction & Scaling Calculation
# ---------------------------------------------------------------------------
sc = results['sc']          
sf = results['scale_factor']
s_scaled = s_orig[:len(sc)] * sf  


# ---------------------------------------------------------------------------
# --- 4. Plot Time Histories and PSA Verification
# ---------------------------------------------------------------------------
logging.info("Generating primary verification plots...")

fig_spec, fig_hist, _ = plot_single_component_results(
    results=results,
    targetPSAlimits=targetPSAlimits,
    T1PSA=TL1,
    T2PSA=TL2,
    zi=dampratio,
    units='g'
)

hist_filename = f"{output_base_name}_TimeHistories.png"
spec_filename = f"{output_base_name}_Spectra.png"
fig_hist.savefig(hist_filename, dpi=300)
fig_spec.savefig(spec_filename, dpi=300)


# ---------------------------------------------------------------------------
# --- 5. Frequency Domain Analysis (FAS & PSD)
# ---------------------------------------------------------------------------
logging.info("Calculating FAS and PSD for spectral comparison...")

output_freq_vector = get_log_freqs(fmin=0.1, fmax=nyquist_freq, pts_per_decade=100)

analysis_params = {
    'sample_rate': fs,
    'smoothing_method': 'konno_ohmachi', 
    'smoothing_coeff': 20.0,
    'downsample_freqs': output_freq_vector 
}

# Calculate FAS
freqs_fas_raw, fas_orig_raw, _, fas_orig_smooth = calculate_earthquake_fas(s_orig, **analysis_params)
_, fas_scaled_raw, _, fas_scaled_smooth = calculate_earthquake_fas(s_scaled, **analysis_params)
_, fas_matched_raw, _, fas_matched_smooth = calculate_earthquake_fas(sc, **analysis_params)

# Calculate PSD
freqs_psd_raw, psd_orig_raw, _, _, _, psd_orig_smooth = calculate_earthquake_psd(s_orig, **analysis_params)
_, psd_scaled_raw, _, _, _, psd_scaled_smooth = calculate_earthquake_psd(s_scaled, **analysis_params)
_, psd_matched_raw, _, _, _, psd_matched_smooth = calculate_earthquake_psd(sc, **analysis_params)


# ---------------------------------------------------------------------------
# --- 6. Plot FAS and PSD Comparisons
# ---------------------------------------------------------------------------
fig_fas, fig_psd = plot_fas_psd_comparison(
    freqs_fas_raw=freqs_fas_raw, fas_orig_raw=fas_orig_raw, 
    fas_scaled_raw=fas_scaled_raw, fas_matched_raw=fas_matched_raw,
    freqs_psd_raw=freqs_psd_raw, psd_orig_raw=psd_orig_raw, 
    psd_scaled_raw=psd_scaled_raw, psd_matched_raw=psd_matched_raw,
    output_freq_vector=output_freq_vector,
    fas_orig_smooth=fas_orig_smooth, fas_scaled_smooth=fas_scaled_smooth, fas_matched_smooth=fas_matched_smooth,
    psd_orig_smooth=psd_orig_smooth, psd_scaled_smooth=psd_scaled_smooth, psd_matched_smooth=psd_matched_smooth
)

fas_filename = f"{output_base_name}_FAS_Comparison.png"
psd_filename = f"{output_base_name}_PSD_Comparison.png"
fig_fas.savefig(fas_filename, dpi=300)
fig_psd.savefig(psd_filename, dpi=300)


# ---------------------------------------------------------------------------
# --- 7. Data Export
# ---------------------------------------------------------------------------
logging.info("Saving matched acceleration time history to disk...")

# Option 1: AT2 Format
at2_filepath = f"{output_base_name}_Matched.AT2"
at2_header_details = {
    'title': f'Matched record from {seed_file} (Target: {target_file})',
    'date': '01/01/2026', 
    'station': eqname.split('_comp_')[0] if '_comp_' in eqname else eqname,
    'component': f"{eqname.split('_comp_')[-1]}-Matched"
}
save_results_as_at2(results, at2_filepath, comp_key='sc', header_details=at2_header_details)

# Option 2: 2-Column TXT (Time, Accel)
txt_2col_filepath = f"{output_base_name}_Matched_2col.txt"
header_2col = (f"Matched acceleration (g) vs. Time (s)\n"
               f"Original Seed: {eqname}\n"
               f"Target Spectrum: {target_file}\n"
               f"Time (s), Acceleration (g)")
save_results_as_2col(results, txt_2col_filepath, comp_key='sc', header_str=header_2col)

# Option 3: 1-Column TXT (Accel)
txt_1col_filepath = f"{output_base_name}_Matched_1col.txt"
header_1col = (f"Matched acceleration (g), dt={results.get('dt', 0.0):.8f}s\n"
               f"Original Seed: {eqname}\n"
               f"Target Spectrum: {target_file}\n"
               f"Data points follow:")
save_results_as_1col(results, txt_1col_filepath, comp_key='sc', header_str=header_1col)
    
logging.info(f"Data successfully exported. Script finished.")
plt.show()