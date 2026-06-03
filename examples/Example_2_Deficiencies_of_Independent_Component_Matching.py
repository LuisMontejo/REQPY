"""
Example 2: Analyzing the Deficiencies of Independent Component Matching
========================================================================

This script illustrates the methodological flaw in the outdated practice of 
matching two horizontal orthogonal components independently to the same 
orientation-independent target spectrum (e.g., RotD100).

Methodological Context:
-----------------------
As introduced in Example 1, modifying components in isolation artificially forces 
each trace to resemble the target envelope. However, when these modified traces are 
recombined and rotated to find the maximum direction response (the actual RotD100), 
the resulting biaxial input heavily overshoots the intended target. 

This script explicitly visualizes this error. It matches two components independently, 
computes the resulting rotated spectra (`PSA180`), and plots the true composite 
`RotD100` alongside its ratio to the target. The output demonstrates why this 
library emphasize direct, simultaneous RotDnn matching.

Workflow:
---------
1. Data Ingestion: Loads two orthogonal seed components and the target spectrum.
2. Independent Matching: Performs standard single-component PSA matching on each 
   trace individually, oblivious to the other.
3. Rotational Analysis: Passes the newly matched traces into the `rotdnn` function 
   to project the biaxial response across all 180 degrees and extract the RotD100 envelope.
4. Visualization: Plots the target, the individually matched components, the full 
   fan of rotated spectra, the final RotD100, and a lower sub-panel detailing the 
   exact over-estimation ratios across the matched period domain.
   
References:
---------
Montejo, L. A., & Suarez, L. E. (2013). "An improved CWT-based algorithm 
for the generation of spectrum-compatible records."
International Journal of Advanced Structural Engineering, 5(1), 26.
https://doi.org/10.1186/2008-6695-5-26

Montejo, L. A. (2021). "Response spectral matching of horizontal ground 
motion components to an orientation-independent spectrum (RotDnn)."
Earthquake Spectra, 37(2), 1127-1144.https://doi.org/10.1177/8755293020970981
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import logging

from reqpy_M import (
    generate_single_component_compatible_record, 
    load_PEERNGA_record, 
    rotdnn
)

plt.close('all')

# ---------------------------------------------------------------------------
# --- Configuration Parameters
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

seed_file_1 = 'RSN175_IMPVALL.H_H-E12140.AT2'             # Seed record comp 1 
seed_file_2 = 'RSN175_IMPVALL.H_H-E12230.AT2'             # Seed record comp 2 
target_file = 'Multi-Period_MCER Spectrum_UPRM_III_C.txt' # Target PSA spectrum 

dampratio = 0.05                             
TL1 = 0.05                                   
TL2 = 6.0                                    
nit_match = 15
nn = 100                                     
output_base_name = f"{seed_file_1[:-10]}_{target_file[:-4]}_RotD{nn}"

# ---------------------------------------------------------------------------
# --- 1. Data Ingestion
# ---------------------------------------------------------------------------
logging.info("Loading seed components and target spectrum...")

s1, dt, n1, name1 = load_PEERNGA_record(seed_file_1)
s2, _, n2, name2 = load_PEERNGA_record(seed_file_2)

n = min(n1, n2)
s1, s2 = s1[:n], s2[:n]
fs = 1 / dt

target_spectrum = np.loadtxt(target_file)
sort_idx = np.argsort(target_spectrum[:, 0])
To = target_spectrum[sort_idx, 0]  
dso = target_spectrum[sort_idx, 1] 
    
# ---------------------------------------------------------------------------
# --- 2. Independent Spectral Matching
# ---------------------------------------------------------------------------
logging.info("Matching Component 1 to target...")
results1 = generate_single_component_compatible_record(
    s=s1, fs=fs, T_PSA=To, targetPSA=dso,
    T1PSA=TL1, T2PSA=TL2, zi=dampratio, nit=nit_match,
    baseline_method='sixth_order')

logging.info("Matching Component 2 to target...")
results2 = generate_single_component_compatible_record(
    s=s2, fs=fs, T_PSA=To, targetPSA=dso,
    T1PSA=TL1, T2PSA=TL2, zi=dampratio, nit=nit_match,
    baseline_method='sixth_order')

sc1 = results1['sc']          
sc2 = results2['sc']          
PSAsc1 = results1['psa_sc']   
PSAsc2 = results2['psa_sc']   
T = results1['periods']       

# ---------------------------------------------------------------------------
# --- 3. Rotational Analysis
# ---------------------------------------------------------------------------
logging.info(f"Computing the resulting RotD{nn} of the combined matched traces...")
PSArotDnn, PSA180 = rotdnn(sc1, sc2, dt, dampratio, T, nn)

# ---------------------------------------------------------------------------
# --- 4. Plotting & Verification
# ---------------------------------------------------------------------------
logging.info("Generating comparison plots...")

fig = plt.figure(figsize=(7.5, 8.5))
gs = gridspec.GridSpec(2, 1, height_ratios=[7, 3], hspace=0.1)
ax1 = fig.add_subplot(gs[0])
ax2 = fig.add_subplot(gs[1], sharex=ax1)

# --- Top Panel: Absolute Spectra ---
ax1.semilogx(T, PSA180.T, lw=1, color='silver', alpha=0.5)
ax1.semilogx(T[0], PSA180[0,0], lw=1, color='silver', alpha=0.5, label='Rotated Spectra')
ax1.semilogx(To, dso, linewidth=2, color='navy', label='Target Spectrum')
ax1.semilogx(To, 1.1 * dso, '--', linewidth=2, color='navy', label='1.1 * Target')
ax1.semilogx(T, PSAsc1, color='cornflowerblue', label='Matched H1')
ax1.semilogx(T, PSAsc2, color='salmon', label='Matched H2')
ax1.semilogx(T, PSArotDnn, color='darkred', lw=1.5, label=f'Resulting RotD{nn}')

ax1.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.12))
ax1.set_xlim(max(T.min(), 0.01), min(T.max(), 20.0)) 
ax1.set_ylim(bottom=0)
ax1.set_ylabel('PSA (g)')
plt.setp(ax1.get_xticklabels(), visible=False)

# --- Bottom Panel: Ratio to Target ---
# Interpolate target spectrum to the exact period vector used for matching
target_interp = np.interp(T, To, dso)
ratio_h1 = PSAsc1 / target_interp
ratio_h2 = PSAsc2 / target_interp
ratio_rotdnn = PSArotDnn / target_interp

ax2.semilogx(T, ratio_h1, color='cornflowerblue', lw=1.0)
ax2.semilogx(T, ratio_h2, color='salmon', lw=1.0)
ax2.semilogx(T, ratio_rotdnn, color='darkred', lw=1.5)

# Target limits
ax2.axhline(1.0, color='navy', lw=2)
ax2.axhline(1.1, color='navy', linestyle='--', lw=2)
ax2.axvspan(TL1, TL2, color='lightgray', alpha=0.3, zorder=0)

ax2.set_xlabel('Period T (s)')
ax2.set_ylabel('Ratio to Target')
ax2.set_ylim(0.8, 1.4)

plt.tight_layout(rect=(0, 0, 1, 0.95))

# Save plot
plot_filename = f"{output_base_name}_Spectra_with_Ratio.png"
plt.savefig(plot_filename, dpi=300)

logging.info(f"Script finished. Note how the resulting RotD{nn} (dark red) "
             f"significantly overshoots the target spectrum (navy) limits within the shaded region.")
plt.show()