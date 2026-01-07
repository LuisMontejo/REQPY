"""
Example 2: RotDnn from Independently Matched Components (Concise)

Separately matches two components to a target spectrum using REQPY_single,
then calculates the resulting RotD100 spectrum and compares it to the target.
This demonstrates the error introduced by independent matching.

"""

# Import necessary functions

from reqpy_M import REQPY_single, load_PEERNGA_record, rotdnn

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# --- Configuration ---
seed_file_1 = 'RSN175_IMPVALL.H_H-E12140.AT2' # Seed record comp1 [g]
seed_file_2 = 'RSN175_IMPVALL.H_H-E12230.AT2' # Seed record comp2 [g]
target_file = 'ASCE7.txt'                    # Target spectrum (T, PSA)
dampratio = 0.05                             # Damping ratio for spectra
TL1 = 0.05                                   # Lower period limit for matching (s)
TL2 = 6.0                                    # Upper period limit for matching (s)
nit_match = 15
nn = 100                                     # Percentile for RotD (100 = RotD100)
output_base_name = seed_file_1[:-10]+'_'+target_file[:-4]+'_RotD'+str(nn) # Base name for output files

# --- Load target spectrum and seed record ---

s1, dt, n1, name1 = load_PEERNGA_record(seed_file_1)
s2, _, n2, name2 = load_PEERNGA_record(seed_file_2)
# Ensure equal length for processing
n = min(n1, n2)
s1, s2 = s1[:n], s2[:n]
fs = 1 / dt

target_spectrum = np.loadtxt(target_file)
sort_idx = np.argsort(target_spectrum[:, 0])
To = target_spectrum[sort_idx, 0]  # Target spectrum periods
dso = target_spectrum[sort_idx, 1] # Target spectrum PSA
    

# --- Perform Independent Spectral Matching ---

results1 = REQPY_single(
    s=s1, fs=fs, dso=dso, To=To,
    T1=TL1, T2=TL2, zi=dampratio, nit=nit_match,
    baseline=True, porder=-1)

results2 = REQPY_single(
    s=s2, fs=fs, dso=dso, To=To,
    T1=TL1, T2=TL2, zi=dampratio, nit=nit_match,
    baseline=True, porder=-1)

# --- Extract Results ---
ccs1 = results1['ccs']      # Matched acceleration for component 1
ccs2 = results2['ccs']      # Matched acceleration for component 2
PSAccs1 = results1['PSAccs']  # PSA of matched component 1
PSAccs2 = results2['PSAccs']  # PSA of matched component 2
T = results1['T']           # Periods from the calculation (same for both)

# --- Calculate RotD100 of the *matched* records ---

# Use the public 'rotdnn' function from the refactored module
PSArotDnn, PSA180 = rotdnn(ccs1, ccs2, dt, dampratio, T, nn)

# --- Plot Results ---
plt.figure(figsize=(6.5, 6.5))

plt.semilogx(T, PSA180.T, lw=1, color='silver', alpha=0.5)
plt.semilogx(T[0], PSA180[0,0], lw=1, color='silver', alpha=0.5, label='Rotated Spectra')
plt.semilogx(To, dso, linewidth=2, color='navy', label='Target Spectrum')
plt.semilogx(To, 1.1 * dso, '--', linewidth=2, color='navy', label='1.1 * Target')
plt.semilogx(T, PSAccs1, color='cornflowerblue', label='Matched H1')
plt.semilogx(T, PSAccs2, color='salmon', label='Matched H2')
plt.semilogx(T, PSArotDnn, color='darkred', lw=1.5, label=f'Resulting RotD{nn}')

# Configure plot
plt.legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.15))
plt.xlim(max(T.min(), 0.01), min(T.max(), 20.0)) # Set reasonable plot limits
plt.ylim(bottom=0)
plt.xlabel('Period T (s)')
plt.ylabel('PSA (g)')
plt.tight_layout(rect=(0, 0, 1, 0.95))

# Save plot
plot_filename = f"{output_base_name}_Spectra.png"
plt.savefig(plot_filename, dpi=300)

plt.show()

print(f"\nScript finished. Note how the resulting RotD{nn} (dark red) is "
      f"significantly higher than the target spectrum (navy).")
