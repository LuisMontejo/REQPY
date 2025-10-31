"""
Example 4: Self-Matching Verification

Tests the numerical stability of the CWT algorithm by "self-matching" a
long-duration record. It calculates the record's own RotD100 spectrum and
then feeds that spectrum back into REQPYrotdnn for one iteration
with baseline correction disabled.

The resulting 'matched' spectrum should be identical to the original.
"""

# Import necessary functions 

from reqpy_M import REQPYrotdnn, rotdnn, plot_rotdnn_results

import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
# --- Configuration ---
seed_file_1 = 'KNG007_NS_X.txt'   # Seed record comp1 [g]
seed_file_2 = 'KNG007_EW_Y.txt'   # Seed record comp2 [g]
dt = 0.02                         # Record time step
fs = 1 / dt                       # Sampling frequency

dampratio = 0.05                  # Damping ratio for spectra
TL1, TL2 = 0, 0                   # Match full range

NS = 200                          # Number of periods for spectrum calculation
nits = 1                          # 1 iteration for self-match
baseline_correct = False          # No baseline correction for self-match
p_order = -1
output_base_name = 'Example4_KNG007_SelfMatch'

# --- Load both components of the seed record ---

gm1 = np.loadtxt(seed_file_1)
s1 = gm1[:, 1]
gm2 = np.loadtxt(seed_file_2)
s2 = gm2[:, 1]

# Ensure equal length
n1 = len(s1); n2 = len(s2); n = min(n1, n2)
s1, s2 = s1[:n], s2[:n]
print(f"Loaded records, Npts={n}, dt={dt:.4f}s")

# --- Define Periods  ---

max_T_record = n * dt / 4.0 # Heuristic max period
FF1 = max(min(4 / (n * dt), 0.1), 1.0 / (max_T_record * 1.5))
FF2 = 1 / (2 * dt)
freqs = np.geomspace(FF2, FF1, NS)
T = 1 / freqs
T = np.sort(T)

# --- 1. Calculate Target Spectrum (self-spectrum) ---
# Use the public 'rotdnn' function
PSArot_target, _ = rotdnn(s1, s2, dt, dampratio, T, 100)

# --- 2. Perform Self-Matching ---
# Use the calculated spectrum (PSArot_target, T) as the target (dso, To)
results = REQPYrotdnn(
    s1=s1,
    s2=s2,
    fs=fs,
    dso=PSArot_target, # Use self-spectrum as target
    To=T,              # Use self-periods as target periods
    nn=100,
    T1=TL1,
    T2=TL2,
    zi=dampratio,
    nit=nits,
    NS=NS,
    baseline=baseline_correct,
    porder=p_order
)
print("Matching complete.")

# --- 3. Plot Results ---
print("\nGenerating comparison plots...")
# Plot the results. 'Original' and 'Matched' lines should be identical.
fig_hist, fig_spec = plot_rotdnn_results(
    results=results,
    s1_orig=s1,
    s2_orig=s2,
    target_spec=(T, PSArot_target), # Plot target spectrum
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

print("\nScript finished. Check plots and time histories, should be very similar.")
