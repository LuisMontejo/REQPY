"""
Example 4: Numerical Stability Verification (Self-Matching)
========================================================================

This script serves as a rigorous numerical validation of the Continuous Wavelet 
Transform (CWT) spectral matching algorithm implemented in the `reqpy_plus` module. 

Methodological Context:
-----------------------
We perform a "self-matching" operation:
    
1. We compute the true RotD100 spectrum of a long-duration raw earthquake record.
2. We feed that exact computed spectrum back into the matching algorithm as the 
   target.
3. We run the algorithm for a single iteration with baseline correction completely 
   disabled (`baseline_method='none'`).

If the CWT reconstruction framework is numerically stable, the resulting "matched" 
time history should be nearly identical to the original input. This verifies that 
the wavelet decomposition and inverse reconstruction do not introduce artificial 
distortion, phase shifts, or energy leakage into the signal.

Workflow:
---------
1. Data Ingestion: Loads two orthogonal components of a raw earthquake record from 
   simple text files.
2. Self-Spectrum Calculation: Generates an appropriate period vector based on the 
   record length and calls `rotdnn` to calculate the record's own true RotD100.
3. Self-Matching: Calls `generate_rotdnn_compatible_record` using the calculated 
   self-spectrum as the target, explicitly skipping baseline corrections.
4. Verification: Generates plots using `plot_rotdnn_results`. The Original (Scaled) 
   and Matched lines should perfectly overlap, confirming algorithm stability.
   
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
    rotdnn, 
    plot_rotdnn_results
)

plt.close('all')

# ---------------------------------------------------------------------------
# --- 1. Configuration Parameters
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# File Paths (Assuming standard 1-column txt files for this example)
seed_file_1 = 'KNG007_NS_X.txt'   
seed_file_2 = 'KNG007_EW_Y.txt'   

# Time constraints
dt = 0.02                         
fs = 1.0 / dt                       

# Analysis & Matching constraints
dampratio = 0.05                  
NS = 200                         # Number of periods for spectrum calculation
nits = 1                         # 1 iteration is sufficient for self-match verification

output_base_name = 'Example4_KNG007_SelfMatch'

# ---------------------------------------------------------------------------
# --- 2. Data Ingestion & Period Vector Definition
# ---------------------------------------------------------------------------
logging.info("Loading long-duration seed record components...")

try:
    # Use [:, 1] to extract all rows, but ONLY the second column (index 1)
    # This ignores the Time column entirely and gives us pure 1D acceleration
    s1 = np.loadtxt(seed_file_1)[:, 1]
    s2 = np.loadtxt(seed_file_2)[:, 1]
except OSError:
    logging.error("Could not find the KNG007 text files. Please ensure they are in the working directory.")
    raise

# Ensure equal length arrays
n = min(len(s1), len(s2))
s1, s2 = s1[:n], s2[:n]

# Dynamically calculate appropriate frequency/period boundaries based on record length
max_T_record = n * dt / 4.0
FF1 = max(min(4.0 / (n * dt), 0.1), 1.0 / (max_T_record * 1.5))
FF2 = 1.0 / (2.0 * dt)

# Generate a logarithmically spaced period vector
freqs = np.geomspace(FF2, FF1, NS)
T = np.sort(1.0 / freqs)


# ---------------------------------------------------------------------------
# --- 3. Calculate Target Spectrum (The Self-Spectrum)
# ---------------------------------------------------------------------------
logging.info("Calculating the record's true RotD100 to use as the matching target...")

# Use the standalone rotdnn function to get the actual envelope
PSArot_target, _ = rotdnn(s1, s2, dt, dampratio, T, 100)


# ---------------------------------------------------------------------------
# --- 4. Perform Self-Matching
# ---------------------------------------------------------------------------
logging.info("Performing self-matching (baseline correction disabled)...")

results = generate_rotdnn_compatible_record(
    s1=s1,
    s2=s2,
    fs=fs,
    T_PSA=T,                  # Use self-periods
    targetPSA=PSArot_target,  # Use self-spectrum as target
    nn=100,
    targetPSAlimits=(0.95, 1.05), 
    T1PSA=T[0],               # Match the full valid range
    T2PSA=T[-1],
    zi=dampratio,
    baseline_method='none',   # CRITICAL: Disable to prevent polynomial adjustments
    nit=nits
)

logging.info("Matching complete.")


# ---------------------------------------------------------------------------
# --- 5. Plot Results & Verification
# ---------------------------------------------------------------------------
logging.info("Generating verification plots. Matched records should perfectly overlap the original scaled records.")

fig_spec, fig_hist, max_deltas = plot_rotdnn_results(
    results=results,
    targetPSAlimits=(0.95, 1.05),
    T1PSA=T[0],
    T2PSA=T[-1],
    zi=dampratio,
    plot_directionality=False, # Not needed for stability check
    units='g'
)

# Save the plots
hist_filename = f"{output_base_name}_TimeHistories.png"
spec_filename = f"{output_base_name}_Spectra.png"
fig_hist.savefig(hist_filename, dpi=300)
fig_spec.savefig(spec_filename, dpi=300)

logging.info(f"Saved stability plots to {hist_filename} and {spec_filename}")

# Print the maximum energy deviation (should be computationally zero)
max_ai_error = max(max(max_deltas['Acc1']), max(max_deltas['Acc2']))
logging.info(f"Maximum Deviation in Normalized Arias Intensity: {max_ai_error:.6e}")

plt.show()
