REQPY: Spectral Matching of Earthquake Records

A Python module for spectral matching of earthquake records using the Continuous Wavelet Transform (CWT) based methodologies described in the referenced papers.

Its primary capabilities include:

Matching a single ground motion component to a target spectrum.

Matching a pair of horizontal components to an orientation-independent target spectrum RotDnn (e.g., RotD100).

Analysis functions for generating standard and rotated (RotDnn) spectra.

Baseline correction routines for processed time histories.

Installation

You can install REQPY using pip:

pip install reqpy_M

Dependencies

REQPY requires the following Python packages:

NumPy

SciPy

Matplotlib

Numba

Quick Start

Example 1: Single Component Matching

from reqpy_M import REQPY_single, load_PEERNGA_record, plot_single_results, save_results_as_1col
import numpy as np
import matplotlib.pyplot as plt

# Load seed record and target spectrum
s_orig, dt, _, _ = load_PEERNGA_record('RSN175_IMPVALL.H_H-E12140.AT2')
target_spec = np.loadtxt('ASCE7.txt')
To, dso = target_spec[:, 0], target_spec[:, 1]
fs = 1/dt

# Perform matching
results = REQPY_single(
    s=s_orig, fs=fs, dso=dso, To=To,
    T1=0.05, T2=6.0, zi=0.05, nit=15
)

# Plot results
fig_hist, fig_spec = plot_single_results(
    results, s_orig=s_orig, target_spec=(To, dso), T1=0.05, T2=6.0
)
# fig_hist.savefig('single_time_history.png')
# fig_spec.savefig('single_spectrum.png')
plt.show()

# Save matched record
save_results_as_1col(results, 'matched_single_1col.txt', comp_key='ccs', header_str=f'Matched accel [g], dt={dt}')


Example 2: Two-Component RotDnn Matching (e.g., RotD100)

from reqpy_M import REQPYrotdnn, load_PEERNGA_record, plot_rotdnn_results, save_results_as_1col
import numpy as np
import matplotlib.pyplot as plt

# Load seed record components
s1_orig, dt, _, _ = load_PEERNGA_record('RSN175_IMPVALL.H_H-E12140.AT2')
s2_orig, _, _, _ = load_PEERNGA_record('RSN175_IMPVALL.H_H-E12230.AT2')
fs = 1/dt

# Load target spectrum
target_spec = np.loadtxt('ASCE7.txt')
To, dso = target_spec[:, 0], target_spec[:, 1]

# Perform RotD100 matching
results = REQPYrotdnn(
    s1=s1_orig, s2=s2_orig, fs=fs, dso=dso, To=To, nn=100,
    T1=0.05, T2=6.0, zi=0.05, nit=15
)

# Plot results
fig_hist, fig_spec = plot_rotdnn_results(
    results, s1_orig=s1_orig, s2_orig=s2_orig, target_spec=(To, dso), T1=0.05, T2=6.0
)
# fig_hist.savefig('rotdnn_time_history.png')
# fig_spec.savefig('rotdnn_spectrum.png')
plt.show()

# Save matched records
header = f'Matched accel [g], dt={dt}'
save_results_as_1col(results, 'matched_rotdnn_comp1_1col.txt', comp_key='scc1', header_str=header)
save_results_as_1col(results, 'matched_rotdnn_comp2_1col.txt', comp_key='scc2', header_str=header)


References

[1] Montejo, L. A. (2021). Response spectral matching of horizontal ground motion components to an orientation-independent spectrum (RotDnn). Earthquake Spectra, 37(2), 1127-1144.

[2] Montejo, L. A. (2023). Spectrally matching pulse‐like records to a target RotD100 spectrum. Earthquake Engineering & Structural Dynamics, 52(9), 2796-2811.

[3] Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm for the generation of spectrum-compatible records. International Journal of Advanced Structural Engineering, 5(1), 26.

[4] Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform in the generation and analysis of spectrum-compatible records. Structural Engineering and Mechanics, 27(2), 173-197.

[5] Suarez, L. E., & Montejo, L. A. (2005). Generation of artificial earthquakes via the wavelet transform. Int. Journal of Solids and Structures, 42(21-22), 5905-5919.

Author

Luis A. Montejo (luis.montejo@upr.edu)

License

This project is licensed under the MIT License - see the LICENSE file for details.