
# REQPY: Spectral Matching & Signal Processing Library

**A comprehensive Python module for spectral matching of earthquake records, supporting single-component, RotDnn, and PSD/FAS-compatible generation.**

[![PyPI version](https://badge.fury.io/py/reqpy-M.svg)](https://badge.fury.io/py/reqpy-M)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4007728.svg)](https://doi.org/10.5281/zenodo.4007728)

## Overview

**REQPY** implements Continuous Wavelet Transform (CWT) based methodologies to modify earthquake acceleration time histories. It allows users to match target response spectra (PSA) while optionally satisfying Fourier Amplitude Spectrum (FAS) and Power Spectral Density (PSD) requirements.

This package consolidates the functionality of the previous `REQPY` and `ReqPyPSD` modules into a single, unified library.

### Key Capabilities
1.  **Single Component Matching:** Match a seed record to a target response spectrum (PSA).
2.  **RotDnn Matching:** Match a pair of horizontal components to an orientation-independent target spectrum (e.g., RotD100).
3.  **Advanced Matching (New in v0.3.0):** Generate records compatible with PSA, minimum PSD, and/or minimum FAS requirements.
4.  **Signal Analysis (New in v0.3.0):** Compute FAS, PSD, RotDnn spectra, Effective Amplitude Spectra (EAS), and Effective Power Spectra (EPSD) with various smoothing options (including Konno-Ohmachi).
5.  **Correction Routines:** Baseline correction and localized time-domain PGA correction.

---

## Installation

Install the package via pip:

```bash
pip install reqpy-M
```

Recommended: To enable faster Konno-Ohmachi smoothing (recommended for large datasets), install with the smoothing extra:

```bash
pip install "reqpy-M[smoothing]"
```

---

## Dependencies

REQPY requires the following Python packages:

* **NumPy**
* **SciPy** (>= 1.6.0)
* **Matplotlib**
* **Numba** (Required for optimized performance)
* **pykooh** (Optional, recommended for faster Konno-Ohmachi smoothing)

---

## Provided usage examples (script files):

**Example 1 - SingleComponentMatchingExample.py**
Matches a single component to a target spectrum.
This updated version also computes and plots the FAS and PSD of the
original, scaled, and matched records for comparison, demonstrating the
new analysis functions.

**Example 2 - GeneratingRotatedAndRotDnnSpectraExample.py**
Separately matches two components to a target spectrum using REQPY_single,
then calculates the resulting RotD100 spectrum and compares it to the target.
This demonstrates the error introduced by independent matching.

**Example 3 - TwoComponentRotDnnMatchingExample.py**
Modifies two horizontal components from a historic record simultaneously so that
the resulting RotD100 response spectrum (computed from the pair)
matches the specified RotD100 design/target spectrum.
This updated version also computes and plots the RotDnn FAS, RotDnn PSD,
Effective FAS, and Effective PSD for the original, scaled, and matched pairs
using the recommended "smooth last" workflow.

**Example 4 - Self matching long duration record verification.py**
Tests the numerical stability of the CWT algorithm by "self-matching" a
long-duration record. It calculates the record's own RotD100 spectrum and
then feeds that spectrum back into REQPYrotdnn for one iteration
with baseline correction disabled.
The resulting 'matched' spectrum should be identical to the original.

**Example 5 - PSA_PSD_Matching.py**
This script demonstrates the complete workflow:
1.  Loading a seed record and target spectra (PSA and PSD).
2.  Generating a compatible record using 'generate_psa_psd_compatible_record'.
3.  Automatically generating verification plots using 'plot_psa_psd_results'.

**Example 6 - PSA_FAS_PSD_Matching.py**
Generating Records Compatible with PSA, Minimum FAS, and Minimum PSD.
This script demonstrates the new functionality for FAS compliance.


---
## References

[1] Suarez, L. A., & Montejo, L. A. (2005). Generation of artificial earthquakes via the wavelet transform. Int. Journal of Solids and Structures, 42(21-22), 5905-5919.

[2] Montejo, L. A. (2025). "Generation of Response Spectrum Compatible Records Satisfying a Minimum Power Spectral Density Function." Earthquake Engineering and Resilience. DOI: 10.1002/eer2.70008

[3]  Montejo, L. A. (2024). "Strong-Motion-Duration-Dependent Power Spectral Density Functions Compatible with Design Response Spectra." Geotechnics 4(4), 1048-1064. DOI: 10.3390/geotechnics4040053

[4] Montejo, L. A. (2021). "Response spectral matching of horizontal ground motion components to an orientation-independent spectrum (RotDnn)." Earthquake Spectra, 37(2), 1127-1144.

[5] Montejo, L. A., & Suarez, L. E. (2013). "An improved CWT-based algorithm for the generation of spectrum-compatible records." International Journal of Advanced Structural Engineering, 5(1), 26.

[6] Suarez, L. E., & Montejo, L. A. (2007). "Applications of the wavelet transform in the generation and analysis of spectrum-compatible records." Structural Engineering and Mechanics, 27(2), 173-197.

[7] Suarez, L. E., & Montejo, L. A. (2005). "Generation of artificial earthquakes via the wavelet transform." Int. Journal of Solids and Structures, 42(21-22), 5905-5919.



## Changelog

### v0.3.0 (Jan 2026)

**Consolidation**: Merged functionality from ReqPyPSD into REQPY.

**New Features**: Added generate_psa_psd_compatible_record and generate_psa_psd_fas_compatible_record for advanced matching.

**Analysis**: Added comprehensive FAS/PSD calculation functions (calculate_earthquake_psd, calculate_fas_rotDnn, etc.) with Konno-Ohmachi smoothing.

**Utilities**: Added pga_correction for localized time-domain scaling.

**Dependencies**: Added optional support for pykooh for faster smoothing.

### v0.2.0 (Oct 2025)

Refactored core functions to return dictionaries.

Applied NumPy docstring standards and type hinting.

Added public plotting functions.

### v0.1.0 (Jan 2025)

Initial PyPI release.

---
## License

This project is licensed under the MIT License - see the LICENSE file for details.

**Author**: Luis A. Montejo (luis.montejo@upr.edu)

**Copyright**: 2021-2026

