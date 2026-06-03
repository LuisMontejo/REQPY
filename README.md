
# REQPY: Spectral Matching & Signal Processing Library

**A comprehensive Python module for spectral matching of earthquake records, supporting single-component, RotDnn, and PSD/FAS-compatible generation.**

[![PyPI version](https://badge.fury.io/py/reqpy-M.svg)](https://badge.fury.io/py/reqpy-M)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4007728.svg)](https://doi.org/10.5281/zenodo.4007728)

## Overview

**REQPY** implements Continuous Wavelet Transform (CWT) based methodologies to modify earthquake acceleration time histories. It allows users to match target 
response spectra (PSA) while optionally satisfying Fourier Amplitude Spectrum (FAS) and Power Spectral Density (PSD) requirements.

This package consolidates the functionality of the previous `REQPY` and `ReqPyPSD` modules into a single, unified library.

### Key Capabilities
1.  **Single Component Matching:** Match a seed record to a target response spectrum (PSA).
2.  **RotDnn Matching:** Match a pair of horizontal components to an orientation-independent target spectrum (e.g., RotD100).
3.  **Advanced Matching (New in v0.4.0):** Generate single-component and biaxial (RotDnn) records compatible with PSA, minimum PSD, and/or minimum FAS requirements.
4.  **Signal Analysis:** Compute FAS, PSD, RotDnn spectra, Effective Amplitude Spectra (EAS), and Effective Power Spectra (EPSD) with various smoothing options (including Konno-Ohmachi).
5.  **Baseline Correction Routines (New in v0.4.0):** Versatile baseline correction methods explicitly selectable during generation 
	(`sixth_order`, `piecewise`, `classic`, or `none`).

---

## Installation

Install the package via pip:

```bash
pip install reqpy-M
```

Recommended: To enable faster Konno-Ohmachi smoothing (recommended for large datasets), install with the smoothing extra:
Albert Kottke. (2025). arkottke/pykooh (v0.5.0). Zenodo. https://doi.org/10.5281/zenodo.15499453

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

**Example 1 - Single_Component_Matching_PSA.py**
Matches a single component to a target spectrum, demonstrating the simplest 
application of the reqpy_M module as a methodological baseline.
Also computes and plots the FAS and PSD of the original, scaled, and matched 
records for comparison, demonstrating the new analysis functions.

**Example 2 - Deficiencies_of_Independent_Component_Matching.py**
Separately matches two components to a target spectrum using REQPY_single,
then calculates the resulting RotD100 spectrum and compares it to the target.
This demonstrates the error introduced by independent matching.

**Example 3 - Two_Component_RotDnn_Matching.py**
Modifies two horizontal components from a historic record simultaneously so that
the resulting RotD100 response spectrum (computed from the pair)
matches the specified RotD100 design/target spectrum.
Also computes and plots the RotDnn FAS, RotDnn PSD, Effective FAS, and 
Effective PSD for the original, scaled, and matched pairs using the recommended
"smooth last" workflow.

**Example 4 - Self_Matching_Long_Duration_Record_Verification.py**
Tests the numerical stability of the CWT algorithm by "self-matching" a
long-duration record. It calculates the record's own RotD100 spectrum and
then feeds that spectrum back into REQPYrotdnn for one iteration
with baseline correction disabled to ensure no artificial distortion or 
energy leakage occurs.

**Example 5 - Single_Component_Matching_PSA_Plus_Minimum_PSD_Compliance.py**
Demonstrates advanced single-component spectral matching, ensuring the resulting 
acceleration history complies with both a Target Response Spectrum (PSA) and a 
minimum Target Power Spectral Density (PSD) function to prevent energy depletion.

**Example 6 - Single_Component_Matching_PSA_Plus_Minimum_FAS_and_PSD_Compliance.py**
Extends Example 5 by introducing an intermediate minimum Fourier Amplitude Spectrum (FAS) 
adjustment stage, using a three-stage approach (PSA -> FAS -> PSD) to preserve the 
temporal phase envelope.

**Example 7 - RotDnn_Matching_PSA_Plus_Minimum_PSD.py**
Integrates simultaneous biaxial matching (RotDnn) with targeted time-frequency-domain energy 
injection, ensuring the resulting acceleration histories comply with an 
orientation-independent Target Response Spectrum (RotD100) and a minimum PSD function.

**Example 8 - RotDnn_Matching_PSA_Plus_Minimum_FAS_and_PSD.py**
Demonstrates the complete, state-of-the-art methodology for generating orientation-independent 
matched records. Incorporates the full three-stage correction process 
(Biaxial PSA -> Biaxial FAS -> Biaxial PSD) to smoothly satisfy all regulatory requirements 
while preserving natural directionality.

---
## References

[1] Montejo, L.A. (2026). "Generation of Orientation-Independent Response Spectrum 
    Matched Records Satisfying Minimum Fourier Amplitude and Power Spectral Density 
    Requirements." https://doi.org/10.31223/X5Z49W
    
[2] Montejo, L. A. (2026). Generation of Fourier Amplitude Spectra and Power Spectral 
    Density Functions Compatible with Orientation-Independent Design Spectra for 
    Bidirectional Seismic Analyses of Nuclear Facilities. Nuclear Engineering and 
    Technology, 104136. https://doi.org/10.1016/j.net.2026.104136

[3] Montejo, L. A. (2025). "Generation of Response Spectrum Compatible Records 
    Satisfying a Minimum Power Spectral Density Function." 
    Earthquake Engineering and Resilience. https://doi.org/10.1002/eer2.70008

[4] Montejo, L. A. (2024). "Strong-Motion-Duration-Dependent Power Spectral 
    Density Functions Compatible with Design Response Spectra." 
    Geotechnics 4(4), 1048-1064. https://doi.org/10.3390/geotechnics4040053

[5] Montejo, L. A. (2021). "Response spectral matching of horizontal ground 
    motion components to an orientation-independent spectrum (RotDnn)."
    Earthquake Spectra, 37(2), 1127-1144.https://doi.org/10.1177/8755293020970981 

[6] Montejo, L. A., & Suarez, L. E. (2013). "An improved CWT-based algorithm 
    for the generation of spectrum-compatible records."
    International Journal of Advanced Structural Engineering, 5(1), 26.
    https://doi.org/10.1186/2008-6695-5-26

[7] Suarez, L. E., & Montejo, L. A. (2007). "Applications of the wavelet 
    transform in the generation and analysis of spectrum-compatible records."
    Structural Engineering and Mechanics, 27(2), 173-197.
    https://doi.org/10.12989/sem.2007.27.2.173

[8] Suarez, L. E., & Montejo, L. A. (2005). "Generation of artificial
    earthquakes via the wavelet transform." 
    Int. Journal of Solids and Structures, 42(21-22), 5905-5919.
    https://doi.org/10.1016/j.ijsolstr.2005.03.025



## Changelog

### v0.4.0 (Jun 2026)

- **Biaxial Advanced Matching:** Introduced `generate_rotdnn_psa_fas_psd_compatible_record` 
  to simultaneously match RotDnn PSA while satisfying minimum FAS and PSD requirements.
  
- **Three-Stage Adjustment:** Implemented a unified `adjustment_mode` ('both', 'fas', 'psd') 
  for sequential compliance correction, minimizing temporal phase disruption.
  
- **Versatile Baseline Corrections:** Added explicitly selectable baseline correction methods 
  (sixth_order [default], piecewise, classic, none) within generation functions to strictly 
  manage velocity and displacement drifts.

- **Expanded Examples Suite:** Updated repository to include 8 fully documented scripts ranging 
  from basic matching to advanced multi-stage RotDnn compliance.

- **Enhanced Verification Plots:** Added comprehensive plotting suites for orientation-independent 
  matching, including polar directionality visualizations.

### v0.3.0 (Jan 2026)

Consolidation: Merged functionality from ReqPyPSD into REQPY.

New Features: Added generate_psa_psd_compatible_record and generate_psa_psd_fas_compatible_record 
for advanced matching.

Analysis: Added comprehensive FAS/PSD calculation functions (calculate_earthquake_psd, calculate_fas_rotDnn, etc.) 
with Konno-Ohmachi smoothing.

Utilities: Added pga_correction for localized time-domain scaling.

Dependencies: Added optional support for pykooh for faster smoothing.

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


