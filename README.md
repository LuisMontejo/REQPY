# Jan 2025 Updates:

-	Fixed minor bugs on the legends of the plots generated
-	Updated numerical integration routine (previous was deprecated by scipy)
-	Optimized response spectra generation routine
-	Automatically saves plots and text files with the generated motions
-	Added optional detrending to the baseline correction routine

# REQPY

A module of python functions to perform spectral matching of 2 horizontal components to a RotDnn target spectrum as described in Montejo (2021). The functions included can also be used to perform single component matching (Montejo and Suarez 2013), perform baseline correction (Suarez and Montejo 2007), and generate single component, rotated and RotDnn spectra.

examples: https://youtu.be/1SFzuP-fjPg
Jan. 2025 updates: https://youtu.be/_5wFK9hNZH4 

cite the paper: Montejo, L. A. (2021). Response spectral matching of horizontal ground motion components to an orientation-independent spectrum (RotDnn). Earthquake Spectra, 37(2), 1127–1144. https://doi.org/10.1177/8755293020970981

cite the code:  [![DOI](https://zenodo.org/badge/287290497.svg)](https://zenodo.org/badge/latestdoi/287290497)

# Other references
Montejo, L. A. (2023). Spectrally matching pulse‐like records to a target RotD100 spectrum. Earthquake Engineering & Structural Dynamics, 52(9), 2796-2811.

Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm for the generation of spectrum-compatible records. International Journal of Advanced Structural Engineering, 5(1), 26.

Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform in the generation and analysis of spectrum-compatible records. Structural Engineering and Mechanics, 27(2), 173-197.

# List of functions included in the module

*REQPYrotdnn: Response spectral matching of horizontal ground motion components to an orientation-independent spectrum (RotDnn)

*REQPY_single: CWT based modification of a single component from a historic record to obtain spectrally equivalent acceleration series 
             
*ResponseSpectrum: decides what approach to use to estimate the response spectrum based on the specified damping value (>=4% frequency domain, <4% piecewise)

*RSPW: Response spectra using a piecewise algorithm

*RSFD: Response spectra (operations performed in the frequency domain)

*ResponseSpectrumTheta: decides what approach to use to estimate the rotated 
response spectra based on damping value (>=4% frequency domain, <4% piecewise)

*RSFDtheta: Rotated response spectra, returns the spectra for each angle 
accommodated in a matrix (operations performed in the frequency domain)

*RSPWtheta: Rotated response spectra, returns the spectra for each angle 
accommodated in a matrix (piecewise approach)

*rotdnn: computes rotated and rotdnn spectra

*basecorr: Performs baseline correction

*baselinecorrect: Performs baseline correction (iteratively calling basecorr)

*cwtzm: Continuous Wavelet Transform using the Suarez-Montejo wavelet via 
convolution in the frequency domain

*zumontw: Generates the Suarez-Montejo Wavelet function

*getdetails: Generates the detail functions from the wavelet coefficients

*CheckPeriodRange: Verifies that the specified matching period range is doable

*load_PEERNGA_record: Load record in .at2 format (PEER NGA Databases)        

