"""
REQPY: Spectral Matching & Signal Processing Library
luis.montejo@upr.edu

This module implements Continuous Wavelet Transform (CWT) based methodologies
to modify earthquake acceleration time histories to match target response 
spectra, while optionally satisfying Fourier Amplitude Spectrum (FAS) and 
Power Spectral Density (PSD) requirements.

Primary Capabilities:
1.  **Single Component Matching:** Match a seed record to a target response 
    spectrum (PSA).
2.  **RotDnn Matching:** Match a pair of horizontal components to an 
    orientation-independent target spectrum (e.g., RotD100).
3.  **Advanced Matching (New in v0.4.0):** Generate single-component and 
    biaxial (RotDnn) records compatible with PSA, minimum PSD, and/or 
    minimum FAS requirements.
4.  **Signal Analysis:** Compute FAS, PSD, RotDnn spectra, Effective 
    Amplitude Spectra (EAS), and Effective Power Spectra (EPSD) with 
    various smoothing options (including Konno-Ohmachi).
5.  **Correction Routines:** Baseline correction and localized time-domain 
    PGA correction.

===============================================================================
REFERENCES
===============================================================================

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

===============================================================================
CHANGELOG
===============================================================================

v0.4.0 (Jun 2026):
- **Biaxial Advanced Matching:** Introduced `generate_rotdnn_psa_fas_psd_compatible_record` 
  to simultaneously match RotDnn PSA while satisfying minimum FAS and PSD requirements.
- **Three-Stage Adjustment:** Implemented a unified `adjustment_mode` ('both', 'fas', 'psd') 
  for sequential compliance correction, minimizing temporal phase disruption.
- **Enhanced Verification Plots:** Added comprehensive plotting suites for 
  orientation-independent matching (`plot_rotdnn_psa_psd_fas_results`).
  
v0.3.0 (Jan 2026):
- **Consolidation:** Merged functionality from `ReqPyPSD` into `REQPY`.
- **New Features:** Added `generate_psa_psd_compatible_record` and 
  `generate_psa_psd_fas_compatible_record` for advanced matching.
- **Analysis:** Added comprehensive FAS/PSD calculation functions (`calculate_earthquake_psd`, 
  `calculate_fas_rotDnn`, etc.) with Konno-Ohmachi smoothing.
- **Utilities:** Added `pga_correction` for localized time-domain scaling.
- **Dependencies:** Added optional support for `pykooh` for faster smoothing.

v0.2.0 (Oct 2025):
- Refactored core functions to return dictionaries instead of tuples.
- Applied NumPy docstring standards and type hinting.
- Added public plotting functions (`plot_single_results`, `plot_rotdnn_results`).
- Improved error handling and input validation.

v0.1.0 (Jan 2025):
- Initial PyPI release.
- Optimized response spectra generation routines.
- Added optional detrending to the baseline correction routine.
"""

__author__ = "Luis A. Montejo"
__copyright__ = "Copyright 2021-2026, Luis A. Montejo"
__license__ = "MIT"
__version__ = "0.4.1"
__email__ = "luis.montejo@upr.edu"

# =============================================================================
# IMPORTS
# =============================================================================
import logging
import numpy as np
from numba import jit
import scipy
from scipy import integrate, signal
from scipy.interpolate import LSQUnivariateSpline
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List, Optional, Union, Literal, Dict, Any
import warnings
from matplotlib.gridspec import GridSpec
from scipy.signal import find_peaks
from scipy.spatial import ConvexHull

from numpy.fft import fft, ifft
    
from matplotlib.lines import Line2D

try:
    import pykooh
    PYKOOH_AVAILABLE = True
except ImportError:
    PYKOOH_AVAILABLE = False

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from scipy.sparse import csr_matrix
    
    
# Set up a logger for the module
log = logging.getLogger(__name__)
# Example basic configuration (user can configure this externally)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# =============================================================================
# PUBLIC API 
# =============================================================================

def generate_rotdnn_psa_fas_psd_compatible_record(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    f_PSA: np.ndarray,
    targetPSA: np.ndarray,
    f_PSD: np.ndarray = None,
    targetPSD: np.ndarray = None,
    f_FAS: np.ndarray = None,
    targetFAS: np.ndarray = None,
    nn: int = 50,
    adjustment_mode: str = 'both',
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    PSDreduction: float = 1.0,
    FASreduction: float = 1.0,
    F1PSA: float = 0.2,
    F2PSA: float = 50.0,
    F1Check: float = 0.3,
    F2Check: float = 30.0,
    zi: float = 0.05,
    baseline_method: str = 'sixth_order',
    PSA_poly_order: int = 4,
    PSAPSD_poly_order: int = 4,
    BL_target_disp: float = 0.0,
    knlocs_ai_pct: List[float] = [1.0, 5.0, 40.0, 75.0, 95.0, 99.0],
    NS: int = 300,
    nit: int = 30,
    maxit: int = 1000,
    localized: bool = True,
    smoothing_method: str = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    prefer_pykooh: bool = True) -> Dict[str, Any]:
    
    """
    Generates a pair of horizontal ground motion components compatible with 
    orientation-independent target spectra (RotDnn), while optionally satisfying 
    minimum Fourier Amplitude Spectrum (FAS) and Power Spectral Density (PSD) requirements.

    Parameters
    ----------
    s1, s2 : numpy.ndarray
        Horizontal seed acceleration time-series pairs (must be same length).
    fs : float
        Sampling frequency of the seed records (Hz).
    f_PSA : numpy.ndarray
        Array of frequencies (Hz) at which the target PSA is defined.
    targetPSA : numpy.ndarray
        Target pseudo-acceleration response spectrum (RotDnn).
    f_PSD : numpy.ndarray, optional
        Array of frequencies (Hz) at which the target PSD is defined.
    targetPSD : numpy.ndarray, optional
        Target power spectral density function (RotDnn).
    f_FAS : numpy.ndarray, optional
        Array of frequencies (Hz) at which the target FAS is defined.
    targetFAS : numpy.ndarray, optional
        Target Fourier amplitude spectrum (RotDnn).
    nn : int, optional
        Percentile for orientation-independent metrics (e.g., 50 for RotD50, 
        100 for RotD100). Default is 50.
    adjustment_mode : str, optional
        Selects the matching stages to run. Options: 'psd', 'fas', or 'both'. 
        Default is 'both'.
    targetPSAlimits : Tuple[float, float], optional
        Tolerance limits (min, max) for PSA matching. Default is (0.9, 1.3).
    PSDreduction : float, optional
        Target ratio for minimum PSD matching. Default is 1.0.
    FASreduction : float, optional
        Target ratio for minimum FAS matching. Default is 1.0.
    F1PSA, F2PSA : float, optional
        Frequency range (Hz) for PSA matching. Default is 0.2 to 50.0.
    F1Check, F2Check : float, optional
        Frequency range (Hz) for FAS/PSD matching. Default is 0.3 to 30.0.
    zi : float, optional
        Damping ratio for the response spectra. Default is 0.05.
    baseline_method : str, optional
        Method for baseline correction: 'none', 'classic', 'piecewise', 'sixth_order'. 
        Default is 'sixth_order'.
    PSA_poly_order : int, optional
        Polynomial order for 'classic' baseline correction (PSA match stage). Default is 4.
    PSAPSD_poly_order : int, optional
        Polynomial order for 'classic' baseline correction (FAS/PSD match stage). Default is 4.
    BL_target_disp : float, optional
        Target final displacement for 'piecewise' baseline correction. Default is 0.0.
    knlocs_ai_pct : List[float], optional
        Knot locations (in Arias Intensity percentiles) for 'piecewise' correction.
    NS : int, optional
        Number of scales for the CWT. Default is 300.
    nit : int, optional
        Number of iterations for the initial PSA matching. Default is 30.
    maxit : int, optional
        Maximum number of iterations for the FAS/PSD adjustment loop. Default is 1000.
    localized : bool, optional
        If True, applies a localized window during PSD modification. Default is True.
    smoothing_method : str, optional
        Smoothing method for spectra computation. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the smoothing method. Default is 20.0.
    prefer_pykooh : bool, optional
        If True, uses pykooh for optimized Konno-Ohmachi smoothing. Default is True.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing matched time histories, uncorrected histories, 
        and spectral metrics. Skipped stages return None for arrays and NaN for metrics.
        
        General Keys:
        - 'mode' (str): The adjustment mode executed ('psd', 'fas', or 'both').
        - 't', 'dt' (np.ndarray, float): Time vector (s) and time step.
        - 'freqs', 'periods' (np.ndarray): Evaluation frequency and period vectors.
        - 'nn', 'scale_factor' (int, float): RotDnn percentile and rigorous initial scale factor.
        - 'target_psa', 'target_psd', 'target_fas': Interpolated target arrays.
        
        For Both Components (i = 1 and 2):
        - 's{i}_scaled': Scaled seed record.
        - 'sc{i}_unc', 'sc{i}_fas_unc', 'sca{i}_unc': Uncorrected matched records for PSA, FAS, and PSD stages.
        - 'sc{i}', 'sc{i}_fas', 'sca{i}': Baseline-corrected records for PSA, FAS, and PSD stages.
        - 'vel{i}_s', 'vel{i}_sc', 'vel{i}_sc_fas', 'vel{i}_sca': Velocity time histories for all stages.
        - 'disp{i}_s', 'disp{i}_sc', 'disp{i}_sc_fas', 'disp{i}_sca': Displacement time histories for all stages.
        - 'ai{i}_s', 'ai{i}_sc', 'ai{i}_sc_fas', 'ai{i}_sca': Normalized Arias Intensity histories.
        - 'csv{i}_s', 'csv{i}_sc', 'csv{i}_sc_fas', 'csv{i}_sca': Normalized Cumulative Squared Velocity histories.
        - 'csd{i}_s', 'csd{i}_sc', 'csd{i}_sc_fas', 'csd{i}_sca': Normalized Cumulative Squared Displacement histories.
        
        Combined Spectra (RotDnn for PSA, FAS, and PSD):
        - 'psa_s', 'psa_sc', 'psa_sc_fas', 'psa_sca': Computed RotDnn PSA.
        - 'psd_s', 'psd_sc', 'psd_sc_fas', 'psd_sca': Computed RotDnn PSD.
        - 'fas_s', 'fas_sc', 'fas_sc_fas', 'fas_sca': Computed RotDnn FAS.
        - 'psa_180_seed', 'psa_180_sc', 'psa_180_fas', 'psa_180_final': 180-degree PSA matrices.
    """
    
    # --- Input Validation & Routing ---
    mode = adjustment_mode.lower().strip()
    if mode not in ['psd', 'fas', 'both']:
        raise ValueError("adjustment_mode must be 'psd', 'fas', or 'both'")
        
    do_fas = mode in ['fas', 'both']
    do_psd = mode in ['psd', 'both']
    
    if do_psd and (f_PSD is None or targetPSD is None):
        raise ValueError("targetPSD and f_PSD must be provided when adjustment_mode includes 'psd'")
    if do_fas and (f_FAS is None or targetFAS is None):
        raise ValueError("targetFAS and f_FAS must be provided when adjustment_mode includes 'fas'")

    pi = np.pi; dt = 1/fs; n1 = len(s1); n2 = len(s2); nt = min(n1, n2)
    s1 = s1[:nt]; s2 = s2[:nt]
    if nt % 2 != 0: 
        s1 = np.append(s1, 0); s2 = np.append(s2, 0); nt += 1
    t = np.linspace(0, (nt-1)*dt, nt)
    
    # --- 1. Target Interpolation Prep ---
    idx_f = np.argsort(f_PSA)
    f_PSA = f_PSA[idx_f]; targetPSA = targetPSA[idx_f]
    To_target = 1.0 / f_PSA; idx_T = np.argsort(To_target)
    To_target = To_target[idx_T]; targetPSA_T = targetPSA[idx_T]
    
    if do_psd:
        idx_psd = np.argsort(f_PSD); f_PSD = f_PSD[idx_psd]; targetPSD = targetPSD[idx_psd]
    if do_fas:
        idx_fas = np.argsort(f_FAS); f_FAS = f_FAS[idx_fas]; targetFAS = targetFAS[idx_fas]
    
    T1PSA, T2PSA = 1.0/F2PSA, 1.0/F1PSA
    T1Check, T2Check = 1.0/F2Check, 1.0/F1Check
    FF1 = min(4/(nt*dt), 0.1); FF2 = 1/(2*dt)
    T1PSA, T2PSA, FF1 = _CheckPeriodRange(T1PSA, T2PSA, To_target, FF1, FF2)

    omega = pi; zeta = 0.05
    freqs = np.geomspace(FF2, FF1, NS); T = 1/freqs; scales = omega / (2*pi*freqs)
    
    ds = log_interp(T, To_target, targetPSA_T)
    TlocsPSA = np.where((T >= T1PSA) & (T <= T2PSA))[0]
    TlocsCheck = np.where((T >= T1Check) & (T <= T2Check))[0] 
    
    TargetPSD_int = log_interp(freqs, f_PSD, targetPSD) if do_psd else None
    TargetFAS_int = log_interp(freqs, f_FAS, targetFAS) if do_fas else None

    # --- 2. Wavelet Decomposition ---
    log.info(f"Performing CWT decomposition (RotD{nn}, Mode: {mode.upper()})...")
    C1 = _cwtzm(s1, fs, scales, omega, zeta); D1, sr1 = _getdetails(t, s1, C1, scales, omega, zeta)
    C2 = _cwtzm(s2, fs, scales, omega, zeta); D2, sr2 = _getdetails(t, s2, C2, scales, omega, zeta)

    # --- 3. Phase 1: RotDnn PSA Matching ---
    theta = np.arange(0, 180, 1)

    # Calculate rigorous Scale Factor
    PSA180_orig, _, _ = compute_rotated_spectra(T, s1, s2, zi, dt, theta)
    PSArotnn_orig = np.percentile(PSA180_orig, nn, axis=0)
    sf = np.sum(ds[TlocsPSA]) / np.sum(PSArotnn_orig[TlocsPSA])
    
    # Apply factor to the CWT proxies
    sr1 = sf * sr1; D1 = sf * D1
    sr2 = sf * sr2; D2 = sf * D2
    
    PSA180or_proxy, _, _ = compute_rotated_spectra(T, sr1, sr2, zi, dt, theta)
    PSArotnn_proxy = np.percentile(PSA180or_proxy, nn, axis=0)

    hPSAbc = np.zeros((NS, nit+1)); ns1 = np.zeros((nt, nit+1)); ns2 = np.zeros((nt, nit+1))
    DN1 = np.zeros((NS, nt, nit+1)); DN2 = np.zeros((NS, nt, nit+1))
    
    hPSAbc[:, 0] = PSArotnn_proxy
    ns1[:, 0] = sr1; ns2[:, 0] = sr2; DN1[:, :, 0] = D1; DN2[:, :, 0] = D2
    factorPSA = np.ones((NS, 1))
    
    log.info(f"Starting RotD{nn} PSA matching ({nit} iterations)...")
    for qq in range(1, nit+1):
        ratio_update = ds[TlocsPSA] / hPSAbc[TlocsPSA, qq-1]
        factorPSA[TlocsPSA, 0] = ratio_update
        
        DN1[:, :, qq] = factorPSA * DN1[:, :, qq-1]
        DN2[:, :, qq] = factorPSA * DN2[:, :, qq-1]
        
        ns1[:, qq] = np.trapezoid(DN1[:, :, qq].T, scales)
        ns2[:, qq] = np.trapezoid(DN2[:, :, qq].T, scales)
        
        PSA180_iter, _, _ = compute_rotated_spectra(T, ns1[:, qq], ns2[:, qq], zi, dt, theta)
        hPSAbc[:, qq] = np.percentile(PSA180_iter, nn, axis=0)
        
    rmsePSA = np.linalg.norm(np.abs(hPSAbc[TlocsPSA, :] - ds[TlocsPSA][:, None]) / ds[TlocsPSA][:, None], axis=0) / np.sqrt(len(TlocsPSA)) * 100
    brloc = np.argmin(rmsePSA)
    sc1 = ns1[:, brloc]; sc2 = ns2[:, brloc]
    D1c = DN1[:, :, brloc]; D2c = DN2[:, :, brloc]
    log.info(f"Best RotD{nn} PSA match at it {brloc}, RMSE: {rmsePSA[brloc]:.2f}%")

    # --- 4. Phase 2: FAS Modification Loop ---
    sc1_fas = None; sc2_fas = None; D1c_fas = None; D2c_fas = None
    fft_freqs = np.fft.rfftfreq(nt, d=dt); fft_freqs[0] = 1e-9 
    
    def _calc_fas_rotdnn_internal(a1, a2, downsample=None):
        f_fft, fas_dict, _ = calculate_fas_rotDnn(
            a1, a2, fs, percentiles=[nn], tukey_alpha=0.1, nfft_method='same',
            detrend_method='linear', smoothing_method=smoothing_method,
            smoothing_coeff=smoothing_coeff, prefer_pykooh=prefer_pykooh,
            downsample_freqs=downsample 
        )
        return f_fft, fas_dict[nn]
        
    if do_fas:
        sc1_fas = np.copy(sc1); sc2_fas = np.copy(sc2); D1c_fas = np.copy(D1c); D2c_fas = np.copy(D2c)
        _, FAS_rotdnn_sc = _calc_fas_rotdnn_internal(sc1_fas, sc2_fas)
        MotionFAS_int = log_interp(freqs, fft_freqs, FAS_rotdnn_sc)
        ratioFAS = MotionFAS_int / TargetFAS_int
        minratioFAS = np.min(ratioFAS[TlocsCheck])
        
        if minratioFAS >= FASreduction:
            log.info("Minimum RotDnn FAS requirement already satisfied.")
        else:
            log.info(f"Starting FAS adjustment. Initial min ratio: {minratioFAS:.2f} (Target: {FASreduction})")
            cont = 0
            while minratioFAS < FASreduction and cont < maxit:
                crit_idx = np.argmin(ratioFAS[TlocsCheck]); crit_neg_loc = TlocsCheck[crit_idx]
                sf_fas = 1.01 * (TargetFAS_int[crit_neg_loc] / MotionFAS_int[crit_neg_loc])
                
                D1c_fas[crit_neg_loc, :] = sf_fas * D1c_fas[crit_neg_loc, :]
                D2c_fas[crit_neg_loc, :] = sf_fas * D2c_fas[crit_neg_loc, :]
                
                sc1_fas = np.trapezoid(D1c_fas.T, scales); sc2_fas = np.trapezoid(D2c_fas.T, scales)
                _, FAS_rotdnn_sc = _calc_fas_rotdnn_internal(sc1_fas, sc2_fas)
                MotionFAS_int = log_interp(freqs, fft_freqs, FAS_rotdnn_sc)
                ratioFAS = MotionFAS_int / TargetFAS_int
                minratioFAS = np.min(ratioFAS[TlocsCheck])
                cont += 1
            log.info(f"FAS adjustment finished at it {cont}. Final min ratio: {minratioFAS:.3f}")

    # --- 5. Phase 3: PSD Modification Loop ---
    sca1 = None; sca2 = None; Dca1 = None; Dca2 = None
    
    def _calc_psd_rotdnn_internal(a1, a2, downsample=None):
        f_fft, psd_dict, _ = calculate_psd_rotDnn(
            a1, a2, fs, percentiles=[nn], duration_percent=(5, 75), 
            tukey_alpha=0.1, nfft_method='same', detrend_method='linear', 
            smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, 
            prefer_pykooh=prefer_pykooh, downsample_freqs=downsample
        )
        return f_fft, psd_dict[nn]

    if do_psd:
        sca1 = np.copy(sc1_fas) if do_fas else np.copy(sc1)
        sca2 = np.copy(sc2_fas) if do_fas else np.copy(sc2)
        Dca1 = np.copy(D1c_fas) if do_fas else np.copy(D1c)
        Dca2 = np.copy(D2c_fas) if do_fas else np.copy(D2c)
        
        _, _, _, t1_h1, t2_h1 = SignificantDuration(sca1, t, 5, 75)
        _, _, _, t1_h2, t2_h2 = SignificantDuration(sca2, t, 5, 75)
        
        _, PSD_rotdnn_sca = _calc_psd_rotdnn_internal(sca1, sca2)
        MotionPSD_int = log_interp(freqs, fft_freqs, PSD_rotdnn_sca)
        ratioPSD = MotionPSD_int / TargetPSD_int
        minratioPSD = np.min(ratioPSD[TlocsCheck])

        if minratioPSD >= PSDreduction:
            log.info("Minimum RotDnn PSD requirement already satisfied.")
        else:
            log.info(f"Starting PSD adjustment. Initial min ratio: {minratioPSD:.2f} (Target: {PSDreduction})")
            cont = 0
            while minratioPSD < PSDreduction and cont < maxit:
                crit_idx = np.argmin(ratioPSD[TlocsCheck]); crit_neg_loc = TlocsCheck[crit_idx]
                factorn = 1.01 * (TargetPSD_int[crit_neg_loc] / MotionPSD_int[crit_neg_loc])**0.5
                
                if localized:
                    locs1 = np.where((t >= t1_h1 - dt/2) & (t <= t2_h1 + dt/2))[0]
                    window1 = np.zeros(nt); window1[locs1] = signal.windows.tukey(len(locs1), 0.1)
                    scaling_vector1 = np.maximum(factorn * window1, 1.0)
                    
                    locs2 = np.where((t >= t1_h2 - dt/2) & (t <= t2_h2 + dt/2))[0]
                    window2 = np.zeros(nt); window2[locs2] = signal.windows.tukey(len(locs2), 0.1)
                    scaling_vector2 = np.maximum(factorn * window2, 1.0)
                    
                    Dca1[crit_neg_loc, :] = scaling_vector1 * Dca1[crit_neg_loc, :]
                    Dca2[crit_neg_loc, :] = scaling_vector2 * Dca2[crit_neg_loc, :]
                else:
                    Dca1[crit_neg_loc, :] = factorn * Dca1[crit_neg_loc, :]
                    Dca2[crit_neg_loc, :] = factorn * Dca2[crit_neg_loc, :]

                sca1 = np.trapezoid(Dca1.T, scales); sca2 = np.trapezoid(Dca2.T, scales)
                _, PSD_rotdnn_sca = _calc_psd_rotdnn_internal(sca1, sca2)
                MotionPSD_int = log_interp(freqs, fft_freqs, PSD_rotdnn_sca)
                ratioPSD = MotionPSD_int / TargetPSD_int
                minratioPSD = np.min(ratioPSD[TlocsCheck])
                cont += 1
            log.info(f"PSD adjustment finished at it {cont}. Final min ratio: {minratioPSD:.3f}")

    # --- 6. Baseline Correction ---
    sc1_unc, sc2_unc = np.copy(sc1), np.copy(sc2)
    sc1_fas_unc = np.copy(sc1_fas) if do_fas else None; sc2_fas_unc = np.copy(sc2_fas) if do_fas else None
    sca1_unc = np.copy(sca1) if do_psd else None; sca2_unc = np.copy(sca2) if do_psd else None
    
    b_meth = baseline_method.lower().strip()
    if b_meth == 'classic':
        log.info("Performing classic polynomial baseline correction...")
        sc1, _, _ = baselinecorrect(sc1, t, porder=PSA_poly_order); sc2, _, _ = baselinecorrect(sc2, t, porder=PSA_poly_order)
        if do_fas: sc1_fas, _, _ = baselinecorrect(sc1_fas, t, porder=PSAPSD_poly_order); sc2_fas, _, _ = baselinecorrect(sc2_fas, t, porder=PSAPSD_poly_order)
        if do_psd: sca1, _, _ = baselinecorrect(sca1, t, porder=PSAPSD_poly_order); sca2, _, _ = baselinecorrect(sca2, t, porder=PSAPSD_poly_order)
    elif b_meth == 'piecewise':
        log.info("Performing piecewise spline baseline correction...")
        sc1 = piecewise_baseline_detrending(sc1, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp); sc2 = piecewise_baseline_detrending(sc2, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
        if do_fas: sc1_fas = piecewise_baseline_detrending(sc1_fas, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp); sc2_fas = piecewise_baseline_detrending(sc2_fas, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
        if do_psd: sca1 = piecewise_baseline_detrending(sca1, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp); sca2 = piecewise_baseline_detrending(sca2, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
    elif b_meth == 'sixth_order':
        log.info("Performing USGS 6th-order polynomial displacement baseline correction...")
        sc1 = baseline_sixth_order(sc1, t); sc2 = baseline_sixth_order(sc2, t)
        if do_fas: sc1_fas = baseline_sixth_order(sc1_fas, t); sc2_fas = baseline_sixth_order(sc2_fas, t)
        if do_psd: sca1 = baseline_sixth_order(sca1, t); sca2 = baseline_sixth_order(sca2, t)
    else:
        log.warning("Skipping baseline correction.")

    # --- 7. Metrics Extraction ---
    log.info("Calculating final metrics for output...")
    from scipy import integrate
    
    freqs_check = np.minimum(get_log_freqs(0.1, fs/2, 100), fs/2 - 1e-6); T_check = 1/freqs_check
    TargetPSA_check = log_interp(T_check, To_target, targetPSA_T)
    TargetPSD_check = log_interp(freqs_check, f_PSD, targetPSD) if do_psd else np.full_like(freqs_check, np.nan)
    TargetFAS_check = log_interp(freqs_check, f_FAS, targetFAS) if do_fas else np.full_like(freqs_check, np.nan)
    
    def calc_norm_sq(arr):
        if arr is None or np.all(np.isnan(arr)): return np.zeros_like(t)
        sq_int = integrate.cumulative_trapezoid(arr**2, t, initial=0)
        return sq_int / sq_int[-1] if sq_int[-1] != 0 else np.zeros_like(t)

    def _calc_kinematics(a1, a2):
        if a1 is None:
            n = np.full_like(t, np.nan)
            return n, n, n, n, n, n, n, n, n, n
        
        v1 = integrate.cumulative_trapezoid(a1, t, initial=0); d1 = integrate.cumulative_trapezoid(v1, t, initial=0)
        v2 = integrate.cumulative_trapezoid(a2, t, initial=0); d2 = integrate.cumulative_trapezoid(v2, t, initial=0)
        
        nai1 = calc_norm_sq(a1); ncsv1 = calc_norm_sq(v1); ncsd1 = calc_norm_sq(d1)
        nai2 = calc_norm_sq(a2); ncsv2 = calc_norm_sq(v2); ncsd2 = calc_norm_sq(d2)
        return v1, v2, d1, d2, nai1, nai2, ncsv1, ncsv2, ncsd1, ncsd2

    def _calc_spectra(a1, a2):
        if a1 is None:
            n = np.full_like(freqs_check, np.nan); m = np.zeros((180, len(freqs_check)))
            return np.full_like(T_check, np.nan), n, n, m
        psa180, _, _ = compute_rotated_spectra(T_check, a1, a2, zi, dt, theta)
        psa = np.percentile(psa180, nn, axis=0)
        
        _, fas_f = _calc_fas_rotdnn_internal(a1, a2, downsample=freqs_check)
        _, psd_f = _calc_psd_rotdnn_internal(a1, a2, downsample=freqs_check)
        
        return psa, psd_f, fas_f, psa180

    s1_scaled = s1 * sf; s2_scaled = s2 * sf
    
    psa_s, psd_s, fas_s, p180_s = _calc_spectra(s1_scaled, s2_scaled)
    psa_sc, psd_sc, fas_sc, p180_sc = _calc_spectra(sc1, sc2)
    psa_sc_f, psd_sc_f, fas_sc_f, p180_sc_f = _calc_spectra(sc1_fas, sc2_fas)
    psa_sca, psd_sca, fas_sca, p180_sca = _calc_spectra(sca1, sca2)
    
    v1_s, v2_s, d1_s, d2_s, nai1_s, nai2_s, ncsv1_s, ncsv2_s, ncsd1_s, ncsd2_s = _calc_kinematics(s1_scaled, s2_scaled)
    v1_sc, v2_sc, d1_sc, d2_sc, nai1_sc, nai2_sc, ncsv1_sc, ncsv2_sc, ncsd1_sc, ncsd2_sc = _calc_kinematics(sc1, sc2)
    v1_sc_f, v2_sc_f, d1_sc_f, d2_sc_f, nai1_sc_f, nai2_sc_f, ncsv1_sc_f, ncsv2_sc_f, ncsd1_sc_f, ncsd2_sc_f = _calc_kinematics(sc1_fas, sc2_fas)
    v1_sca, v2_sca, d1_sca, d2_sca, nai1_sca, nai2_sca, ncsv1_sca, ncsv2_sca, ncsd1_sca, ncsd2_sca = _calc_kinematics(sca1, sca2)

    return {
        't': t, 'dt': dt, 'freqs': freqs_check, 'periods': T_check, 'mode': mode, 'nn': nn, 'scale_factor': sf,
        
        's1_scaled': s1_scaled, 'sc1_unc': sc1_unc, 'sc1_fas_unc': sc1_fas_unc, 'sca1_unc': sca1_unc,
        'sc1': sc1, 'sc1_fas': sc1_fas, 'sca1': sca1,
        'vel1_s': v1_s, 'vel1_sc': v1_sc, 'vel1_sc_fas': v1_sc_f, 'vel1_sca': v1_sca,
        'disp1_s': d1_s, 'disp1_sc': d1_sc, 'disp1_sc_fas': d1_sc_f, 'disp1_sca': d1_sca,
        'ai1_s': nai1_s, 'ai1_sc': nai1_sc, 'ai1_sc_fas': nai1_sc_f, 'ai1_sca': nai1_sca,
        'csv1_s': ncsv1_s, 'csv1_sc': ncsv1_sc, 'csv1_sc_fas': ncsv1_sc_f, 'csv1_sca': ncsv1_sca,
        'csd1_s': ncsd1_s, 'csd1_sc': ncsd1_sc, 'csd1_sc_fas': ncsd1_sc_f, 'csd1_sca': ncsd1_sca,
        
        's2_scaled': s2_scaled, 'sc2_unc': sc2_unc, 'sc2_fas_unc': sc2_fas_unc, 'sca2_unc': sca2_unc,
        'sc2': sc2, 'sc2_fas': sc2_fas, 'sca2': sca2,
        'vel2_s': v2_s, 'vel2_sc': v2_sc, 'vel2_sc_fas': v2_sc_f, 'vel2_sca': v2_sca,
        'disp2_s': d2_s, 'disp2_sc': d2_sc, 'disp2_sc_fas': d2_sc_f, 'disp2_sca': d2_sca,
        'ai2_s': nai2_s, 'ai2_sc': nai2_sc, 'ai2_sc_fas': nai2_sc_f, 'ai2_sca': nai2_sca,
        'csv2_s': ncsv2_s, 'csv2_sc': ncsv2_sc, 'csv2_sc_fas': ncsv2_sc_f, 'csv2_sca': ncsv2_sca,
        'csd2_s': ncsd2_s, 'csd2_sc': ncsd2_sc, 'csd2_sc_fas': ncsd2_sc_f, 'csd2_sca': ncsd2_sca,

        'psa_s': psa_s, 'psa_sc': psa_sc, 'psa_sc_fas': psa_sc_f, 'psa_sca': psa_sca,
        'psd_s': psd_s, 'psd_sc': psd_sc, 'psd_sc_fas': psd_sc_f, 'psd_sca': psd_sca,
        'fas_s': fas_s, 'fas_sc': fas_sc, 'fas_sc_fas': fas_sc_f, 'fas_sca': fas_sca,
        'target_psa': TargetPSA_check, 'target_psd': TargetPSD_check, 'target_fas': TargetFAS_check,
        
        'psa_180_seed': p180_s, 'psa_180_sc': p180_sc, 'psa_180_fas': p180_sc_f, 'psa_180_final': p180_sca
    }

def generate_single_component_psa_fas_psd_compatible_record(
    s: np.ndarray,
    fs: float,
    f_PSA: np.ndarray,
    targetPSA: np.ndarray,
    f_PSD: np.ndarray = None,
    targetPSD: np.ndarray = None,
    f_FAS: np.ndarray = None,
    targetFAS: np.ndarray = None,
    adjustment_mode: str = 'both', 
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    PSDreduction: float = 1.0,
    FASreduction: float = 1.0,
    targetPGA: float = -1,
    F1PSA: float = 0.2,
    F2PSA: float = 50.0,
    F1Check: float = 0.3,
    F2Check: float = 30.0,
    zi: float = 0.05,
    baseline_method: str = 'sixth_order',
    PSA_poly_order: int = 4,
    PSAPSD_poly_order: int = 4,
    BL_target_disp: float = 0.0,
    knlocs_ai_pct: List[float] = [1.0, 5.0, 40.0, 75.0, 95.0, 99.0],
    NS: int = 300,
    nit: int = 30,
    maxit: int = 1000,
    localized: bool = True,
    smoothing_method: str = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    prefer_pykooh: bool = True) -> Dict[str, Any]:
    
    """
    Unified function to generate a record compatible with a Target PSA, 
    and optionally minimum Target FAS, and minimum Target PSD.

    Parameters
    ----------
    s : numpy.ndarray
        Seed acceleration time-series.
    fs : float
        Sampling frequency of the seed record (Hz).
    f_PSA, targetPSA : numpy.ndarray
        Frequency array (Hz) and target amplitudes for PSA.
    f_PSD, targetPSD : numpy.ndarray, optional
        Frequency array (Hz) and target amplitudes for PSD.
    f_FAS, targetFAS : numpy.ndarray, optional
        Frequency array (Hz) and target amplitudes for FAS.
    adjustment_mode : str, optional
        Selects the adjustment stages to run after PSA matching. 
        Options: 'psd', 'fas', or 'both'. Default is 'both'.
    targetPSAlimits : tuple of float, optional
        Acceptable matching limits (min, max) for the PSA ratio.
    PSDreduction : float, optional
        Minimum required PSD level relative to the target (e.g., 0.7).
    FASreduction : float, optional
        Minimum required FAS level relative to the target.
    targetPGA : float, optional
        Target Peak Ground Acceleration. Default is -1 (no correction).
    F1PSA, F2PSA : float, optional
        Frequency range (Hz) for PSA matching. Default is 0.2 to 50.0.
    F1Check, F2Check : float, optional
        Frequency range (Hz) to check and adjust both FAS and PSD. 
    zi : float, optional
        Damping ratio for response spectrum calculation. Default is 0.05.
    baseline_method : str, optional
        Method for baseline correction: 'none', 'classic', 'piecewise', 'sixth_order'. 
    PSA_poly_order, PSAPSD_poly_order : int, optional
        Polynomial orders for 'classic' baseline correction.
    BL_target_disp : float, optional
        Target final displacement for 'piecewise' baseline correction.
    knlocs_ai_pct : List[float], optional
        Knot locations for 'piecewise' correction.
    NS : int, optional
        Number of scales/frequencies used for the CWT. Default is 300.
    nit, maxit : int, optional
        Iterations for PSA and FAS/PSD phases.
    localized : bool, optional
        If True, PSD adjustments are applied only to the strong motion portion (SD5-75).
    smoothing_method, smoothing_coeff, prefer_pykooh: 
        Smoothing configurations for spectral calculations.

    Returns
    -------
    results : Dict[str, Any]
        A comprehensive dictionary containing the matching results, time histories, and 
        spectral metrics for all evaluated stages. If a specific adjustment stage (FAS or PSD) 
        is skipped based on the `adjustment_mode`, its corresponding time-history keys will 
        return `None`, and its spectral/metric array keys will return arrays padded with `NaN`.
        
        Keys included:
        - 'mode' (str): The adjustment mode executed ('psd', 'fas', or 'both').
        - 't', 'dt' (np.ndarray, float): Time vector (s) and time step.
        - 'freqs', 'periods' (np.ndarray): Frequency (Hz) and period (s) vectors used for evaluation.
        
        Time Histories (Accelerations in input units):
        - 's_scaled': Seed record linearly scaled to minimize PSA error in the target range.
        - 'sc': Baseline-corrected record after the PSA matching phase.
        - 'sc_fas': Baseline-corrected record after the FAS adjustment phase.
        - 'sca': Final baseline-corrected record after the PSD adjustment phase.
        - 'sc_unc', 'sc_fas_unc', 'sca_unc': Uncorrected versions (before baseline correction) of the above.
        
        Kinematics & Energy Metrics (vel, disp, ai, cav):
        - 'vel_s', 'vel_sc', 'vel_sc_fas', 'vel_sca': Velocity time histories.
        - 'disp_s', 'disp_sc', 'disp_sc_fas', 'disp_sca': Displacement time histories.
        - 'ai_s', 'ai_sc', 'ai_sc_fas', 'ai_sca': Arias Intensity buildup over time.
        - 'cav_s', 'cav_sc', 'cav_sc_fas', 'cav_sca': Cumulative Absolute Velocity buildup.
        
        Spectra (PSA, PSD, FAS):
        - 'psa_s', 'psa_sc', 'psa_sc_fas', 'psa_sca': Computed Pseudo-Spectral Accelerations.
        - 'psd_s', 'psd_sc', 'psd_sc_fas', 'psd_sca': Computed Power Spectral Densities.
        - 'fas_s', 'fas_sc', 'fas_sc_fas', 'fas_sca': Computed Fourier Amplitude Spectra.
        
        Targets:
        - 'target_psa': Target PSA interpolated to the 'periods' vector.
        - 'target_psd': Target PSD interpolated to the 'freqs' vector.
        - 'target_fas': Target FAS interpolated to the 'freqs' vector.
    """
    
    # --- Input Validation & Routing ---
    mode = adjustment_mode.lower().strip()
    if mode not in ['psd', 'fas', 'both']:
        raise ValueError("adjustment_mode must be 'psd', 'fas', or 'both'")
        
    do_fas = mode in ['fas', 'both']
    do_psd = mode in ['psd', 'both']
    
    if do_psd and (f_PSD is None or targetPSD is None):
        raise ValueError("targetPSD and f_PSD must be provided when adjustment_mode includes 'psd'")
    if do_fas and (f_FAS is None or targetFAS is None):
        raise ValueError("targetFAS and f_FAS must be provided when adjustment_mode includes 'fas'")

    pi = np.pi
    dt = 1/fs
    nt = len(s)
    if nt % 2 != 0: 
        s = np.append(s, 0); nt += 1
    t = np.linspace(0, (nt-1)*dt, nt)
    
    # --- 1. Target Interpolation Prep (BUG FIX: Sort arrays properly) ---
    idx_f = np.argsort(f_PSA)
    f_PSA = f_PSA[idx_f]; targetPSA = targetPSA[idx_f]
    To_target = 1.0 / f_PSA; idx_T = np.argsort(To_target)
    To_target = To_target[idx_T]; targetPSA_T = targetPSA[idx_T]
    
    if do_psd:
        idx_psd = np.argsort(f_PSD)
        f_PSD = f_PSD[idx_psd]
        targetPSD = targetPSD[idx_psd]
        
    if do_fas:
        idx_fas = np.argsort(f_FAS)
        f_FAS = f_FAS[idx_fas]
        targetFAS = targetFAS[idx_fas]
    
    T1PSA, T2PSA = 1.0/F2PSA, 1.0/F1PSA
    T1Check, T2Check = 1.0/F2Check, 1.0/F1Check
    
    FF1 = min(4/(nt*dt), 0.1); FF2 = 1/(2*dt)
    T1PSA, T2PSA, FF1 = _CheckPeriodRange(T1PSA, T2PSA, To_target, FF1, FF2)

    omega = pi; zeta = 0.05
    freqs = np.geomspace(FF2, FF1, NS) 
    T = 1/freqs; scales = omega / (2*pi*freqs)
    
    ds = log_interp(T, To_target, targetPSA_T)
    TlocsPSA = np.where((T >= T1PSA) & (T <= T2PSA))[0]
    TlocsCheck = np.where((T >= T1Check) & (T <= T2Check))[0] 
    
    TargetPSD_int = log_interp(freqs, f_PSD, targetPSD) if do_psd else None
    TargetFAS_int = log_interp(freqs, f_FAS, targetFAS) if do_fas else None

    # --- 2. Wavelet Decomposition ---
    log.info(f"Performing CWT decomposition (Mode: {mode.upper()})...")
    C = _cwtzm(s, fs, scales, omega, zeta)
    D, sr = _getdetails(t, s, C, scales, omega, zeta)

    # --- 3. Phase 1: PSA Matching (Always Runs) ---
    hPSAbc = np.zeros((NS, nit+1)); ns = np.zeros((nt, nit+1)); DN = np.zeros((NS, nt, nit+1))
    hPSAbc[:, 0], _, _ = compute_spectrum(T, sr, zi, dt); ns[:, 0] = sr; DN[:, :, 0] = D
    factorPSA = np.ones((NS, 1))
    
    log.info(f"Starting PSA matching ({nit} iterations)...")
    for qq in range(1, nit+1):
        factorPSA[TlocsPSA, 0] = ds[TlocsPSA] / hPSAbc[TlocsPSA, qq-1]
        DN[:, :, qq] = factorPSA * DN[:, :, qq-1]
        ns[:, qq] = np.trapezoid(DN[:, :, qq].T, scales)
        hPSAbc[:, qq], _, _ = compute_spectrum(T, ns[:, qq], zi, dt)
        
    rmsePSA = np.linalg.norm(np.abs(hPSAbc[TlocsPSA, :] - ds[TlocsPSA][:, None]) / ds[TlocsPSA][:, None], axis=0) / np.sqrt(len(TlocsPSA)) * 100
    brloc = np.argmin(rmsePSA)
    sc = ns[:, brloc]; Dc = DN[:, :, brloc]
    log.info(f"Best PSA match at it {brloc}, RMSE: {rmsePSA[brloc]:.2f}%")

    # --- 4. Phase 2: FAS Modification Loop ---
    sc_fas = None; Dc_fas = None
    fft_freqs = np.fft.rfftfreq(len(sc), d=dt); fft_freqs[0] = 1e-9 
    
    if do_fas:
        sc_fas = np.copy(sc); Dc_fas = np.copy(Dc)
        _, _, _, FAS_smooth_sc = calculate_earthquake_fas(sc_fas, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=None)
        MotionFAS_int = log_interp(freqs, fft_freqs, FAS_smooth_sc)
        ratioFAS = MotionFAS_int / TargetFAS_int
        minratioFAS = np.min(ratioFAS[TlocsCheck])
        
        if minratioFAS < FASreduction:
            log.info(f"Starting FAS adjustment. Initial min ratio: {minratioFAS:.2f} (Target: {FASreduction})")
            cont = 0
            while minratioFAS < FASreduction and cont < maxit:
                crit_idx = np.argmin(ratioFAS[TlocsCheck]); crit_neg_loc = TlocsCheck[crit_idx]
                Dc_fas[crit_neg_loc, :] = 1.01 * (TargetFAS_int[crit_neg_loc] / MotionFAS_int[crit_neg_loc]) * Dc_fas[crit_neg_loc, :]
                sc_fas = np.trapezoid(Dc_fas.T, scales)
                _, _, _, FAS_smooth_sc = calculate_earthquake_fas(sc_fas, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=None)
                MotionFAS_int = log_interp(freqs, fft_freqs, FAS_smooth_sc)
                ratioFAS = MotionFAS_int / TargetFAS_int
                minratioFAS = np.min(ratioFAS[TlocsCheck])
                cont += 1
            log.info(f"FAS adjustment finished at it {cont}. Final min ratio: {minratioFAS:.3f}")
        else:
            log.info("Minimum FAS requirement already satisfied.")

    # --- 5. Phase 3: PSD Modification Loop ---
    sca = None; Dca = None
    
    if do_psd:
        # Base the PSD adjustment on the previous stage's result
        sca = np.copy(sc_fas) if do_fas else np.copy(sc)
        Dca = np.copy(Dc_fas) if do_fas else np.copy(Dc)
        
        _, _, _, t1_h1, t2_h1 = SignificantDuration(sca, t, 5, 75)
        _, _, _, _, _, PSDavg_sca = calculate_earthquake_psd(sca, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=None)
        MotionPSD_int = log_interp(freqs, fft_freqs, PSDavg_sca)
        ratio = MotionPSD_int / TargetPSD_int
        minratioPSD = np.min(ratio[TlocsCheck])

        if minratioPSD < PSDreduction:
            log.info(f"Starting PSD adjustment. Initial min ratio: {minratioPSD:.2f} (Target: {PSDreduction})")
            cont = 0
            while minratioPSD < PSDreduction and cont < maxit:
                crit_idx = np.argmin(ratio[TlocsCheck]); crit_neg_loc = TlocsCheck[crit_idx]
                factorn = 1.01 * (TargetPSD_int[crit_neg_loc] / MotionPSD_int[crit_neg_loc])**0.5
                
                if localized:
                    locs1 = np.where((t >= t1_h1 - dt/2) & (t <= t2_h1 + dt/2))[0]
                    window = np.zeros(nt); window[locs1] = signal.windows.tukey(len(locs1), 0.1)
                    Dca[crit_neg_loc, :] = np.maximum(factorn * window, 1.0) * Dca[crit_neg_loc, :]
                else:
                    Dca[crit_neg_loc, :] = factorn * Dca[crit_neg_loc, :]

                sca = np.trapezoid(Dca.T, scales)
                _, _, _, _, _, PSDavg_sca = calculate_earthquake_psd(sca, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=None)
                MotionPSD_int = log_interp(freqs, fft_freqs, PSDavg_sca)
                ratio = MotionPSD_int / TargetPSD_int
                minratioPSD = np.min(ratio[TlocsCheck])
                cont += 1
            log.info(f"PSD adjustment finished at it {cont}. Final min ratio: {minratioPSD:.3f}")
        else:
            log.info("Minimum PSD requirement already satisfied.")

    # --- 6. Final Corrections (PGA & Baseline) ---
    if targetPGA != -1:
        log.info(f"Correcting PGA to {targetPGA}g...")
        if do_psd: sca = pga_correction(targetPGA, t, sca)
        if do_fas: sc_fas = pga_correction(targetPGA, t, sc_fas)
        
    sc_unc = np.copy(sc)
    sc_fas_unc = np.copy(sc_fas) if do_fas else None
    sca_unc = np.copy(sca) if do_psd else None
    
    b_meth = baseline_method.lower().strip()
    if b_meth == 'classic':
        log.info("Performing classic polynomial baseline correction...")
        sc, _, _ = baselinecorrect(sc, t, porder=PSA_poly_order)
        if do_fas: sc_fas, _, _ = baselinecorrect(sc_fas, t, porder=PSAPSD_poly_order)
        if do_psd: sca, _, _ = baselinecorrect(sca, t, porder=PSAPSD_poly_order)
    elif b_meth == 'piecewise':
        log.info("Performing piecewise spline baseline correction...")
        sc = piecewise_baseline_detrending(sc, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
        if do_fas: sc_fas = piecewise_baseline_detrending(sc_fas, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
        if do_psd: sca = piecewise_baseline_detrending(sca, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
    elif b_meth == 'sixth_order':
        log.info("Performing USGS 6th-order polynomial displacement baseline correction...")
        sc = baseline_sixth_order(sc, t)
        if do_fas: sc_fas = baseline_sixth_order(sc_fas, t)
        if do_psd: sca = baseline_sixth_order(sca, t)

    # --- 7. Final Scaling Checks ---
    freqs_check = np.minimum(get_log_freqs(0.1, fs/2, 100), fs/2 - 1e-6); T_check = 1/freqs_check
    TargetPSA_check = log_interp(T_check, To_target, targetPSA_T)
    TargetPSD_check = log_interp(freqs_check, f_PSD, targetPSD) if do_psd else np.full_like(freqs_check, np.nan)
    TargetFAS_check = log_interp(freqs_check, f_FAS, targetFAS) if do_fas else np.full_like(freqs_check, np.nan)
    
    TlocsPSA_chk = np.where((T_check >= T1PSA) & (T_check <= T2PSA))[0]
    FlocsCheck_chk = np.where((freqs_check >= F1Check) & (freqs_check <= F2Check))[0]

    # Reusable logic to check and scale final records
    def _apply_final_scaling(acc_trace, label_name):
        if acc_trace is None: return acc_trace
        psa_tmp, _, _ = compute_spectrum(T_check, acc_trace, zi, dt)
        f_psa = np.ones(len(T_check)); f_psd = np.ones(len(freqs_check)); f_fas = np.ones(len(freqs_check))
        
        if len(TlocsPSA_chk) > 0:
            f_psa[TlocsPSA_chk] = targetPSAlimits[0] * TargetPSA_check[TlocsPSA_chk] / psa_tmp[TlocsPSA_chk]
        if do_psd and len(FlocsCheck_chk) > 0:
            r_psd = calculate_earthquake_psd(acc_trace, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check)
            p_chk = log_interp(freqs_check, fft_freqs, r_psd[5]) if smoothing_method == 'variable_window' else r_psd[5]
            f_psd[FlocsCheck_chk] = (PSDreduction * TargetPSD_check[FlocsCheck_chk] / p_chk[FlocsCheck_chk])**0.5
        if do_fas and len(FlocsCheck_chk) > 0:
            _, _, _, f_chk = calculate_earthquake_fas(acc_trace, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check)
            f_fas[FlocsCheck_chk] = (FASreduction * TargetFAS_check[FlocsCheck_chk] / f_chk[FlocsCheck_chk])
        
        f_max = np.max(np.hstack((f_psa, f_psd, f_fas)))
        if f_max > 1.0:
            log.info(f"Final scaling required for {label_name} record: {f_max:.3f}")
            return f_max * acc_trace
        return acc_trace

    if do_psd: sca = _apply_final_scaling(sca, "SCA (PSD)")
    elif do_fas: sc_fas = _apply_final_scaling(sc_fas, "SC_FAS (FAS)")
    else: sc = _apply_final_scaling(sc, "SC (PSA Only)")

    # --- 8. Metrics Extraction ---
    log.info("Calculating final metrics for output...")
    
    def _calc_metrics(acc):
        # Gracefully handle skipped stages by padding with NaNs
        if acc is None:
            return (np.full_like(T_check, np.nan), np.full_like(freqs_check, np.nan), np.full_like(freqs_check, np.nan),
                    np.full_like(t, np.nan), np.full_like(t, np.nan), np.full_like(t, np.nan), np.full_like(t, np.nan))
            
        vel = integrate.cumulative_trapezoid(acc, t, initial=0); disp = integrate.cumulative_trapezoid(vel, t, initial=0)
        ai = integrate.cumulative_trapezoid(acc**2, t, initial=0); cav = integrate.cumulative_trapezoid(np.abs(vel), t, initial=0) 
        psa, _, _ = compute_spectrum(T_check, acc, zi, dt)
        r_psd = calculate_earthquake_psd(acc, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check)
        psd_f = log_interp(freqs_check, fft_freqs, r_psd[5]) if smoothing_method == 'variable_window' else r_psd[5]
        _, _, _, fas_f = calculate_earthquake_fas(acc, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check)
        return psa, psd_f, fas_f, vel, disp, ai, cav

    psa_seed_raw, _, _ = compute_spectrum(T_check, s, zi, dt)
    s_scaled = s * (np.sum(TargetPSA_check[TlocsPSA_chk]) / np.sum(psa_seed_raw[TlocsPSA_chk]))
    
    psa_s, psd_s, fas_s, v_s, d_s, ai_s, cav_s = _calc_metrics(s_scaled)
    psa_sc, psd_sc, fas_sc, v_sc, d_sc, ai_sc, cav_sc = _calc_metrics(sc)
    psa_sc_f, psd_sc_f, fas_sc_f, v_sc_f, d_sc_f, ai_sc_f, cav_sc_f = _calc_metrics(sc_fas)
    psa_sca, psd_sca, fas_sca, v_sca, d_sca, ai_sca, cav_sca = _calc_metrics(sca)

    return {
        't': t, 'dt': dt, 'freqs': freqs_check, 'periods': T_check, 'mode': mode,
        's_scaled': s_scaled, 'sc_unc': sc_unc, 'sc_fas_unc': sc_fas_unc, 'sca_unc': sca_unc,
        'sc': sc, 'sc_fas': sc_fas, 'sca': sca,
        'psa_s': psa_s, 'psa_sc': psa_sc, 'psa_sc_fas': psa_sc_f, 'psa_sca': psa_sca,
        'psd_s': psd_s, 'psd_sc': psd_sc, 'psd_sc_fas': psd_sc_f, 'psd_sca': psd_sca,
        'fas_s': fas_s, 'fas_sc': fas_sc, 'fas_sc_fas': fas_sc_f, 'fas_sca': fas_sca,
        'vel_s': v_s, 'vel_sc': v_sc, 'vel_sc_fas': v_sc_f, 'vel_sca': v_sca,
        'disp_s': d_s, 'disp_sc': d_sc, 'disp_sc_fas': d_sc_f, 'disp_sca': d_sca,
        'ai_s': ai_s, 'ai_sc': ai_sc, 'ai_sc_fas': ai_sc_f, 'ai_sca': ai_sca,
        'cav_s': cav_s, 'cav_sc': cav_sc, 'cav_sc_fas': cav_sc_f, 'cav_sca': cav_sca,
        'target_psa': TargetPSA_check, 'target_psd': TargetPSD_check, 'target_fas': TargetFAS_check
    }

def generate_rotdnn_compatible_record(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    T_PSA: np.ndarray,
    targetPSA: np.ndarray,
    nn: int = 100,
    targetPSAlimits: Tuple[float, float] = (0.9, 1.1),
    T1PSA: float = 0.02,
    T2PSA: float = 5.0,
    zi: float = 0.05,
    baseline_method: str = 'sixth_order',
    PSA_poly_order: int = 4,
    BL_target_disp: float = 0.0,
    knlocs_ai_pct: List[float] = [1.0, 5.0, 40.0, 75.0, 95.0, 99.0],
    NS: int = 300,
    nit: int = 16) -> Dict[str, Any]:
    
    """
    Generates a pair of horizontal ground motion components compatible with 
    an orientation-independent target response spectrum (RotDnn).

    Parameters
    ----------
    s1, s2 : numpy.ndarray
        Horizontal seed acceleration time-series pairs (must be same length).
    fs : float
        Sampling frequency of the seed records (Hz).
    T_PSA : numpy.ndarray
        Array of periods (s) at which the target PSA is defined.
    targetPSA : numpy.ndarray
        Target pseudo-acceleration response spectrum (RotDnn).
    nn : int, optional
        Percentile for orientation-independent metrics (e.g., 50 for RotD50, 
        100 for RotD100). Default is 100.
    targetPSAlimits : Tuple[float, float], optional
        Tolerance limits (min, max) for PSA matching. Default is (0.9, 1.1).
    T1PSA, T2PSA : float, optional
        Period range (s) for PSA matching. Default is 0.02 to 5.0.
    zi : float, optional
        Damping ratio for the response spectra. Default is 0.05.
    baseline_method : str, optional
        Method for baseline correction: 'none', 'classic', 'piecewise', 'sixth_order'. 
        Default is 'sixth_order'.
    PSA_poly_order : int, optional
        Polynomial order for 'classic' baseline correction. Default is 4.
    BL_target_disp : float, optional
        Target final displacement for 'piecewise' baseline correction. Default is 0.0.
    knlocs_ai_pct : List[float], optional
        Knot locations (in Arias Intensity percentiles) for 'piecewise' correction.
    NS : int, optional
        Number of scales for the CWT. Default is 300.
    nit : int, optional
        Number of iterations for the initial PSA matching. Default is 16.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing matched time histories, uncorrected histories, 
        and spectral metrics.
        
        General Keys:
        - 'mode' (str): The adjustment mode executed (always 'psa' here).
        - 't', 'dt' (np.ndarray, float): Time vector (s) and time step.
        - 'freqs', 'periods' (np.ndarray): Evaluation frequency and period vectors.
        - 'nn', 'scale_factor' (int, float): RotDnn percentile and rigorous initial scale factor.
        - 'target_psa' (np.ndarray): Target PSA interpolated to the 'periods' vector.
        
        For Both Components (i = 1 and 2):
        - 's{i}_scaled': Scaled seed record.
        - 'sc{i}_unc': Uncorrected matched record.
        - 'sc{i}': Baseline-corrected matched record.
        - 'vel{i}_s', 'vel{i}_sc': Velocity time histories (scaled vs matched).
        - 'disp{i}_s', 'disp{i}_sc': Displacement time histories (scaled vs matched).
        - 'ai{i}_s', 'ai{i}_sc': Normalized Arias Intensity histories.
        - 'csv{i}_s', 'csv{i}_sc': Normalized Cumulative Squared Velocity histories.
        - 'csd{i}_s', 'csd{i}_sc': Normalized Cumulative Squared Displacement histories.
        
        Combined and Individual Spectra:
        - 'psa_s', 'psa_sc': Computed RotDnn PSA (scaled vs matched).
        - 'psa_s1', 'psa_s2': Computed individual component PSA for the scaled seed.
        - 'psa_sc1', 'psa_sc2': Computed individual component PSA for the matched record.
        - 'psa_180_seed', 'psa_180_sc': 180-degree PSA matrices.
        
        Error Metrics:
        - 'rmsefin' (float): Final Root Mean Square Error (%) prior to baseline correction.
        - 'meanefin' (float): Final Mean Error/Misfit (%) prior to baseline correction.
    """
    
    pi = np.pi; dt = 1/fs; n1 = len(s1); n2 = len(s2); nt = min(n1, n2)
    
    if n1 != n2:
        log.info(f"Records have different lengths ({n1} and {n2}). Truncating to {nt} points.")
        
    s1 = s1[:nt]; s2 = s2[:nt]
    if nt % 2 != 0: 
        s1 = np.append(s1, 0); s2 = np.append(s2, 0); nt += 1
    t = np.linspace(0, (nt-1)*dt, nt)
    
    # --- 1. Target Interpolation Prep (Period Based) ---
    idx_T = np.argsort(T_PSA)
    To_target = T_PSA[idx_T]; targetPSA_T = targetPSA[idx_T]
    
    FF1 = min(4/(nt*dt), 0.1); FF2 = 1/(2*dt)
    T1PSA, T2PSA, FF1 = _CheckPeriodRange(T1PSA, T2PSA, To_target, FF1, FF2)
    
    log.info(f"Final matching period range set to: [{T1PSA:.3f}s, {T2PSA:.3f}s]")

    omega = pi; zeta = 0.05
    freqs = np.geomspace(FF2, FF1, NS); T = 1/freqs; scales = omega / (2*pi*freqs)
    
    ds = log_interp(T, To_target, targetPSA_T)
    TlocsPSA = np.where((T >= T1PSA) & (T <= T2PSA))[0]
    
    # --- 2. Wavelet Decomposition ---
    log.info(f"Performing CWT decomposition (RotD{nn}, PSA Only)...")
    C1 = _cwtzm(s1, fs, scales, omega, zeta); D1, sr1 = _getdetails(t, s1, C1, scales, omega, zeta)
    C2 = _cwtzm(s2, fs, scales, omega, zeta); D2, sr2 = _getdetails(t, s2, C2, scales, omega, zeta)

    # --- 3. Phase 1: RotDnn PSA Matching ---
    theta = np.arange(0, 180, 1)

    PSA180_orig, _, _ = compute_rotated_spectra(T, s1, s2, zi, dt, theta)
    PSArotnn_orig = np.percentile(PSA180_orig, nn, axis=0)
    sf = np.sum(ds[TlocsPSA]) / np.sum(PSArotnn_orig[TlocsPSA])
    
    log.info(f"Initial scaling factor (rigorous RotD{nn}): {sf:.4f}")
    
    sr1 = sf * sr1; D1 = sf * D1
    sr2 = sf * sr2; D2 = sf * D2
    
    PSA180or_proxy, _, _ = compute_rotated_spectra(T, sr1, sr2, zi, dt, theta)
    PSArotnn_proxy = np.percentile(PSA180or_proxy, nn, axis=0)

    hPSAbc = np.zeros((NS, nit+1)); ns1 = np.zeros((nt, nit+1)); ns2 = np.zeros((nt, nit+1))
    DN1 = np.zeros((NS, nt, nit+1)); DN2 = np.zeros((NS, nt, nit+1))
    
    hPSAbc[:, 0] = PSArotnn_proxy
    ns1[:, 0] = sr1; ns2[:, 0] = sr2; DN1[:, :, 0] = D1; DN2[:, :, 0] = D2
    factorPSA = np.ones((NS, 1))
    
    meane = np.zeros(nit + 1)
    rmse = np.zeros(nit + 1)
    
    diff_0 = np.abs(hPSAbc[TlocsPSA, 0] - ds[TlocsPSA]) / ds[TlocsPSA]
    meane[0] = np.mean(diff_0) * 100
    rmse[0] = np.linalg.norm(diff_0) / np.sqrt(len(TlocsPSA)) * 100
    
    log.info(f"Starting RotD{nn} PSA matching ({nit} iterations)...")
    log.info(f"Iteration  0: RMSE={rmse[0]:.2f}%, Misfit={meane[0]:.2f}%")

    for qq in range(1, nit+1):
        ratio_update = ds[TlocsPSA] / hPSAbc[TlocsPSA, qq-1]
        factorPSA[TlocsPSA, 0] = ratio_update
        
        DN1[:, :, qq] = factorPSA * DN1[:, :, qq-1]
        DN2[:, :, qq] = factorPSA * DN2[:, :, qq-1]
        
        ns1[:, qq] = np.trapezoid(DN1[:, :, qq].T, scales)
        ns2[:, qq] = np.trapezoid(DN2[:, :, qq].T, scales)
        
        PSA180_iter, _, _ = compute_rotated_spectra(T, ns1[:, qq], ns2[:, qq], zi, dt, theta)
        hPSAbc[:, qq] = np.percentile(PSA180_iter, nn, axis=0)
        
        diff_qq = np.abs(hPSAbc[TlocsPSA, qq] - ds[TlocsPSA]) / ds[TlocsPSA]
        meane[qq] = np.mean(diff_qq) * 100
        rmse[qq] = np.linalg.norm(diff_qq) / np.sqrt(len(TlocsPSA)) * 100
        log.info(f"Iteration {qq:2d}: RMSE={rmse[qq]:.2f}%, Misfit={meane[qq]:.2f}%")
        
    brloc = np.argmin(rmse)
    sc1 = ns1[:, brloc]; sc2 = ns2[:, brloc]
    meanefin = meane[brloc]
    rmsefin = rmse[brloc]
    
    log.info(f"Best RotD{nn} PSA match at it {brloc}, RMSE: {rmsefin:.2f}%")
    log.info(f"Final RMSE (pre-BC): {rmsefin:.2f}%")
    log.info(f"Final Misfit (pre-BC): {meanefin:.2f}%")

    # --- 4. Baseline Correction ---
    sc1_unc, sc2_unc = np.copy(sc1), np.copy(sc2)
    
    b_meth = baseline_method.lower().strip()
    if b_meth == 'classic':
        log.info("Performing classic polynomial baseline correction...")
        sc1, _, _ = baselinecorrect(sc1, t, porder=PSA_poly_order)
        sc2, _, _ = baselinecorrect(sc2, t, porder=PSA_poly_order)
    elif b_meth == 'piecewise':
        log.info("Performing piecewise spline baseline correction...")
        sc1 = piecewise_baseline_detrending(sc1, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
        sc2 = piecewise_baseline_detrending(sc2, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
    elif b_meth == 'sixth_order':
        log.info("Performing USGS 6th-order polynomial displacement baseline correction...")
        sc1 = baseline_sixth_order(sc1, t); sc2 = baseline_sixth_order(sc2, t)
    elif b_meth == 'none':
        log.info("Skipping baseline correction as requested.")
    else:
        log.warning(f"Unknown baseline_method '{baseline_method}'. Skipping baseline correction.")

    if b_meth != 'none':
        PSA180_final_bc, _, _ = compute_rotated_spectra(T, sc1, sc2, zi, dt, theta)
        PSArotnn_bc = np.percentile(PSA180_final_bc, nn, axis=0)
        diff_bc = np.abs(PSArotnn_bc[TlocsPSA] - ds[TlocsPSA]) / ds[TlocsPSA]
        meanefin_bc = np.mean(diff_bc) * 100
        rmsefin_bc = np.linalg.norm(diff_bc) / np.sqrt(len(TlocsPSA)) * 100
        log.info(f"After Baseline Correction: RMSE={rmsefin_bc:.2f}%, Misfit={meanefin_bc:.2f}%")

    # --- 5. Metrics Extraction ---
    log.info("Calculating final metrics for output...")
    
    freqs_check = np.minimum(get_log_freqs(0.1, fs/2, 100), fs/2 - 1e-6); T_check = 1/freqs_check
    TargetPSA_check = log_interp(T_check, To_target, targetPSA_T)
    
    def calc_norm_sq(arr):
        if arr is None or np.all(np.isnan(arr)): return np.zeros_like(t)
        sq_int = integrate.cumulative_trapezoid(arr**2, t, initial=0)
        return sq_int / sq_int[-1] if sq_int[-1] != 0 else np.zeros_like(t)

    def _calc_kinematics(a1, a2):
        v1 = integrate.cumulative_trapezoid(a1, t, initial=0); d1 = integrate.cumulative_trapezoid(v1, t, initial=0)
        v2 = integrate.cumulative_trapezoid(a2, t, initial=0); d2 = integrate.cumulative_trapezoid(v2, t, initial=0)
        
        nai1 = calc_norm_sq(a1); ncsv1 = calc_norm_sq(v1); ncsd1 = calc_norm_sq(d1)
        nai2 = calc_norm_sq(a2); ncsv2 = calc_norm_sq(v2); ncsd2 = calc_norm_sq(d2)
        return v1, v2, d1, d2, nai1, nai2, ncsv1, ncsv2, ncsd1, ncsd2

    def _calc_spectra(a1, a2):
        psa180, _, _ = compute_rotated_spectra(T_check, a1, a2, zi, dt, theta)
        psa = np.percentile(psa180, nn, axis=0)
        # Extract individual components (0-deg and 90-deg from the 180 matrix)
        psa_comp1 = psa180[0, :]
        psa_comp2 = psa180[90, :]
        return psa, psa180, psa_comp1, psa_comp2

    s1_scaled = s1 * sf; s2_scaled = s2 * sf
    
    psa_s, p180_s, psa_s1, psa_s2 = _calc_spectra(s1_scaled, s2_scaled)
    psa_sc, p180_sc, psa_sc1, psa_sc2 = _calc_spectra(sc1, sc2)
    
    v1_s, v2_s, d1_s, d2_s, nai1_s, nai2_s, ncsv1_s, ncsv2_s, ncsd1_s, ncsd2_s = _calc_kinematics(s1_scaled, s2_scaled)
    v1_sc, v2_sc, d1_sc, d2_sc, nai1_sc, nai2_sc, ncsv1_sc, ncsv2_sc, ncsd1_sc, ncsd2_sc = _calc_kinematics(sc1, sc2)

    return {
        't': t, 'dt': dt, 'freqs': freqs_check, 'periods': T_check, 'mode': 'psa', 'nn': nn, 'scale_factor': sf,
        
        's1_scaled': s1_scaled, 'sc1_unc': sc1_unc, 'sc1': sc1,
        'vel1_s': v1_s, 'vel1_sc': v1_sc, 'disp1_s': d1_s, 'disp1_sc': d1_sc,
        'ai1_s': nai1_s, 'ai1_sc': nai1_sc, 'csv1_s': ncsv1_s, 'csv1_sc': ncsv1_sc, 'csd1_s': ncsd1_s, 'csd1_sc': ncsd1_sc,
        
        's2_scaled': s2_scaled, 'sc2_unc': sc2_unc, 'sc2': sc2,
        'vel2_s': v2_s, 'vel2_sc': v2_sc, 'disp2_s': d2_s, 'disp2_sc': d2_sc,
        'ai2_s': nai2_s, 'ai2_sc': nai2_sc, 'csv2_s': ncsv2_s, 'csv2_sc': ncsv2_sc, 'csd2_s': ncsd2_s, 'csd2_sc': ncsd2_sc,

        'psa_s': psa_s, 'psa_sc': psa_sc,
        'psa_s1': psa_s1, 'psa_s2': psa_s2,
        'psa_sc1': psa_sc1, 'psa_sc2': psa_sc2,
        'target_psa': TargetPSA_check,
        'psa_180_seed': p180_s, 'psa_180_sc': p180_sc,
        
        'rmsefin': rmsefin, 'meanefin': meanefin
    }

def generate_single_component_compatible_record(
    s: np.ndarray,
    fs: float,
    T_PSA: np.ndarray,
    targetPSA: np.ndarray,
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    T1PSA: float = 0.02,
    T2PSA: float = 5.0,
    zi: float = 0.05,
    baseline_method: str = 'sixth_order',
    PSA_poly_order: int = 4,
    BL_target_disp: float = 0.0,
    knlocs_ai_pct: List[float] = [1.0, 5.0, 40.0, 75.0, 95.0, 99.0],
    NS: int = 300,
    nit: int = 30) -> Dict[str, Any]:
    
    """
    Generates a single horizontal ground motion component compatible with 
    a target response spectrum (PSA), defined by Period.

    Parameters
    ----------
    s : numpy.ndarray
        Seed record (acceleration time series in g's).
    fs : float
        Sampling frequency of the seed record (Hz).
    T_PSA : numpy.ndarray
        Array of periods (s) at which the target PSA is defined.
    targetPSA : numpy.ndarray
        Target pseudo-acceleration response spectrum.
    targetPSAlimits : Tuple[float, float], optional
        Tolerance limits (min, max) for PSA matching. Default is (0.9, 1.3).
    T1PSA, T2PSA : float, optional
        Period range (s) for PSA matching. Default is 0.02 to 5.0.
    zi : float, optional
        Damping ratio for the response spectra. Default is 0.05.
    baseline_method : str, optional
        Method for baseline correction: 'none', 'classic', 'piecewise', 'sixth_order'. 
        Default is 'sixth_order'.
    PSA_poly_order : int, optional
        Polynomial order for 'classic' baseline correction. Default is 4.
    BL_target_disp : float, optional
        Target final displacement for 'piecewise' baseline correction. Default is 0.0.
    knlocs_ai_pct : List[float], optional
        Knot locations (in Arias Intensity percentiles) for 'piecewise' correction.
    NS : int, optional
        Number of scales for the CWT. Default is 300.
    nit : int, optional
        Number of iterations for the initial PSA matching. Default is 30.

    Returns
    -------
    results : Dict[str, Any]
        Dictionary containing matched time histories, uncorrected histories, 
        and spectral metrics.
        
        General Keys:
        - 'mode' (str): The adjustment mode executed (always 'psa' here).
        - 't', 'dt' (np.ndarray, float): Time vector (s) and time step.
        - 'freqs', 'periods' (np.ndarray): Evaluation frequency and period vectors.
        - 'scale_factor' (float): Rigorous initial scale factor applied to the seed.
        - 'target_psa' (np.ndarray): Target PSA interpolated to the 'periods' vector.
        
        Time Histories & Kinematics:
        - 's_scaled': Scaled seed record.
        - 'sc_unc': Uncorrected matched record.
        - 'sc': Baseline-corrected matched record.
        - 'vel_s', 'vel_sc': Velocity time histories (scaled vs matched).
        - 'disp_s', 'disp_sc': Displacement time histories (scaled vs matched).
        - 'ai_s', 'ai_sc': Normalized Arias Intensity histories.
        - 'csv_s', 'csv_sc': Normalized Cumulative Squared Velocity histories.
        - 'csd_s', 'csd_sc': Normalized Cumulative Squared Displacement histories.
        
        Spectra:
        - 'psa_s', 'psa_sc': Computed PSA (scaled vs matched).
        
        Error Metrics:
        - 'rmsefin' (float): Final Root Mean Square Error (%) prior to baseline correction.
        - 'meanefin' (float): Final Mean Error/Misfit (%) prior to baseline correction.
    """
    
    pi = np.pi; dt = 1/fs; n = len(s)
    
    if n % 2 != 0: 
        s = np.append(s, 0); n += 1
    t = np.linspace(0, (n-1)*dt, n)
    
    # --- 1. Target Interpolation Prep (Period Based) ---
    idx_T = np.argsort(T_PSA)
    To_target = T_PSA[idx_T]; targetPSA_T = targetPSA[idx_T]
    
    FF1 = min(4/(n*dt), 0.1); FF2 = 1/(2*dt)
    T1PSA, T2PSA, FF1 = _CheckPeriodRange(T1PSA, T2PSA, To_target, FF1, FF2)
    
    log.info(f"Final matching period range set to: [{T1PSA:.3f}s, {T2PSA:.3f}s]")

    omega = pi; zeta = 0.05
    freqs = np.geomspace(FF2, FF1, NS); T = 1/freqs; scales = omega / (2*pi*freqs)
    
    # Standard linear interpolation for codebase design spectrum
    ds = log_interp(T, To_target, targetPSA_T)
    TlocsPSA = np.where((T >= T1PSA) & (T <= T2PSA))[0]
    
    if len(TlocsPSA) == 0:
        raise ValueError("No target spectrum points found within the specified matching range.")

    # --- 2. Wavelet Decomposition ---
    log.info(f"Performing CWT decomposition (Single Component)...")
    C = _cwtzm(s, fs, scales, omega, zeta); D, sr = _getdetails(t, s, C, scales, omega, zeta)

    # --- 3. Phase 1: PSA Matching ---
    PSAs_orig, _, _ = compute_spectrum(T, s, zi, dt)
    sf = np.sum(ds[TlocsPSA]) / np.sum(PSAs_orig[TlocsPSA])
    
    log.info(f"Initial scaling factor: {sf:.4f}")
    
    sr = sf * sr; D = sf * D
    PSAsr_scaled, _, _ = compute_spectrum(T, sr, zi, dt)

    hPSAbc = np.zeros((NS, nit+1)); ns = np.zeros((n, nit+1)); DN = np.zeros((NS, n, nit+1))
    
    hPSAbc[:, 0] = PSAsr_scaled
    ns[:, 0] = sr; DN[:, :, 0] = D
    factorPSA = np.ones((NS, 1))
    
    meane = np.zeros(nit + 1)
    rmse = np.zeros(nit + 1)
    
    # Compute Initial Errors (Iteration 0)
    diff_0 = np.abs(hPSAbc[TlocsPSA, 0] - ds[TlocsPSA]) / ds[TlocsPSA]
    meane[0] = np.mean(diff_0) * 100
    rmse[0] = np.linalg.norm(diff_0) / np.sqrt(len(TlocsPSA)) * 100
    
    log.info(f"Starting single-component PSA matching ({nit} iterations)...")
    log.info(f"Iteration  0: RMSE={rmse[0]:.2f}%, Misfit={meane[0]:.2f}%")

    for qq in range(1, nit+1):
        ratio_update = ds[TlocsPSA] / hPSAbc[TlocsPSA, qq-1]
        factorPSA[TlocsPSA, 0] = ratio_update
        
        DN[:, :, qq] = factorPSA * DN[:, :, qq-1]
        ns[:, qq] = np.trapezoid(DN[:, :, qq].T, scales)
        
        hPSAbc[:, qq], _, _ = compute_spectrum(T, ns[:, qq], zi, dt)
        
        # LOGGING: Iteration tracking
        diff_qq = np.abs(hPSAbc[TlocsPSA, qq] - ds[TlocsPSA]) / ds[TlocsPSA]
        meane[qq] = np.mean(diff_qq) * 100
        rmse[qq] = np.linalg.norm(diff_qq) / np.sqrt(len(TlocsPSA)) * 100
        log.info(f"Iteration {qq:2d}: RMSE={rmse[qq]:.2f}%, Misfit={meane[qq]:.2f}%")
        
    brloc = np.argmin(rmse)
    sc = ns[:, brloc]
    meanefin = meane[brloc]
    rmsefin = rmse[brloc]
    
    log.info(f"Best match at it {brloc}, RMSE: {rmsefin:.2f}%")
    log.info(f"Final RMSE (pre-BC): {rmsefin:.2f}%")
    log.info(f"Final Misfit (pre-BC): {meanefin:.2f}%")

    # --- 4. Baseline Correction ---
    sc_unc = np.copy(sc)
    
    b_meth = baseline_method.lower().strip()
    if b_meth == 'classic':
        log.info("Performing classic polynomial baseline correction...")
        sc, _, _ = baselinecorrect(sc, t, porder=PSA_poly_order)
    elif b_meth == 'piecewise':
        log.info("Performing piecewise spline baseline correction...")
        sc = piecewise_baseline_detrending(sc, t, knlocs_ai_pct=knlocs_ai_pct, target_disp=BL_target_disp)
    elif b_meth == 'sixth_order':
        log.info("Performing USGS 6th-order polynomial displacement baseline correction...")
        sc = baseline_sixth_order(sc, t)
    elif b_meth == 'none':
        log.info("Skipping baseline correction as requested.")
    else:
        log.warning(f"Unknown baseline_method '{baseline_method}'. Skipping baseline correction.")

    if b_meth != 'none':
        PSA_final_bc, _, _ = compute_spectrum(T, sc, zi, dt)
        diff_bc = np.abs(PSA_final_bc[TlocsPSA] - ds[TlocsPSA]) / ds[TlocsPSA]
        meanefin_bc = np.mean(diff_bc) * 100
        rmsefin_bc = np.linalg.norm(diff_bc) / np.sqrt(len(TlocsPSA)) * 100
        log.info(f"After Baseline Correction: RMSE={rmsefin_bc:.2f}%, Misfit={meanefin_bc:.2f}%")

    # --- 5. Metrics Extraction ---
    log.info("Calculating final metrics for output...")
    
    # 100-point logarithmic evaluation grid
    freqs_check = np.minimum(get_log_freqs(0.1, fs/2, 100), fs/2 - 1e-6); T_check = 1/freqs_check
    TargetPSA_check = log_interp(T_check, To_target, targetPSA_T)
    
    def calc_norm_sq(arr):
        if arr is None or np.all(np.isnan(arr)): return np.zeros_like(t)
        sq_int = integrate.cumulative_trapezoid(arr**2, t, initial=0)
        return sq_int / sq_int[-1] if sq_int[-1] != 0 else np.zeros_like(t)

    def _calc_kinematics(a):
        v = integrate.cumulative_trapezoid(a, t, initial=0); d = integrate.cumulative_trapezoid(v, t, initial=0)
        nai = calc_norm_sq(a); ncsv = calc_norm_sq(v); ncsd = calc_norm_sq(d)
        return v, d, nai, ncsv, ncsd

    def _calc_spectra(a):
        psa, _, _ = compute_spectrum(T_check, a, zi, dt)
        return psa

    s_scaled = s * sf
    psa_s = _calc_spectra(s_scaled)
    psa_sc = _calc_spectra(sc)
    
    v_s, d_s, nai_s, ncsv_s, ncsd_s = _calc_kinematics(s_scaled)
    v_sc, d_sc, nai_sc, ncsv_sc, ncsd_sc = _calc_kinematics(sc)

    return {
        't': t, 'dt': dt, 'freqs': freqs_check, 'periods': T_check, 'mode': 'psa', 'scale_factor': sf,
        
        's_scaled': s_scaled, 'sc_unc': sc_unc, 'sc': sc,
        'vel_s': v_s, 'vel_sc': v_sc, 'disp_s': d_s, 'disp_sc': d_sc,
        'ai_s': nai_s, 'ai_sc': nai_sc, 'csv_s': ncsv_s, 'csv_sc': ncsv_sc, 'csd_s': ncsd_s, 'csd_sc': ncsd_sc,

        'psa_s': psa_s, 'psa_sc': psa_sc,
        'target_psa': TargetPSA_check,
        
        'rmsefin': rmsefin, 'meanefin': meanefin
    }

@jit(nopython=True, cache=True)
def compute_spectrum_pw(
        T: np.ndarray, 
        s: np.ndarray, 
        zeta: float, 
        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """Calculates response spectra using the exact solution for piecewise
    linear excitation (time-domain), strictly assuming underdamping.

    Internal helper function. Preferred for low damping ratios.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s). Must contain positive values.
    s : np.ndarray
        Input ground acceleration time series (g). Assumed valid.
    zeta : float
        Damping ratio. Must be >= 0 and < 1 for this function.
    dt : float
        Time step of the acceleration series (s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Pseudo-spectral acceleration (g).
        - PSV (np.ndarray): Pseudo-spectral velocity (units like g*s).
        - SD (np.ndarray): Relative spectral displacement (units like g*s^2).
          Returns arrays of NaNs if zeta < 0 or zeta >= 1.

    Notes
    -----
    - Solves the relative motion EOM: u'' + 2ζωn u' + ωn^2 u = -ag(t).
    - Implements the exact solution for underdamped systems (0 <= zeta < 1)
      assuming linear variation of -ag(t) between time steps, using a
      state-space formulation U(t+dt) = A*U(t) + B*P(t).
    - **Strictly requires 0 <= zeta < 1.** Returns NaNs and logs an error
      if zeta is outside this range.
    - Handles T=0 explicitly in the final calculation block.
    """
    pi = np.pi
    nper = len(T)
    n = len(s)
    # Initialize output arrays
    SD = np.zeros(nper) 
    PSA = np.zeros(nper)
    PSV = np.zeros(nper)

    # --- Validate damping ratio - Return NaNs immediately if invalid ---
    if not 0 <= zeta < 1:
        SD[:] = np.nan
        PSA[:] = np.nan
        PSV[:] = np.nan
        return PSA, PSV, SD
    # --- Damping is valid (0 <= zeta < 1) ---
    
    # Input for relative displacement equation uses negative ground acceleration
    s_input = -s
    
    # Define tolerance for T=0 check
    small_tolerance = 1e-12
    
    # Handle T=0 case directly before loop
    mask_T0 = (T <= small_tolerance)
    if np.any(mask_T0):
        #log.debug("Assigning T=0 values (SD=0, PSV=0, PSA=PGA).")
        pga = np.max(np.abs(s))
        SD[mask_T0] = 0.0
        PSV[mask_T0] = 0.0
        PSA[mask_T0] = pga
        
    # Loop through strictly positive periods only
    # Define valid_indices based on T > small_tolerance
    valid_indices = np.where(T > small_tolerance)[0]

    for k in valid_indices:
        period = T[k]
        wn = 2 * pi / period # Natural frequency (rad/s)
        wn_sq = wn**2
        wn_cb = wn_sq * wn # Used in B matrix coeffs

        # State vector: u_state = [displacement, velocity]^T
        u_state = np.zeros((n, 2)).T # Stores [disp, vel] history

        # --- Coefficients for state-space matrices A and B (Underdamped Case ONLY) ---
        
        # Calculate damped frequency and related terms 
        sqrt_term = np.sqrt(1.0 - zeta**2)
        wd = wn * sqrt_term
        wd_inv = 1.0 / wd
        zeta_term = zeta / sqrt_term

        e_zwt = np.exp(-zeta * wn * dt)
        cos_wdt = np.cos(wd * dt)
        sin_wdt = np.sin(wd * dt)

        # Matrix A elements
        _a11 = e_zwt * (cos_wdt + zeta_term * sin_wdt)
        _a12 = e_zwt * wd_inv * sin_wdt
        _a21 = -wn * (1.0/sqrt_term) * e_zwt * sin_wdt
        _a22 = e_zwt * (cos_wdt - zeta_term * sin_wdt)

        # Matrix B elements
        _b11 = e_zwt * (((2 * zeta**2 - 1) / (wn_sq * dt) + zeta / wn) * wd_inv * sin_wdt +
                       (2 * zeta / (wn_cb * dt) + 1 / wn_sq) * cos_wdt) - 2 * zeta / (wn_cb * dt)
        _b12 = -e_zwt * (((2 * zeta**2 - 1) / (wn_sq * dt)) * wd_inv * sin_wdt +
                        (2 * zeta / (wn_cb * dt)) * cos_wdt) - (1 / wn_sq) + 2 * zeta / (wn_cb * dt)
        _b21 = -((_a11 - 1) / (wn_sq * dt)) - _a12
        _b22 = -_b21 - _a12

        # Assemble final matrices A and B
        A = np.array([[_a11, _a12], [_a21, _a22]])
        B = np.array([[_b11, _b12], [_b21, _b22]])

        # --- Time stepping using state-space solution ---
        s_input_step = np.empty(2, dtype=np.float64)
        for q in range(n - 1):
            # U_{q+1} = A * U_q + B * P_q
            s_input_step[0] = s_input[q]
            s_input_step[1] = s_input[q+1]
            u_state[:, q + 1] = A @ u_state[:, q] + B @ s_input_step

        # Find maximum absolute displacement
        SD[k] = np.max(np.abs(u_state[0, :]))

    # --- Calculate Pseudo Spectra from SD ---
    mask_Tvalid = (T > small_tolerance)
    omega_n = 2 * pi / T[mask_Tvalid] 
    PSV[mask_Tvalid] = omega_n * SD[mask_Tvalid]
    PSA[mask_Tvalid] = omega_n**2 * SD[mask_Tvalid]
    return PSA, PSV, SD

def compute_spectrum_fd(
        T: np.ndarray, 
        s: np.ndarray, 
        z: float, 
        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Optimized Response spectra via Frequency Domain (SD, PSA, PSV only).

    Internal helper function.

    Methodology:
    - Solves the SDOF equation of motion in the frequency domain using the 
      Fast Fourier Transform (FFT).
    - Trailing Zeros: The input acceleration time series is zero-padded (typically 
      to the next power of 2) prior to the FFT to prevent circular convolution 
      wrap-around errors and maximize algorithmic efficiency.
    - Transfer Function: The exact analytical SDOF transfer function is applied 
      directly at the discrete FFT frequencies. 

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s : np.ndarray
        Input acceleration time series (g).
    z : float
        Damping ratio.
    dt : float
        Time step (s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Pseudo-spectral acceleration (g).
        - PSV (np.ndarray): Pseudo-spectral velocity (units like g*s).
        - SD (np.ndarray): Relative spectral displacement (units like g*s^2).
    """
    pi = np.pi
    npo = len(s)
    nT = len(T)
    SD = np.zeros(nT)

    n_pad_min = int(10 * np.max(T) / dt if nT > 0 and np.max(T) > 0 else 0)
    n_fft = int(2**np.ceil(np.log2(npo + n_pad_min)))
    s_padded = np.pad(s, (0, n_fft - npo))

    freqs = np.fft.rfftfreq(n_fft, dt)
    ww = 2 * pi * freqs
    ffts = np.fft.rfft(s_padded)

    m = 1.0
    
    # Loop through strictly positive periods only
    # Define valid_indices based on T > small_tolerance
    small_tolerance = 1e-12
    valid_indices = np.where(T > small_tolerance)[0]
    
    for kk in valid_indices:
        wn = 2 * pi / T[kk]
        k_stiff = m * wn**2
        c_damp = 2 * z * m * wn

        denominator = (-m * ww**2 + k_stiff + 1j * c_damp * ww)
        denominator[np.abs(denominator) < 1e-15] = 1e-15
        H_disp = (-m) / denominator # Transfer function U(w)/Ag(w)

        fft_disp = H_disp * ffts
        d = np.fft.irfft(fft_disp, n_fft)
        SD[kk] = np.max(np.abs(d[:npo]))

    # Calculate Pseudo Spectra from SD, handle T=0
    with np.errstate(divide='ignore', invalid='ignore'):
        PSV = (2 * pi / T) * SD
        PSA = (2 * pi / T)**2 * SD
        PSV[T <= small_tolerance] = 0.0
        PSA[T <= small_tolerance] = np.max(np.abs(s)) 
        SD[T <= small_tolerance] = 0.0
    return PSA, PSV, SD

def compute_rotated_spectra_fd(
        T: np.ndarray, 
        s1: np.ndarray, 
        s2: np.ndarray, 
        zeta: float, 
        dt: float, 
        theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Calculates rotated response spectra via Frequency Domain.

    Internal helper function.

    Methodology:
    - Solves the SDOF equation of motion in the frequency domain for two 
      orthogonal components using the Fast Fourier Transform (FFT).
    - Trailing Zeros: The input acceleration time series are zero-padded 
      (calculated based on the maximum period, then up to the next power of 2) 
      prior to the FFT to prevent circular convolution wrap-around errors.
    - Transfer Function: The exact analytical SDOF transfer function is applied 
      directly at the discrete FFT frequencies for each component. 
    - Rotation: The inverted time-domain displacements are projected onto 
      the specified rotation angles to compute the maximum rotated responses.
    - True Rotated PGA: For T=0, it explicitly computes the peak ground 
      acceleration directly from the rotated time histories.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s1 : np.ndarray
        Acceleration time series for component 1 (g).
    s2 : np.ndarray
        Acceleration time series for component 2 (g).
    zeta : float
        Damping ratio.
    dt : float
        Time step (s).
    theta : np.ndarray
        Vector of angles (degrees) for rotation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Rotated PSA (num_angles x num_periods) in g.
        - PSV (np.ndarray): Rotated PSV.
        - SD (np.ndarray): Rotated SD.
    """
    
    pi = np.pi
    theta_rad = np.deg2rad(theta) # Convert angles to radians
    ntheta = len(theta)
    npo1 = len(s1); npo2 = len(s2); npo = min(npo1, npo2) # Use minimum length
    nT = len(T)
    SD = np.zeros((ntheta, nT))
    
    # --- Validate damping ratio - Return NaNs immediately if invalid ---
    if zeta<0:
        log.error(f"Invalid damping ratio zeta={zeta:.4f}. _RSFDtheta requires zeta > 0. Returning NaNs.") # <-- Corrected log message
        nan_shape = (ntheta, nT)
        # Return tuple of NaN arrays with the expected output shape
        return (np.full(nan_shape, np.nan),
                np.full(nan_shape, np.nan),
                np.full(nan_shape, np.nan))
    # --- Damping is valid (0 <= zeta < 1) ---

    # Padding
    n_pad_min = int(10 * np.max(T) / dt if nT > 0 and np.max(T) > 0 else 0)
    n_fft = int(2**np.ceil(np.log2(npo + n_pad_min)))
    s1_pad = np.pad(s1[:npo], (0, n_fft - npo))
    s2_pad = np.pad(s2[:npo], (0, n_fft - npo))

    # FFTs
    freqs = np.fft.rfftfreq(n_fft, dt)
    ww = 2 * pi * freqs
    ffts1 = np.fft.rfft(s1_pad)
    ffts2 = np.fft.rfft(s2_pad)

    m = 1.0
    
    # Loop through strictly positive periods only
    # Define valid_indices based on T > small_tolerance
    small_tolerance = 1e-12
    valid_indices = np.where(T > small_tolerance)[0]

    for kk in valid_indices:
        
        wn = 2 * pi / T[kk]
        k_stiff = m * wn**2
        c_damp = 2 * zeta * m * wn

        denominator = (-m * ww**2 + k_stiff + 1j * c_damp * ww)
        denominator[np.abs(denominator) < 1e-15] = 1e-15
        H_disp = (-m) / denominator # Transfer function U(w)/Ag(w)

        # Response in frequency domain
        fft_disp1 = H_disp * ffts1
        fft_disp2 = H_disp * ffts2

        # Response in time domain
        d1 = np.fft.irfft(fft_disp1, n_fft)[:npo]
        d2 = np.fft.irfft(fft_disp2, n_fft)[:npo]

        # Rotate using broadcasting and find max displacement
        cos_th = np.cos(theta_rad)[:, np.newaxis]
        sin_th = np.sin(theta_rad)[:, np.newaxis]
        drot = d1 * cos_th + d2 * sin_th # (ntheta, npo)
        SD[:, kk] = np.max(np.abs(drot), axis=1) # Max over time axis

    # --- Calculate Pseudo Spectra from SD, handle T=0 ---
    with np.errstate(divide='ignore', invalid='ignore'):
        # Calculate omega_n, will be inf where T is near zero
        omega_n = 2 * pi / T
        PSV = omega_n * SD
        PSA = omega_n**2 * SD

    # Explicitly set T=0 values using a mask
    mask_T0 = (T <= small_tolerance)

    if np.any(mask_T0):
        log.debug("Assigning T=0 values for rotated spectra (calculating true rotated PGA)...")

        # --- Calculate True Rotated PGA ---
        # Ensure we use the original truncated s1, s2 (length npo)
        s1_trunc = s1[:npo]
        s2_trunc = s2[:npo]

        # Use broadcasting to create rotated time series for all angles at once
        # cos_th has shape (ntheta, 1), s1_trunc has shape (npo,) -> result (ntheta, npo)
        cos_th = np.cos(theta_rad)[:, np.newaxis]
        sin_th = np.sin(theta_rad)[:, np.newaxis]
        s_rotated_histories = s1_trunc * cos_th + s2_trunc * sin_th # Shape (ntheta, npo)

        # Find the peak absolute value for each rotated history (each row)
        rotated_pga = np.max(np.abs(s_rotated_histories), axis=1) # Shape (ntheta,)
        # --- End True Rotated PGA Calculation ---

        # Apply corrections using the mask (works element-wise for each angle)
        # PSA shape is (ntheta, nT), mask_T0 is (nT,)
        # rotated_pga shape is (ntheta,)
        # Assign rotated_pga to the columns where mask_T0 is True
        # Need to transpose rotated_pga to align for broadcasting or loop
        PSA[:, mask_T0] = rotated_pga[:, np.newaxis] # Broadcast (ntheta, 1) to columns

        # PSV and SD are zero for all angles where T=0
        PSV[:, mask_T0] = 0.0
        SD[:, mask_T0] = 0.0

    return PSA, PSV, SD

@jit(nopython=True, cache=True)
def compute_rotated_spectra_pw(
        T: np.ndarray, 
        s1: np.ndarray, 
        s2: np.ndarray, 
        zeta: float, 
        dt: float, 
        theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Calculates rotated response spectra via Piecewise (time-domain) integration.

    Internal helper function. Preferred for low damping ratios or when exact 
    time-domain precision is required.

    Methodology:
    - Solves the relative motion equation of motion ($u'' + 2\zeta\omega_n u' + \omega_n^2 u = -a_g(t)$) 
      in the time domain.
    - Piecewise Linear: Uses the exact analytical solution for underdamped 
      systems ($0 \le \zeta < 1$) by assuming the input acceleration varies 
      linearly between consecutive time steps.
    - Rotation: Computes the spectral displacement time histories independently 
      for both orthogonal components, rotates these concurrent histories to the 
      specified angles, and finds the peak absolute displacement per angle.
    - True Rotated PGA: For periods at or near zero ($T=0$), it bypasses 
      displacement division and explicitly computes the peak ground acceleration 
      directly from the rotated input acceleration time histories.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s1 : np.ndarray
        Acceleration time series for component 1 (g).
    s2 : np.ndarray
        Acceleration time series for component 2 (g).
    zeta : float
        Damping ratio. Strictly requires $0 \le \zeta < 1$.
    dt : float
        Time step (s).
    theta : np.ndarray
        Vector of angles (degrees) for rotation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Rotated Pseudo-spectral acceleration (num_angles x num_periods) in g.
        - PSV (np.ndarray): Rotated Pseudo-spectral velocity (units like g*s).
        - SD (np.ndarray): Rotated relative spectral displacement (units like g*s^2).
          Returns arrays of NaNs if zeta < 0 or zeta >= 1.
    """
    
    pi = np.pi
    theta_rad = np.deg2rad(theta)
    ntheta = len(theta)
    nT = len(T)
    SD = np.zeros((ntheta, nT))
    n1 = len(s1); n2 = len(s2); n = min(n1, n2)

    s1 = s1[:n]; s2 = s2[:n]
    s_input1 = -s1
    s_input2 = -s2
    
    # --- Validate damping ratio - Return NaNs immediately if invalid ---
    if not 0 <= zeta < 1:
        nan_shape = (ntheta, nT)
        # Return tuple of NaN arrays with the expected output shape
        return (np.full(nan_shape, np.nan),
                np.full(nan_shape, np.nan),
                np.full(nan_shape, np.nan))
    # --- Damping is valid (0 <= zeta < 1) --- 

    # Define tolerance for T=0 check
    small_tolerance = 1e-12
    
    # Loop through strictly positive periods only
    # Define valid_indices based on T > small_tolerance
    valid_indices = np.where(T > small_tolerance)[0]

    for k in valid_indices:

        wn = 2 * pi / T[k]
        wd = wn * np.sqrt(1 - zeta**2)
        if wd < 1e-9: continue # Skip if damped frequency is too low

        u1 = np.zeros((n, 2)).T # [disp, vel] history for s1
        u2 = np.zeros((n, 2)).T # [disp, vel] history for s2

        # Coefficients (assuming _RSPW's coefficients are correct)
        e_zwt = np.exp(-zeta * wn * dt)
        cos_wdt = np.cos(wd * dt)
        sin_wdt = np.sin(wd * dt)
        wd_inv = 1.0 / wd
        zisq = 1 / np.sqrt(1 - zeta**2) # Used in original coeffs

        a11 = e_zwt * (cos_wdt + zeta * zisq * sin_wdt)
        a12 = e_zwt * wd_inv * sin_wdt
        a21 = -wn * zisq * e_zwt * sin_wdt
        a22 = e_zwt * (cos_wdt - zeta * zisq * sin_wdt) 
        
        A = np.array([[a11, a12], [a21, a22]])

        _b11 = e_zwt * (((2 * zeta**2 - 1) / (wn**2 * dt) + zeta / wn) * wd_inv * sin_wdt +
                       (2 * zeta / (wn**3 * dt) + 1 / wn**2) * cos_wdt) - 2 * zeta / (wn**3 * dt)
        _b12 = -e_zwt * (((2 * zeta**2 - 1) / (wn**2 * dt)) * wd_inv * sin_wdt +
                        (2 * zeta / (wn**3 * dt)) * cos_wdt) - (1 / wn**2) + 2 * zeta / (wn**3 * dt)
        _b21 = -((a11 - 1) / (wn**2 * dt)) - a12
        _b22 = -_b21 - a12
        B = np.array([[_b11, _b12], [_b21, _b22]])
        
        s_input_step1 = np.empty(2, dtype=np.float64) 
        s_input_step2 = np.empty(2, dtype=np.float64) 

        # Time stepping
        for q in range(n - 1):
            s_input_step1[0] = s_input1[q]     
            s_input_step1[1] = s_input1[q + 1] 
            
            s_input_step2[0] = s_input2[q]     
            s_input_step2[1] = s_input2[q + 1] 
            
            u1[:, q + 1] = A @ u1[:, q] + B @ s_input_step1
            u2[:, q + 1] = A @ u2[:, q] + B @ s_input_step2 
            
        d1 = u1[0, :]
        d2 = u2[0, :]

        # Rotate using broadcasting and find max
        cos_th = np.cos(theta_rad).reshape(-1, 1) # Ensure cos_th is (ntheta, 1)
        sin_th = np.sin(theta_rad).reshape(-1, 1) # Ensure sin_th is (ntheta, 1)
        drot = d1 * cos_th + d2 * sin_th # Result is (ntheta, n)
        #SD[:, k] = np.max(np.abs(drot), axis=1)
        for angle_idx in range(ntheta):
        # Find the max absolute value in the time history for this angle
            max_abs_disp_for_angle = 0.0
            for time_idx in range(n): # n is the number of time points
                abs_disp = np.abs(drot[angle_idx, time_idx])
                if abs_disp > max_abs_disp_for_angle:
                    max_abs_disp_for_angle = abs_disp
            SD[angle_idx, k] = max_abs_disp_for_angle # Assign to the correct slot

    # Calculate omega_n, will be inf where T is near zero
    omega_n = 2 * pi / T
    PSV = omega_n * SD
    PSA = omega_n**2 * SD

    # Explicitly set T=0 values using a mask
    mask_T0 = (T <= small_tolerance)

    if np.any(mask_T0):
    
            # --- Calculate True Rotated PGA ---
            cos_th = np.cos(theta_rad).reshape(-1, 1) # Shape (ntheta, 1)
            sin_th = np.sin(theta_rad).reshape(-1, 1) # Shape (ntheta, 1)
            s_rotated_histories = s1 * cos_th + s2 * sin_th # Shape (ntheta, n)
    
            # Find the peak absolute value for each rotated history (each row) using a loop
            rotated_pga = np.zeros(ntheta) # Initialize array for results

            for angle_idx in range(ntheta):
                max_abs_pga_for_angle = 0.0
                for time_idx in range(n): # n is the number of time points
                    abs_pga = np.abs(s_rotated_histories[angle_idx, time_idx])
                    if abs_pga > max_abs_pga_for_angle:
                        max_abs_pga_for_angle = abs_pga
                rotated_pga[angle_idx] = max_abs_pga_for_angle
    
            # Apply corrections using the mask
            PSA[:, mask_T0] = rotated_pga[:, np.newaxis]
            PSV[:, mask_T0] = 0.0
            SD[:, mask_T0] = 0.0
        
    return PSA, PSV, SD

def rotdnn(
    s1: np.ndarray,
    s2: np.ndarray,
    dt: float,
    zi: float,
    T: np.ndarray,
    nn: Union[int, float, List[float], Tuple[float, ...]] = 50
) -> Tuple[Union[np.ndarray, Dict[float, np.ndarray]], np.ndarray]:
    
    """
    Computes rotated and orientation-independent (RotDnn) Pseudo-Spectral 
    Acceleration (PSA) from two orthogonal horizontal components.

    This is a primary analysis function for evaluating the directionality of 
    ground motions. It supports both single percentiles (legacy) and multiple 
    percentiles simultaneously (modern).

    Methodology:
    - Rotates the input acceleration histories from 0 to 179 degrees in 
      1-degree increments using standard geometric projection 
      (s_rot = s1*cos(θ) + s2*sin(θ)).
    - Computes the response spectrum for every single rotated time history 
      at the specified periods using the frequency-domain (FFT) solver.
    - Evaluates the requested percentile(s) (e.g., 50th for median, 100th 
      for maximum) across all 180 computed spectral amplitudes independently 
      at each period bin to construct the final RotDnn envelope(s).

    Parameters
    ----------
    s1 : np.ndarray
        Acceleration series in the first orthogonal horizontal direction (g).
    s2 : np.ndarray
        Acceleration series in the second orthogonal horizontal direction (g).
    dt : float
        Time step (s).
    zi : float
        Damping ratio for spectra calculation (e.g., 0.05).
    T : np.ndarray
        Periods at which to calculate the spectra (s).
    nn : int, float, list, or tuple, optional
        Percentile(s) for the RotDnn calculation. Can be a single number 
        (e.g., 50) or a collection (e.g., [50, 100]). Default is 50.

    Returns
    -------
    Tuple[Union[np.ndarray, Dict[float, np.ndarray]], np.ndarray]
        - PSArotnn: If `nn` is a single number, returns a single np.ndarray (g). 
          If `nn` is a list/tuple, returns a dictionary of np.ndarrays keyed by percentile.
        - PSA180 (np.ndarray): Matrix (180 x num_periods) containing
          the PSA spectrum at all angles from 0 to 179 degrees (g).
    """
    # Ensure equal length for processing
    n = min(len(s1), len(s2))
    s1 = s1[:n]
    s2 = s2[:n]

    theta = np.arange(0, 180, 1)

    # Use the optimized Frequency Domain solver
    # Ignore warnings for T=0 calculations as they are handled internally
    with np.errstate(divide='ignore', invalid='ignore'):
         PSA180, _, _ = compute_rotated_spectra_fd(T, s1, s2, zi, dt, theta)
    
    # --- Smart Percentile Extraction (Hybrid Approach) ---
    if isinstance(nn, (int, float)):
        # Legacy behavior: single array return
        PSArotnn = np.percentile(PSA180, nn, axis=0)
        return PSArotnn, PSA180
    else:
        # Modern behavior: dictionary return
        PSArotnn_dict = {}
        for p in nn:
            PSArotnn_dict[p] = np.percentile(PSA180, p, axis=0)
        return PSArotnn_dict, PSA180

def compute_spectrum(
        T: np.ndarray, 
        s: np.ndarray, 
        z: float, 
        dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Dispatcher: selects the optimal response spectrum algorithm based on the damping ratio.

    Internal helper function. 

    Methodology:
    - Acts as a smart router to balance computational speed and numerical stability.
    - For damping ratios z >= 0.02 (2%), it routes the calculation to the 
      Frequency Domain solver (`compute_spectrum_fd`), which is significantly faster 
      and perfectly stable for standard to high damping.
    - For extremely low damping ratios z < 0.02, it routes to the Piecewise 
      time-domain solver (`compute_spectrum_pw`). This avoids the high-frequency 
      noise and numerical inaccuracies that FFT-based solvers can introduce in 
      highly resonant (underdamped) systems.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods to calculate spectrum (s).
    s : np.ndarray
        Input acceleration time series (g).
    z : float
        Damping ratio (e.g., 0.05 for 5%).
    dt : float
        Time step of the acceleration series (s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Pseudo-spectral acceleration (g).
        - PSV (np.ndarray): Pseudo-spectral velocity (units like g*s).
        - SD (np.ndarray): Relative spectral displacement (units like g*s^2).
    """
    if z >= 0.02:
        log.debug("Using Frequency Domain (FD) method for spectrum calculation (z>=3%).")
        return compute_spectrum_fd(T, s, z, dt) 
    else:
        log.debug("Using Piecewise (PW) method for spectrum calculation (z<3%).")
        return compute_spectrum_pw(T, s, z, dt)

def compute_rotated_spectra(
        T: np.ndarray, 
        s1: np.ndarray, 
        s2: np.ndarray, 
        z: float, 
        dt: float, 
        theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Dispatcher: selects the optimal rotated response spectrum algorithm 
    based on the damping ratio.

    Internal helper function. 

    Methodology:
    - Acts as a smart router to balance computational speed and numerical stability 
      when computing rotated spectra across multiple angles.
    - For damping ratios z >= 0.02 (2%), it routes the calculation to the 
      Frequency Domain solver (`compute_rotated_spectra_fd`), utilizing 
      efficient FFTs to evaluate all periods simultaneously.
    - For extremely low damping ratios z < 0.02, it routes to the Piecewise 
      time-domain solver (`compute_rotated_spectra_pw`). This avoids spectral 
      leakage and accurately captures highly resonant, narrow-banded responses 
      that the discrete frequency bins of the FFT might miss.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s1 : np.ndarray
        Acceleration time series for component 1 (g).
    s2 : np.ndarray
        Acceleration time series for component 2 (g).
    z : float
        Damping ratio (e.g., 0.05 for 5%).
    dt : float
        Time step (s).
    theta : np.ndarray
        Vector of angles (degrees) for rotation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Rotated Pseudo-spectral acceleration (num_angles x num_periods) in g.
        - PSV (np.ndarray): Rotated Pseudo-spectral velocity (units like g*s).
        - SD (np.ndarray): Rotated Relative spectral displacement (units like g*s^2).
    """
    
    if z >= 0.02:
        log.debug("Using FD method for rotated spectra (z>=3%).")
        return compute_rotated_spectra_fd(T, s1, s2, z, dt, theta)
    else:
        log.debug("Using PW method for rotated spectra (z<3%).")
        return compute_rotated_spectra_pw(T, s1, s2, z, dt, theta)    

def calculate_earthquake_psd(
    accel_series: np.ndarray,
    sample_rate: float,
    tukey_alpha: float = 0.1,
    duration_percent: Optional[Tuple[float, float]] = (5, 75),
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    nfft_base: Literal['total', 'strong'] = 'total',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear',
    smoothing_method: Literal['none', 'konno_ohmachi', 'variable_window'] = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    downsample_freqs: Optional[np.ndarray] = None,
    prefer_pykooh: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, float, float, Optional[np.ndarray], Optional[np.ndarray]]:
    
    """
    Calculates the Power Spectral Density (PSD) of an earthquake acceleration record.

    Methodology:
    - Arias Intensity & Duration: Computes the cumulative Arias Intensity of the 
      record to determine the strong motion duration (typically the time between 
      5% and 75% of the total energy buildup).
    - Windowing: Extracts the strong motion segment (or uses the entire signal) 
      and applies a Tukey window to taper the ends smoothly to zero, minimizing 
      spectral leakage.
    - FFT & Normalization: Zero-pads the signal (typically to the next power of 2) 
      to maximize Fast Fourier Transform (FFT) efficiency. The one-sided PSD is 
      computed by squaring the magnitude of the FFT and normalizing it by the 
      strong motion duration.
    - Smoothing & Interpolation: Optionally applies Konno-Ohmachi or variable 
      boxcar smoothing to the highly variable raw PSD. The smoothed spectrum can 
      then be downsampled/interpolated to a specific target frequency grid.

    Parameters
    ----------
    accel_series : np.ndarray
        The 1D acceleration time-series (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    tukey_alpha : float, optional
        Shape parameter for the Tukey window, representing the fraction of the 
        window inside the cosine tapered region. Default is 0.1.
    duration_percent : Tuple[float, float] or None, optional
        The start and end percentages of Arias Intensity defining the strong 
        motion duration. If None, the entire record is used. Default is (5, 75).
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT padding. 
        Default is 'nextpow2'.
    nfft_base : {'total', 'strong'}, optional
        Determines whether the padding length is based on the 'total' record 
        length or strictly the 'strong' motion segment. Default is 'total'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method used to detrend the extracted signal before windowing. 
        Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method used to smooth the computed PSD. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the chosen smoothing algorithm (e.g., the 'b' bandwidth 
        parameter for Konno-Ohmachi). Default is 20.0.
    downsample_freqs : np.ndarray or None, optional
        Target frequency vector (Hz) to interpolate the smoothed PSD. 
        If None, the native raw FFT frequency grid is returned.
    prefer_pykooh : bool, optional
        If True, utilizes the optimized `pykooh` library for Konno-Ohmachi 
        smoothing (if installed). Default is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, float, Optional[np.ndarray], Optional[np.ndarray]]
        - freqs (np.ndarray): The raw frequency vector (Hz) from the FFT.
        - psd (np.ndarray): The raw, unsmoothed Power Spectral Density (g^2*s).
        - sd (float): The calculated strong motion duration (s).
        - ai (float): The total Arias Intensity of the record (m/s).
        - freqs_smooth (np.ndarray or None): The target frequency grid for the smoothed PSD.
        - psd_smooth (np.ndarray or None): The smoothed Power Spectral Density.
    """
    
    n_total = len(accel_series)
    dt = 1.0 / sample_rate
    t = np.linspace(0, (n_total - 1) * dt, n_total)

    # 1. Determine Duration and Times
    if duration_percent is None:
        duration_percent = (0, 100)
    
    sd, _, ai, t1, t2 = SignificantDuration(
        accel_series, t, duration_percent[0], duration_percent[1]
    )
    
    # 2. Slice Strong Motion
    locs = np.where((t >= t1 - dt/2) & (t <= t2 + dt/2))
    strong_motion = accel_series[locs]
    
    # 3. Window and Detrend
    if len(strong_motion) > 0:
        win = signal.windows.tukey(len(strong_motion), tukey_alpha)
        if detrend_method:
            strong_motion = signal.detrend(strong_motion, type=detrend_method)
        strong_motion = strong_motion * win

    # 4. FFT Setup
    base_n = n_total if nfft_base == 'total' else len(strong_motion)
    if nfft_method == 'nextpow2':
        n_fft = int(2**np.ceil(np.log2(base_n)))
    elif nfft_method == 'same':
        n_fft = base_n
    else:
        n_fft = int(nfft_method)
        
    # 5. Calculate PSD
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    fft_vals = np.fft.rfft(strong_motion, n_fft)
    mags = dt * np.abs(fft_vals)
    psd = 2 * mags**2 / (2 * np.pi * sd)
    
    # 6. Smooth PSD
    psd_smooth = np.copy(psd)
    freqs_smooth = freqs # Default to FFT freqs
    
    if smoothing_method != 'none':
        if downsample_freqs is not None:
             freqs_smooth = downsample_freqs
        
        if smoothing_method == 'variable_window':
            psd_smooth = _smooth_boxcar_variable(freqs_smooth, freqs, psd, smoothing_coeff)
        elif smoothing_method == 'konno_ohmachi':
            if PYKOOH_AVAILABLE and prefer_pykooh:
                psd_smooth = pykooh.smooth(freqs_smooth, freqs, psd, smoothing_coeff)
            else:
                psd_smooth = _konno_ohmachi_1998_downsample(freqs_smooth, freqs, psd, smoothing_coeff)
            
    return freqs, psd, sd, ai, freqs_smooth, psd_smooth

def calculate_earthquake_fas(
    accel_series: np.ndarray,
    sample_rate: float,
    tukey_alpha: float = 0.1,
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear',
    smoothing_method: Literal['none', 'konno_ohmachi', 'variable_window'] = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    downsample_freqs: Optional[np.ndarray] = None,
    prefer_pykooh: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    
    """
    Calculates the Fourier Amplitude Spectrum (FAS) of an earthquake acceleration record.

    Methodology:
    - Preprocessing: Detrends the acceleration signal (e.g., linear) to remove 
      baseline drift and applies a Tukey window to taper the ends smoothly to 
      zero, reducing spectral leakage.
    - FFT & Normalization: Zero-pads the signal (typically to the next power of 2) 
      to optimize the Fast Fourier Transform (FFT) efficiency. Computes the 
      one-sided Fourier Amplitude Spectrum (FAS), multiplying by the time step 
      (dt) to maintain consistent amplitude density units (e.g., g*s).
    - Smoothing & Interpolation: Optionally applies Konno-Ohmachi or variable 
      boxcar smoothing to the highly variable raw FAS. The smoothed spectrum 
      can then be downsampled/interpolated directly onto a specific target 
      frequency grid.

    Parameters
    ----------
    accel_series : np.ndarray
        The 1D acceleration time-series data (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    tukey_alpha : float, optional
        Tukey window shape parameter, representing the fraction of the window 
        inside the cosine tapered region. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points (nFFT) for the FFT padding. 
        Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal prior to windowing. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method used to smooth the calculated FAS. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the chosen smoothing algorithm (e.g., 'b' parameter 
        for Konno-Ohmachi). Default is 20.0.
    downsample_freqs : np.ndarray or None, optional
        Target frequency vector (Hz) to interpolate the smoothed FAS. 
        If None, the native raw FFT frequency grid is returned.
    prefer_pykooh : bool, optional
        If True, utilizes the optimized `pykooh` library for Konno-Ohmachi 
        smoothing (if installed). Default is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]
        - freqs (np.ndarray): The raw frequency vector (Hz) from the FFT.
        - fas (np.ndarray): The raw, unsmoothed one-sided Fourier Amplitude Spectrum (g*s).
        - freqs_smooth (np.ndarray or None): The target frequency grid for the smoothed FAS.
        - fas_smooth (np.ndarray or None): The smoothed Fourier Amplitude Spectrum.

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).
    """
    # --- 1. Input Validation and Initial Setup ---
    if not isinstance(accel_series, np.ndarray) or accel_series.ndim != 1:
        raise ValueError("`accel_series` must be a 1D NumPy array.")

    n_total = len(accel_series)
    dt = 1.0 / sample_rate

    # --- 2. Pre-processing (Windowing and Detrending) ---
    window = signal.windows.tukey(n_total, tukey_alpha)
    processed_series = accel_series * window

    if detrend_method in ('linear', 'constant'):
        processed_series = signal.detrend(processed_series, type=detrend_method)
    elif detrend_method is not None:
        raise ValueError("`detrend_method` must be 'linear', 'constant', or None.")

    # --- 3. Determine nFFT ---
    if nfft_method == 'nextpow2':
        n_fft = int(2**np.ceil(np.log2(n_total)))
    elif nfft_method == 'same':
        n_fft = n_total
    elif isinstance(nfft_method, int):
        n_fft = nfft_method
    else:
        raise ValueError("`nfft_method` must be 'nextpow2', 'same', or an integer.")

    # --- 4. Calculate FAS ---
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    fft_result = np.fft.rfft(processed_series, n_fft)
    fas = dt * np.abs(fft_result)

    # --- VALIDATION BLOCK ---
    if downsample_freqs is not None:
        min_raw_freq = freqs.min()
        max_raw_freq = freqs.max()
        min_out_freq = downsample_freqs.min()
        max_out_freq = downsample_freqs.max()

        if min_out_freq < min_raw_freq or max_out_freq > max_raw_freq:
            raise ValueError(
                f"User-specified `downsample_freqs` are outside the valid frequency range. "
                f"Please provide frequencies between {min_raw_freq:.2f} Hz and {max_raw_freq:.2f} Hz."
            )
    # --- END VALIDATION ---

    # --- 5. Smoothing and Downsampling ---
    freqs_smooth, fas_smooth = None, None
    if smoothing_method != 'none':
        output_freqs = downsample_freqs if downsample_freqs is not None else freqs
        freqs_smooth = output_freqs

        if smoothing_method == 'konno_ohmachi':
            if prefer_pykooh and PYKOOH_AVAILABLE:
                fas_smooth = pykooh.smooth(output_freqs, freqs, fas, smoothing_coeff)
            else:
                if prefer_pykooh and not PYKOOH_AVAILABLE:
                    warnings.warn("`pykooh` not found. Using in-house Konno-Ohmachi smoothing.", UserWarning)
                fas_smooth = _konno_ohmachi_1998_downsample(output_freqs, freqs, fas, smoothing_coeff)
        elif smoothing_method == 'variable_window':
            fas_smooth = _smooth_boxcar_variable(output_freqs, freqs, fas, percentage=smoothing_coeff)
        else:
            raise ValueError(
                "`smoothing_method` must be 'none', 'konno_ohmachi', or 'variable_window'."
            )

    return freqs, fas, freqs_smooth, fas_smooth

def calculate_rotated_fas(
    accel_series_1: np.ndarray,
    accel_series_2: np.ndarray,
    sample_rate: float,
    rotation_angles: int = 180,
    tukey_alpha: float = 0.1,
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Calculates the Fourier Amplitude Spectrum (FAS) for rotated horizontal components.

    Methodology:
    - Preprocessing: Ensures both orthogonal horizontal time-series are the same length, 
      detrends them to remove baseline drift, and applies a Tukey window to taper 
      the ends smoothly to zero, reducing spectral leakage.
    - FFT: Zero-pads both signals (e.g., to the next power of 2) to optimize Fast 
      Fourier Transform (FFT) efficiency, and computes their complex spectra via 
      a real-input FFT (rFFT).
    - Frequency-Domain Rotation: Rather than rotating the histories in the time domain 
      step-by-step, it exploits the linearity of the Fourier Transform. It projects the 
      base complex spectra onto the specified rotation angles using standard geometric 
      projection ($F_{rot} = F_1 \cos\theta + F_2 \sin\theta$) using fast array broadcasting.
    - Normalization: Computes the magnitude of the rotated complex spectra to get the 
      one-sided FAS, scaling by the time step (dt) to maintain amplitude density units (e.g., g*s).

    Parameters
    ----------
    accel_series_1 : numpy.ndarray
        The 1D acceleration time-series for the first horizontal component (g).
    accel_series_2 : numpy.ndarray
        The 1D acceleration time-series for the second horizontal component (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    rotation_angles : int, optional
        The number of rotation angles to compute, evenly spaced from 0 to 180
        degrees (exclusive of 180). Default is 180.
    tukey_alpha : float, optional
        Tukey window shape parameter, between 0 (rectangular) and 1 (Hann).
        Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal. Default is 'linear'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - freqs (numpy.ndarray): The 1D array of frequencies for the FAS (Hz).
        - fas_rotated (numpy.ndarray): The 2D array of rotated FAS values (g*s). 
          Shape is (rotation_angles, num_freqs).
    """
    
    # --- 1. Input Validation and Initial Setup ---
    if accel_series_1.shape != accel_series_2.shape or accel_series_1.ndim != 1:
        raise ValueError("Input series must be 1D NumPy arrays of the same length.")

    n_total = len(accel_series_1)
    dt = 1.0 / sample_rate

    # --- 2. Pre-processing (applied identically to both components) ---
    processed_series_1 = np.copy(accel_series_1)
    processed_series_2 = np.copy(accel_series_2)
    
    if detrend_method in ('linear', 'constant'):
        processed_series_1 = signal.detrend(processed_series_1, type=detrend_method)
        processed_series_2 = signal.detrend(processed_series_2, type=detrend_method)
    elif detrend_method is not None:
        raise ValueError("`detrend_method` must be 'linear', 'constant', or None.")

    window = signal.windows.tukey(n_total, tukey_alpha)
    processed_series_1 *= window
    processed_series_2 *= window

    # --- 3. Determine nFFT ---
    if nfft_method == 'nextpow2':
        n_fft = int(2**np.ceil(np.log2(n_total)))
    elif nfft_method == 'same':
        n_fft = n_total
    elif isinstance(nfft_method, int):
        n_fft = nfft_method
    else:
        raise ValueError("`nfft_method` must be 'nextpow2', 'same', or an integer.")

    # --- 4. Calculate Rotated FAS in Frequency Domain ---
    freqs = np.fft.rfftfreq(n_fft, d=dt)
    angles_rad = np.linspace(0, np.pi, rotation_angles, endpoint=False)
    
    fft_1 = np.fft.rfft(processed_series_1, n_fft)
    fft_2 = np.fft.rfft(processed_series_2, n_fft)
    
    # Use meshgrid for efficient broadcasting
    fft_1_grid, angles_grid = np.meshgrid(fft_1, angles_rad, sparse=True)
    fft_2_grid, _ = np.meshgrid(fft_2, angles_rad, sparse=True)
    
    # Apply rotation formula and get FAS
    fas_rotated = dt * np.abs(fft_1_grid * np.cos(angles_grid) + fft_2_grid * np.sin(angles_grid))
    
    return freqs, fas_rotated

def calculate_rotated_psd(
    accel_series_1: np.ndarray,
    accel_series_2: np.ndarray,
    sample_rate: float,
    rotation_angles: int = 180,
    duration_percent: Optional[Tuple[float, float]] = (5, 75),
    tukey_alpha: float = 0.1,
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear'
    ) -> Tuple[np.ndarray, np.ndarray]:
    
    """
    Calculates the Power Spectral Density (PSD) for rotated horizontal components.

    Methodology:
    - Because the strong motion duration is angle-dependent, this function must 
      perform the rotation sequentially in the time domain rather than the 
      frequency domain.
    - Time-Domain Rotation: Iterates through the specified angles, geometrically 
      projecting the two orthogonal acceleration histories to create a new rotated 
      time series for each angle ($s_{rot} = s_1 \cos\theta + s_2 \sin\theta$).
    - Dynamic Duration: For every single rotated history, it recalculates the 
      Arias Intensity buildup to isolate the specific strong motion segment 
      (e.g., 5% to 75% thresholds) for that exact angle. 
    - Preprocessing: Detrends the isolated strong motion segment to remove baseline 
      drift and applies a Tukey window to taper the ends smoothly.
    - FFT & Normalization: Zero-pads the segment to a consistent nFFT length and 
      computes the real-input FFT (rFFT). The one-sided PSD is computed by squaring 
      the magnitude and dividing by the angle-specific strong motion duration.
      
    Warning: Because it must perform duration extraction and an FFT for every single 
    angle in a loop, this function is computationally intensive.

    Parameters
    ----------
    accel_series_1 : np.ndarray
        The 1D acceleration time-series for the first horizontal component (g).
    accel_series_2 : np.ndarray
        The 1D acceleration time-series for the second horizontal component (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    rotation_angles : int, optional
        The number of rotation angles to compute, evenly spaced from 0 to 180
        degrees (exclusive of 180). Default is 180.
    duration_percent : Tuple[float, float] or None, optional
        The start and end percentages of Arias Intensity defining the strong 
        motion duration. If None, the entire record is used. Default is (5, 75).
    tukey_alpha : float, optional
        Tukey window shape parameter, representing the fraction of the window 
        inside the cosine tapered region. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT padding. The same 
        nFFT length is used across all angles to ensure consistent frequency bins. 
        Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal prior to windowing. Default is 'linear'.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - freqs (np.ndarray): The 1D array of frequencies for the PSD (Hz).
        - psd_rotated (np.ndarray): The 2D array of rotated PSD values (g^2*s). 
          Shape is (rotation_angles, num_freqs).
    """
    # --- 1. Input Validation and Initial Setup ---
    if accel_series_1.shape != accel_series_2.shape or accel_series_1.ndim != 1:
        raise ValueError("Input series must be 1D NumPy arrays of the same length.")

    n_total = len(accel_series_1)
    dt = 1.0 / sample_rate
    time_vector = np.linspace(0, (n_total - 1) * dt, n_total)
    angles_rad = np.linspace(0, np.pi, rotation_angles, endpoint=False)

    # --- 2. Determine a single, consistent nFFT for all rotations ---
    # The base number of points is the full signal length, as the strong motion
    # part can vary for each angle.
    if nfft_method == 'nextpow2':
        n_fft = int(2**np.ceil(np.log2(n_total)))
    elif nfft_method == 'same':
        n_fft = n_total
    elif isinstance(nfft_method, int):
        n_fft = nfft_method
    else:
        raise ValueError("`nfft_method` must be 'nextpow2', 'same', or an integer.")

    freqs = np.fft.rfftfreq(n_fft, d=dt)
    psd_rotated = np.zeros((rotation_angles, len(freqs)))

    # --- 3. Loop through each angle to perform rotation and PSD calculation ---
    for i, angle in enumerate(angles_rad):
        # a. Rotate the time-series
        s_rot = accel_series_1 * np.cos(angle) + accel_series_2 * np.sin(angle)

        # b. Determine strong motion segment for this specific rotation
        if duration_percent:
            strong_duration, _, _, t1, t2 = SignificantDuration(
                s_rot, time_vector, ival=duration_percent[0], fval=duration_percent[1]
            )
            strong_motion_mask = (time_vector >= t1) & (time_vector <= t2)
            s_strong = s_rot[strong_motion_mask]
        else: # Use the whole signal
            strong_duration = (n_total - 1) * dt
            s_strong = s_rot

        # c. Pre-process the strong motion segment
        if detrend_method in ('linear', 'constant'):
            s_processed = signal.detrend(s_strong, type=detrend_method)
        else:
            s_processed = s_strong
            
        window = signal.windows.tukey(len(s_processed), tukey_alpha)
        s_processed *= window

        # d. Calculate FFT and PSD for this rotation
        fft_rot = np.fft.rfft(s_processed, n_fft)
        psd_rotated[i, :] = (2 * (dt**2) * np.abs(fft_rot)**2) / (2 * np.pi * strong_duration)

    return freqs, psd_rotated   

def calculate_fas_rotDnn(
    accel_series_1: np.ndarray,
    accel_series_2: np.ndarray,
    sample_rate: float,
    percentiles: Union[List[float], Tuple[float, ...]] = (50, 100),
    tukey_alpha: float = 0.1,
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear',
    smoothing_method: Literal['none', 'konno_ohmachi', 'variable_window'] = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    downsample_freqs: Optional[np.ndarray] = None,
    smooth_last: bool = True,
    prefer_pykooh: bool = True
) -> Tuple[np.ndarray, Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    """
    Computes orientation-independent (RotDnn) Fourier Amplitude Spectra (FAS) 
    for multiple percentiles simultaneously.

    Methodology:
    - Frequency-Domain Rotation: Calls `calculate_rotated_fas` to compute the complex 
      spectra of both orthogonal components and project them across 180 angles 
      simultaneously in the frequency domain using fast array broadcasting.
    - Percentile Extraction: Evaluates the requested percentile(s) (e.g., 50th, 100th) 
      across all 180 rotated spectra independently at each raw FFT frequency bin.
    - Raw Interpolation: The raw RotDnn envelopes are explicitly log-log interpolated 
      onto the target `downsample_freqs` grid to provide a baseline for comparison.
    - Smoothing Workflows (`smooth_last`):
      - If True (Default): Extracts the raw RotDnn percentile envelope first, and then 
        applies the smoothing matrix directly to that envelope. The smoothing algorithm 
        (e.g., Konno-Ohmachi) inherently maps the data onto the `downsample_freqs` 
        grid via matrix multiplication without requiring standard interpolation.
      - If False (Smooth First): Applies the smoothing matrix to all 180 individual 
        rotated spectra first (simultaneously mapping them to the new frequency grid), 
        and then evaluates the percentiles of those smoothed spectra.

    Parameters
    ----------
    accel_series_1 : np.ndarray
        The 1D acceleration time-series for the first horizontal component (g).
    accel_series_2 : np.ndarray
        The 1D acceleration time-series for the second horizontal component (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    percentiles : List[float] or Tuple[float, ...], optional
        A container of percentiles to compute from the rotated spectra 
        (e.g., (50, 100) for RotD50 and RotD100). Default is (50, 100).
    tukey_alpha : float, optional
        Tukey window shape parameter, representing the fraction of the window 
        inside the cosine tapered region. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT padding. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal prior to windowing. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated FAS. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the chosen smoothing algorithm (e.g., 'b' parameter 
        for Konno-Ohmachi). Default is 20.0.
    downsample_freqs : np.ndarray or None, optional
        Target frequency vector (Hz) to interpolate the final spectra. 
        If None, the native raw FFT frequency grid is used.
    smooth_last : bool, optional
        Controls the smoothing order (see Methodology). Default is True.
    prefer_pykooh : bool, optional
        If True, utilizes the optimized `pykooh` library for Konno-Ohmachi 
        smoothing (if installed). Default is True.

    Returns
    -------
    Tuple[np.ndarray, Dict[float, np.ndarray], Dict[float, np.ndarray]]
        - output_freqs (np.ndarray): The frequency vector (Hz) for the output spectra.
        - fas_rotdnn_smooth (Dict[float, np.ndarray]): Dictionary where keys are the 
          requested percentiles and values are the smoothed RotDnn FAS arrays (g*s).
        - fas_rotdnn_raw_interp (Dict[float, np.ndarray]): Dictionary where keys are the 
          requested percentiles and values are the raw (unsmoothed) RotDnn FAS arrays 
          interpolated to the `output_freqs` grid (g*s).

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).
    """
    # --- 1. Calculate the full suite of raw rotated FAS ---
    freqs_raw, fas_rot_raw = calculate_rotated_fas(
        accel_series_1,
        accel_series_2,
        sample_rate,
        rotation_angles=180,
        tukey_alpha=tukey_alpha,
        nfft_method=nfft_method,
        detrend_method=detrend_method
    )

    output_freqs = downsample_freqs if downsample_freqs is not None else freqs_raw

    # --- VALIDATION BLOCK ---
    if downsample_freqs is not None:
        min_raw_freq = freqs_raw.min()
        max_raw_freq = freqs_raw.max()  # Nyquist frequency
        min_out_freq = output_freqs.min()
        max_out_freq = output_freqs.max()

        if min_out_freq < min_raw_freq or max_out_freq > max_raw_freq:
            raise ValueError(
                f"User-specified `downsample_freqs` are outside the valid frequency range. "
                f"Please provide frequencies between {min_raw_freq:.2f} Hz and {max_raw_freq:.2f} Hz."
            )

    percentiles_list = list(percentiles)

    # --- 2. Calculate the raw RotD spectra and interpolate ---
    fas_rotdnn_raw = np.percentile(fas_rot_raw, percentiles_list, axis=0)
    
    fas_rotdnn_raw_interp = {}
    for i, p in enumerate(percentiles_list):
        percentile_data = fas_rotdnn_raw[i, :]
        interp_data = np.full_like(output_freqs, np.nan, dtype=np.float64)

        epsilon = 1e-20
        valid_source_freq_mask = (freqs_raw > 0)
        valid_target_freq_mask = (output_freqs > 0)

        safe_percentile_data = np.copy(percentile_data)
        safe_percentile_data[(safe_percentile_data <= 0) & valid_source_freq_mask] = epsilon

        interp_data[valid_target_freq_mask] = log_interp(
            output_freqs[valid_target_freq_mask],
            freqs_raw[valid_source_freq_mask],
            safe_percentile_data[valid_source_freq_mask]
        )
        
        if output_freqs[0] == 0:
            interp_data[0] = np.interp(0.0, freqs_raw, percentile_data)
        
        fas_rotdnn_raw_interp[p] = interp_data

    # --- 3. Apply the chosen smoothing workflow ---
    fas_rotdnn_smooth = {}

    # If no smoothing is requested, the "smoothed" result is the raw interpolated one.
    if smoothing_method == 'none':
        fas_rotdnn_smooth = fas_rotdnn_raw_interp.copy()
        return output_freqs, fas_rotdnn_smooth, fas_rotdnn_raw_interp

    if smooth_last:
        # Workflow: Percentile -> Smooth
        for i, p in enumerate(percentiles_list):
            raw_percentile_spectrum = fas_rotdnn_raw[i, :]
            if smoothing_method == 'konno_ohmachi':
                if prefer_pykooh and PYKOOH_AVAILABLE:
                    fas_rotdnn_smooth[p] = pykooh.smooth(output_freqs, freqs_raw, raw_percentile_spectrum, smoothing_coeff)
                else:
                    if prefer_pykooh and not PYKOOH_AVAILABLE:
                        warnings.warn("`pykooh` not found. Using in-house Konno-Ohmachi smoothing.", UserWarning)
                    fas_rotdnn_smooth[p] = _konno_ohmachi_1998_downsample(output_freqs, freqs_raw, raw_percentile_spectrum, b=smoothing_coeff)
            elif smoothing_method == 'variable_window':
                fas_rotdnn_smooth[p] = _smooth_boxcar_variable(output_freqs, freqs_raw, raw_percentile_spectrum, percentage=smoothing_coeff)
            else:
                raise ValueError("`smoothing_method` is invalid.")
             
    else: # "Smooth First" workflow
        # Workflow: Smooth -> Percentile
        if smoothing_method == 'konno_ohmachi':
            if prefer_pykooh and PYKOOH_AVAILABLE:
                smoother = pykooh.CachedSmoother(freqs_raw, output_freqs, smoothing_coeff)
                W_kooh = smoother._weights
                fas_rot_smooth = fas_rot_raw @ W_kooh
            else:
                if prefer_pykooh and not PYKOOH_AVAILABLE:
                    warnings.warn("`pykooh` not found. Using in-house Konno-Ohmachi smoothing.", UserWarning)
                W = _konno_ohmachi_1998_sparse_matrix(output_freqs, freqs_raw, b=smoothing_coeff)
                fas_rot_smooth = (W @ fas_rot_raw.T).T
        
        elif smoothing_method == 'variable_window':
            fas_rot_smooth = np.zeros((fas_rot_raw.shape[0], len(output_freqs)))
            for i in range(fas_rot_raw.shape[0]):
                fas_rot_smooth[i,:] = _smooth_boxcar_variable(output_freqs, freqs_raw, fas_rot_raw[i,:], percentage=smoothing_coeff)
        else:
             raise ValueError("`smoothing_method` is invalid.")

        smoothed_percentiles = np.percentile(fas_rot_smooth, percentiles_list, axis=0)
        for i, p in enumerate(percentiles_list):
            fas_rotdnn_smooth[p] = smoothed_percentiles[i, :]

    return output_freqs, fas_rotdnn_smooth, fas_rotdnn_raw_interp

def calculate_psd_rotDnn(
    accel_series_1: np.ndarray,
    accel_series_2: np.ndarray,
    sample_rate: float,
    percentiles: Union[List[float], Tuple[float, ...]] = (50, 100),
    duration_percent: Optional[Tuple[float, float]] = (5, 75),
    tukey_alpha: float = 0.1,
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear',
    smoothing_method: Literal['none', 'konno_ohmachi', 'variable_window'] = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    downsample_freqs: Optional[np.ndarray] = None,
    smooth_last: bool = True,
    prefer_pykooh: bool = True
    ) -> Tuple[np.ndarray, Dict[float, np.ndarray], Dict[float, np.ndarray]]:
    
    """
    Computes orientation-independent (RotDnn) Power Spectral Density (PSD) 
    spectra for multiple percentiles simultaneously.

    Methodology:
    - Time-Domain Rotation & PSD: Calls `calculate_rotated_psd` to perform 
      sequential time-domain rotations, extract angle-dependent strong motion 
      durations, and compute the 180 separate PSDs.
    - Percentile Extraction: Evaluates the requested percentile(s) (e.g., 50th, 100th) 
      across all 180 rotated spectra independently at each raw FFT frequency bin.
    - Raw Interpolation: The raw RotDnn envelopes are explicitly log-log interpolated 
      onto the target `downsample_freqs` grid to provide a baseline for comparison.
    - Smoothing Workflows (`smooth_last`):
      - If True (Default): Extracts the raw RotDnn percentile envelope first, and then 
        applies the smoothing matrix directly to that envelope. The smoothing algorithm 
        (e.g., Konno-Ohmachi) inherently maps the data onto the `downsample_freqs` 
        grid via matrix multiplication without requiring standard interpolation.
      - If False (Smooth First): Applies the smoothing matrix to all 180 individual 
        rotated spectra first (simultaneously mapping them to the new frequency grid), 
        and then evaluates the percentiles of those smoothed spectra.

    Parameters
    ----------
    accel_series_1 : np.ndarray
        The 1D acceleration time-series for the first horizontal component (g).
    accel_series_2 : np.ndarray
        The 1D acceleration time-series for the second horizontal component (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    percentiles : List[float] or Tuple[float, ...], optional
        A container of percentiles to compute from the rotated spectra 
        (e.g., (50, 100) for RotD50 and RotD100). Default is (50, 100).
    duration_percent : Tuple[float, float] or None, optional
        The start and end percentages of Arias Intensity defining the strong 
        motion duration. If None, the entire record is used. Default is (5, 75).
    tukey_alpha : float, optional
        Tukey window shape parameter, representing the fraction of the window 
        inside the cosine tapered region. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT padding. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal prior to windowing. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated PSD. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the chosen smoothing algorithm (e.g., 'b' parameter 
        for Konno-Ohmachi). Default is 20.0.
    downsample_freqs : np.ndarray or None, optional
        Target frequency vector (Hz) to interpolate the final spectra. 
        If None, the native raw FFT frequency grid is used.
    smooth_last : bool, optional
        Controls the smoothing order (see Methodology). Default is True.
    prefer_pykooh : bool, optional
        If True, utilizes the optimized `pykooh` library for Konno-Ohmachi 
        smoothing (if installed). Default is True.

    Returns
    -------
    Tuple[np.ndarray, Dict[float, np.ndarray], Dict[float, np.ndarray]]
        - output_freqs (np.ndarray): The frequency vector (Hz) for the output spectra.
        - psd_rotdnn_smooth (Dict[float, np.ndarray]): Dictionary where keys are the 
          requested percentiles and values are the smoothed RotDnn PSD arrays (g^2*s).
        - psd_rotdnn_raw_interp (Dict[float, np.ndarray]): Dictionary where keys are the 
          requested percentiles and values are the raw (unsmoothed) RotDnn PSD arrays 
          interpolated to the `output_freqs` grid (g^2*s).

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).
    """
    # --- 1. Calculate the full suite of raw rotated PSD ---
    freqs_raw, psd_rot_raw = calculate_rotated_psd(
        accel_series_1,
        accel_series_2,
        sample_rate,
        rotation_angles=180,
        duration_percent=duration_percent,
        tukey_alpha=tukey_alpha,
        nfft_method=nfft_method,
        detrend_method=detrend_method
    )

    output_freqs = downsample_freqs if downsample_freqs is not None else freqs_raw

    # --- VALIDATION BLOCK ---
    if downsample_freqs is not None:
        min_raw_freq = freqs_raw.min()
        max_raw_freq = freqs_raw.max()
        min_out_freq = output_freqs.min()
        max_out_freq = output_freqs.max()

        if min_out_freq < min_raw_freq or max_out_freq > max_raw_freq:
            raise ValueError(
                f"User-specified `downsample_freqs` are outside the valid frequency range. "
                f"Please provide frequencies between {min_raw_freq:.2f} Hz and {max_raw_freq:.2f} Hz."
            )

    percentiles_list = list(percentiles)

    # --- 2. Calculate the raw RotD spectra and interpolate ---
    # Handle NaNs from rotations with zero duration before calculating percentiles
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        psd_rotdnn_raw = np.nanpercentile(psd_rot_raw, percentiles_list, axis=0)
    
    psd_rotdnn_raw_interp = {}
    for i, p in enumerate(percentiles_list):
        percentile_data = psd_rotdnn_raw[i, :]
        interp_data = np.full_like(output_freqs, np.nan, dtype=np.float64)

        epsilon = 1e-20
        valid_source_freq_mask = (freqs_raw > 0)
        valid_target_freq_mask = (output_freqs > 0)

        safe_percentile_data = np.copy(percentile_data)
        safe_percentile_data[(safe_percentile_data <= 0) & valid_source_freq_mask] = epsilon

        interp_data[valid_target_freq_mask] = log_interp(
            output_freqs[valid_target_freq_mask],
            freqs_raw[valid_source_freq_mask],
            safe_percentile_data[valid_source_freq_mask]
        )
        
        if output_freqs[0] == 0:
            interp_data[0] = np.interp(0.0, freqs_raw, percentile_data)
        
        psd_rotdnn_raw_interp[p] = interp_data

    # --- 3. Apply the chosen smoothing workflow ---
    psd_rotdnn_smooth = {}

    # If no smoothing is requested, the "smoothed" result is the raw interpolated one.
    if smoothing_method == 'none':
        psd_rotdnn_smooth = psd_rotdnn_raw_interp.copy()
        return output_freqs, psd_rotdnn_smooth, psd_rotdnn_raw_interp

    if smooth_last:
        # Workflow: Percentile -> Smooth
        for i, p in enumerate(percentiles_list):
            raw_percentile_spectrum = psd_rotdnn_raw[i, :]
            if smoothing_method == 'konno_ohmachi':
                if prefer_pykooh and PYKOOH_AVAILABLE:
                    psd_rotdnn_smooth[p] = pykooh.smooth(output_freqs, freqs_raw, raw_percentile_spectrum, smoothing_coeff)
                else:
                    if prefer_pykooh and not PYKOOH_AVAILABLE:
                        warnings.warn("`pykooh` not found. Using in-house Konno-Ohmachi smoothing.", UserWarning)
                    psd_rotdnn_smooth[p] = _konno_ohmachi_1998_downsample(output_freqs, freqs_raw, raw_percentile_spectrum, b=smoothing_coeff)
            elif smoothing_method == 'variable_window':
                psd_rotdnn_smooth[p] = _smooth_boxcar_variable(output_freqs, freqs_raw, raw_percentile_spectrum, percentage=smoothing_coeff)
            else:
                raise ValueError("`smoothing_method` is invalid.")
             
    else: # "Smooth First" workflow
        # Workflow: Smooth -> Percentile
        if smoothing_method == 'konno_ohmachi':
            if prefer_pykooh and PYKOOH_AVAILABLE:
                smoother = pykooh.CachedSmoother(freqs_raw, output_freqs, smoothing_coeff)
                W_kooh = smoother._weights
                # Replace NaNs with 0 before matrix multiplication
                psd_rot_smooth = np.nan_to_num(psd_rot_raw) @ W_kooh
            else:
                if prefer_pykooh and not PYKOOH_AVAILABLE:
                    warnings.warn("`pykooh` not found. Using in-house Konno-Ohmachi smoothing.", UserWarning)
                W = _konno_ohmachi_1998_sparse_matrix(output_freqs, freqs_raw, b=smoothing_coeff)
                psd_rot_smooth = (W @ np.nan_to_num(psd_rot_raw).T).T
        
        elif smoothing_method == 'variable_window':
            psd_rot_smooth = np.zeros((psd_rot_raw.shape[0], len(output_freqs)))
            for i in range(psd_rot_raw.shape[0]):
                # Ignore rows that are all NaN from the start
                if not np.all(np.isnan(psd_rot_raw[i,:])):
                    psd_rot_smooth[i,:] = _smooth_boxcar_variable(output_freqs, freqs_raw, psd_rot_raw[i,:], percentage=smoothing_coeff)
        else:
             raise ValueError("`smoothing_method` is invalid.")

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            smoothed_percentiles = np.nanpercentile(psd_rot_smooth, percentiles_list, axis=0)
        for i, p in enumerate(percentiles_list):
            psd_rotdnn_smooth[p] = smoothed_percentiles[i, :]

    return output_freqs, psd_rotdnn_smooth, psd_rotdnn_raw_interp

def calculate_eas(
    accel_series_1: np.ndarray,
    accel_series_2: np.ndarray,
    sample_rate: float,
    tukey_alpha: float = 0.1,
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear',
    smoothing_method: Literal['none', 'konno_ohmachi', 'variable_window'] = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    downsample_freqs: Optional[np.ndarray] = None,
    smooth_last: bool = True,
    prefer_pykooh: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Computes the Effective Amplitude Spectrum (EAS) for two orthogonal horizontal components.

    Methodology:
    - FAS Computation: Calculates the raw Fourier Amplitude Spectrum (FAS) independently 
      for each orthogonal component using standard preprocessing (detrending, Tukey 
      windowing, zero-padded rFFT).
    - EAS Formulation: Combines the two orthogonal FAS arrays into a single orientation-independent 
      Effective Amplitude Spectrum using the quadratic mean (Root-Mean-Square): 
      $EAS = \\sqrt{0.5 \\times (FAS_1^2 + FAS_2^2)}$.
    - Raw Interpolation: Explicitly log-log interpolates the raw EAS onto the requested 
      `downsample_freqs` grid to provide a baseline for comparison.
    - Smoothing Workflows (`smooth_last`):
      - If True (Default): Evaluates the combined raw EAS first, and then applies the 
        smoothing matrix directly to that envelope. The smoothing algorithm (e.g., 
        Konno-Ohmachi) inherently maps the data onto the `downsample_freqs` grid.
      - If False (Smooth First): Smooths the individual FAS arrays of each component 
        first, and then calculates the EAS by combining the smoothed FAS arrays.

    Parameters
    ----------
    accel_series_1 : np.ndarray
        The 1D acceleration time-series for the first horizontal component (g).
    accel_series_2 : np.ndarray
        The 1D acceleration time-series for the second horizontal component (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    tukey_alpha : float, optional
        Tukey window shape parameter, representing the fraction of the window 
        inside the cosine tapered region. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT padding. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal prior to windowing. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated spectra. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the chosen smoothing algorithm (e.g., 'b' parameter 
        for Konno-Ohmachi). Default is 20.0.
    downsample_freqs : np.ndarray or None, optional
        Target frequency vector (Hz) to interpolate the final spectra. 
        If None, the native raw FFT frequency grid is used.
    smooth_last : bool, optional
        Controls the smoothing order (see Methodology). Default is True.
    prefer_pykooh : bool, optional
        If True, utilizes the optimized `pykooh` library for Konno-Ohmachi 
        smoothing (if installed). Default is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - output_freqs (np.ndarray): The frequency vector (Hz) for the output spectra.
        - eas_smooth (np.ndarray): The smoothed Effective Amplitude Spectrum (g*s).
        - eas_raw_interp (np.ndarray): The raw (unsmoothed) EAS array 
          interpolated to the `output_freqs` grid (g*s).

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).
    """
    
    # --- 1. Calculate the raw FAS for each component ---
    common_args = {
        'sample_rate': sample_rate, 'tukey_alpha': tukey_alpha,
        'nfft_method': nfft_method, 'detrend_method': detrend_method,
        'smoothing_method': 'none' # Always get raw FAS first
    }
    # These external function calls need to be defined in your FinalModule.py
    # from FinalModule import calculate_earthquake_fas, log_interp, _smooth_variable_window, konno_ohmachi_1998_downsample
    freqs_raw, fas1_raw, _, _ = calculate_earthquake_fas(accel_series_1, **common_args)
    _, fas2_raw, _, _ = calculate_earthquake_fas(accel_series_2, **common_args)

    output_freqs = downsample_freqs if downsample_freqs is not None else freqs_raw

    # --- VALIDATION BLOCK ---
    if downsample_freqs is not None:
        min_raw_freq, max_raw_freq = freqs_raw.min(), freqs_raw.max()
        min_out_freq, max_out_freq = output_freqs.min(), output_freqs.max()
        if min_out_freq < min_raw_freq or max_out_freq > max_raw_freq:
            raise ValueError(
                f"User-specified `downsample_freqs` are outside the valid frequency range. "
                f"Please provide frequencies between {min_raw_freq:.2f} Hz and {max_raw_freq:.2f} Hz."
            )

    # --- 2. Calculate the raw EAS and interpolate it ---
    eas_raw = np.sqrt(0.5 * (fas1_raw**2 + fas2_raw**2))

    # Interpolate raw result to the output frequencies for a fair comparison
    eas_raw_interp = np.full_like(output_freqs, np.nan, dtype=np.float64)
    epsilon = 1e-20
    valid_source_freq_mask = (freqs_raw > 0)
    valid_target_freq_mask = (output_freqs > 0)
    
    safe_eas_raw = np.copy(eas_raw)
    safe_eas_raw[(safe_eas_raw <= 0) & valid_source_freq_mask] = epsilon
    
    eas_raw_interp[valid_target_freq_mask] = log_interp(
        output_freqs[valid_target_freq_mask],
        freqs_raw[valid_source_freq_mask],
        safe_eas_raw[valid_source_freq_mask]
    )
    if output_freqs[0] == 0:
        eas_raw_interp[0] = np.interp(0.0, freqs_raw, eas_raw)

    # --- 3. Apply the chosen smoothing workflow ---
    
    # If no smoothing is requested, the "smoothed" result is the raw interpolated one.
    if smoothing_method == 'none':
        eas_smooth = np.copy(eas_raw_interp)
        return output_freqs, eas_smooth, eas_raw_interp

    if smooth_last:
        # Workflow: Combine -> Smooth
        if smoothing_method == 'konno_ohmachi':
            if prefer_pykooh and PYKOOH_AVAILABLE:
                eas_smooth = pykooh.smooth(output_freqs, freqs_raw, eas_raw, smoothing_coeff)
            else:
                if prefer_pykooh and not PYKOOH_AVAILABLE:
                    warnings.warn("`pykooh` not found. Using in-house Konno-Ohmachi smoothing.", UserWarning)
                eas_smooth = _konno_ohmachi_1998_downsample(output_freqs, freqs_raw, eas_raw, b=smoothing_coeff)
        elif smoothing_method == 'variable_window':
            eas_smooth = _smooth_boxcar_variable(output_freqs, freqs_raw, eas_raw, percentage=smoothing_coeff)
        else:
             raise ValueError("`smoothing_method` is invalid.")
             
    else: # "Smooth First" workflow
        # Workflow: Smooth -> Combine
        smooth_args = {
            'sample_rate': sample_rate, 'tukey_alpha': tukey_alpha,
            'nfft_method': nfft_method, 'detrend_method': detrend_method,
            'smoothing_method': smoothing_method, 'smoothing_coeff': smoothing_coeff,
            'downsample_freqs': output_freqs, 'prefer_pykooh': prefer_pykooh
        }
        _, _, _, fas1_smooth = calculate_earthquake_fas(accel_series_1, **smooth_args)
        _, _, _, fas2_smooth = calculate_earthquake_fas(accel_series_2, **smooth_args)
        
        eas_smooth = np.sqrt(0.5 * (fas1_smooth**2 + fas2_smooth**2))

    return output_freqs, eas_smooth, eas_raw_interp

def calculate_epsd(
    accel_series_1: np.ndarray,
    accel_series_2: np.ndarray,
    sample_rate: float,
    duration_percent: Optional[Tuple[float, float]] = (5, 75),
    tukey_alpha: float = 0.1,
    nfft_method: Union[Literal['nextpow2', 'same'], int] = 'nextpow2',
    nfft_base: Literal['total', 'strong'] = 'total',
    detrend_method: Optional[Literal['linear', 'constant']] = 'linear',
    smoothing_method: Literal['none', 'konno_ohmachi', 'variable_window'] = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    downsample_freqs: Optional[np.ndarray] = None,
    smooth_last: bool = True,
    prefer_pykooh: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Computes the Effective Power Spectral Density (EPSD) for two orthogonal horizontal components.

    Methodology:
    - PSD Computation: Calculates the raw Power Spectral Density (PSD) independently 
      for each orthogonal component by extracting their respective strong motion 
      durations, applying Tukey windowing, and performing a zero-padded rFFT.
    - EPSD Formulation: Combines the two orthogonal PSD arrays into a single orientation-independent 
      Effective Power Spectrum using the quadratic mean (Root-Mean-Square): 
      $EPSD = \\sqrt{0.5 \\times (PSD_1^2 + PSD_2^2)}$.
    - Raw Interpolation: Explicitly log-log interpolates the raw EPSD onto the requested 
      `downsample_freqs` grid to provide a baseline for comparison.
    - Smoothing Workflows (`smooth_last`):
      - If True (Default): Evaluates the combined raw EPSD first, and then applies the 
        smoothing matrix directly to that envelope. The smoothing algorithm (e.g., 
        Konno-Ohmachi) inherently maps the data onto the `downsample_freqs` grid.
      - If False (Smooth First): Smooths the individual PSD arrays of each component 
        first, and then calculates the EPSD by combining the smoothed PSD arrays.

    Parameters
    ----------
    accel_series_1 : np.ndarray
        The 1D acceleration time-series for the first horizontal component (g).
    accel_series_2 : np.ndarray
        The 1D acceleration time-series for the second horizontal component (g).
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    duration_percent : Tuple[float, float] or None, optional
        The start and end percentages of Arias Intensity defining the strong
        motion duration for each component. If None, the entire signal is used. Default is (5, 75).
    tukey_alpha : float, optional
        Tukey window shape parameter, representing the fraction of the window 
        inside the cosine tapered region. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT padding. Default is 'nextpow2'.
    nfft_base : {'total', 'strong'}, optional
        Determines whether the padding length is based on the 'total' record length 
        or strictly the 'strong' motion segment. Default is 'total'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal prior to windowing. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated spectra. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the chosen smoothing algorithm (e.g., 'b' parameter 
        for Konno-Ohmachi). Default is 20.0.
    downsample_freqs : np.ndarray or None, optional
        Target frequency vector (Hz) to interpolate the final spectra. 
        If None, the native raw FFT frequency grid is used.
    smooth_last : bool, optional
        Controls the smoothing order (see Methodology). Default is True.
    prefer_pykooh : bool, optional
        If True, utilizes the optimized `pykooh` library for Konno-Ohmachi 
        smoothing (if installed). Default is True.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - output_freqs (np.ndarray): The frequency vector (Hz) for the output spectra.
        - epsd_smooth (np.ndarray): The smoothed Effective Power Spectral Density (g^2*s).
        - epsd_raw_interp (np.ndarray): The raw (unsmoothed) EPSD array 
          interpolated to the `output_freqs` grid (g^2*s).

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).
    """
    # --- 1. Calculate the raw PSD for each component ---
    common_args = {
        'sample_rate': sample_rate, 'tukey_alpha': tukey_alpha,
        'duration_percent': duration_percent, 'nfft_method': nfft_method,
        'nfft_base': nfft_base, 'detrend_method': detrend_method,
        'smoothing_method': 'none' # Always get raw PSDs first
    }
    # These external function calls need to be defined in your FinalModule.py
    # from FinalModule import calculate_earthquake_psd, log_interp, smooth_boxcar_variable, konno_ohmachi_1998_downsample
    freqs_raw, psd1_raw, _, _, _, _ = calculate_earthquake_psd(accel_series_1, **common_args)
    _, psd2_raw, _, _, _, _ = calculate_earthquake_psd(accel_series_2, **common_args)

    output_freqs = downsample_freqs if downsample_freqs is not None else freqs_raw
    # --- VALIDATION BLOCK ---
    if downsample_freqs is not None:
        min_raw_freq, max_raw_freq = freqs_raw.min(), freqs_raw.max()
        min_out_freq, max_out_freq = output_freqs.min(), output_freqs.max()
        if min_out_freq < min_raw_freq or max_out_freq > max_raw_freq:
            raise ValueError(
                f"User-specified `downsample_freqs` are outside the valid frequency range. "
                f"Please provide frequencies between {min_raw_freq:.2f} Hz and {max_raw_freq:.2f} Hz."
            )

    # --- 2. Calculate the raw EPSD and interpolate it ---
    epsd_raw = np.sqrt(0.5 * (psd1_raw**2 + psd2_raw**2))

    epsd_raw_interp = np.full_like(output_freqs, np.nan, dtype=np.float64)
    epsilon = 1e-20
    valid_source_freq_mask = (freqs_raw > 0)
    valid_target_freq_mask = (output_freqs > 0)
    
    safe_epsd_raw = np.copy(epsd_raw)
    safe_epsd_raw[(safe_epsd_raw <= 0) & valid_source_freq_mask] = epsilon
    
    epsd_raw_interp[valid_target_freq_mask] = log_interp(
        output_freqs[valid_target_freq_mask],
        freqs_raw[valid_source_freq_mask],
        safe_epsd_raw[valid_source_freq_mask]
    )
    if output_freqs[0] == 0:
        epsd_raw_interp[0] = np.interp(0.0, freqs_raw, epsd_raw)

    # --- 3. Apply the chosen smoothing workflow ---

    # If no smoothing is requested, the "smoothed" result is the raw interpolated one.
    if smoothing_method == 'none':
        epsd_smooth = np.copy(epsd_raw_interp)
        return output_freqs, epsd_smooth, epsd_raw_interp

    if smooth_last:
        # Workflow: Combine -> Smooth
        if smoothing_method == 'konno_ohmachi':
            if prefer_pykooh and PYKOOH_AVAILABLE:
                epsd_smooth = pykooh.smooth(output_freqs, freqs_raw, epsd_raw, smoothing_coeff)
            else:
                if prefer_pykooh and not PYKOOH_AVAILABLE:
                    warnings.warn("`pykooh` not found. Using in-house Konno-Ohmachi smoothing.", UserWarning)
                epsd_smooth = _konno_ohmachi_1998_downsample(output_freqs, freqs_raw, epsd_raw, b=smoothing_coeff)
        elif smoothing_method == 'variable_window':
            epsd_smooth = _smooth_boxcar_variable(output_freqs, freqs_raw, epsd_raw, percentage=smoothing_coeff)
        else:
             raise ValueError("`smoothing_method` is invalid.")
             
    else: # "Smooth First" workflow
        # Workflow: Smooth -> Combine
        smooth_args = {
            'sample_rate': sample_rate, 'tukey_alpha': tukey_alpha,
            'duration_percent': duration_percent, 'nfft_method': nfft_method,
            'nfft_base': nfft_base, 'detrend_method': detrend_method,
            'smoothing_method': smoothing_method, 'smoothing_coeff': smoothing_coeff,
            'downsample_freqs': output_freqs, 'prefer_pykooh': prefer_pykooh
        }
        _, _, _, _, _, psd1_smooth = calculate_earthquake_psd(accel_series_1, **smooth_args)
        _, _, _, _, _, psd2_smooth = calculate_earthquake_psd(accel_series_2, **smooth_args)
        
        epsd_smooth = np.sqrt(0.5 * (psd1_smooth**2 + psd2_smooth**2))

    return output_freqs, epsd_smooth, epsd_raw_interp

def baseline_sixth_order(
        accel: np.ndarray, 
        t: np.ndarray) -> np.ndarray:
    """
    Applies a baseline correction to an acceleration time history using a 
    6th-order polynomial fit on the displacement.

    Methodology:
    - Integration: Performs double integration (using the cumulative trapezoidal 
      rule) on the raw input acceleration to estimate the uncorrected velocity 
      and displacement time histories.
    - Polynomial Fitting: Formulates and solves a least-squares problem to fit a 
      6th-order polynomial to the uncorrected displacement history. The 0th and 1st 
      order coefficients are explicitly forced to zero ($d_{fit} = c_2 t^2 + c_3 t^3 + 
      c_4 t^4 + c_5 t^5 + c_6 t^6$) to ensure the initial displacement and velocity 
      conditions remain physically logical (starting at zero).
    - Differentiation & Correction: Computes the second analytical derivative of this 
      fitted displacement polynomial to determine the low-frequency drift present in 
      the acceleration domain. This drift is subtracted from the original acceleration 
      to produce a baseline-corrected record that prevents artificial, unbounded 
      drifts in final displacement.

    Parameters
    ----------
    accel : np.ndarray
        The 1D input acceleration time-series (typically in g).
    t : np.ndarray
        The 1D time vector corresponding to the acceleration series (s).

    Returns
    -------
    np.ndarray
        The baseline-corrected acceleration time-series (same shape as input).
    """
    dt = t[1] - t[0]
    vel = integrate.cumulative_trapezoid(accel, dx=dt, initial=0.0)
    disp = integrate.cumulative_trapezoid(vel, dx=dt, initial=0.0)
    
    A = np.column_stack([t**2, t**3, t**4, t**5, t**6])
    c, _, _, _ = np.linalg.lstsq(A, disp, rcond=None)
    
    accel_trend = 2*c[0] + 6*c[1]*t + 12*c[2]*(t**2) + 20*c[3]*(t**3) + 30*c[4]*(t**4)
    
    return accel - accel_trend

def piecewise_baseline_detrending(
    accel: np.ndarray, 
    t: np.ndarray, 
    knlocs_ai_pct: List[float] = [1.0, 5.0, 40.0, 75.0, 95.0, 99.0],
    target_disp: float = 0.0
) -> np.ndarray:
    """
    Applies a piecewise cubic spline baseline correction and forces the final 
    displacement to a specific targeted value.

    Methodology:
    - Integration & Arias Intensity: Integrates the input acceleration to obtain 
      raw velocity and computes the normalized Arias Intensity buildup.
    - Dynamic Knot Placement: Maps the user-specified Arias Intensity percentages 
      (`knlocs_ai_pct`) to specific times. These serve as dynamic internal knots 
      that adapt to the energy release characteristics of the specific earthquake.
    - Spline Fitting: Fits a piecewise cubic spline (`LSQUnivariateSpline`) to the 
      uncorrected velocity history using these dynamic knot locations to capture 
      the low-frequency drift.
    - Detrending: Computes the analytical first derivative of the velocity spline 
      to extract the corresponding acceleration drift. This trend is subtracted 
      from the raw acceleration.
    - Displacement Forcing: Integrates the corrected acceleration to check the final 
      displacement. If it differs from `target_disp`, a specific linear acceleration 
      correction term is superimposed to drive the final displacement precisely 
      to the target without introducing a residual final velocity error.

    Parameters
    ----------
    accel : np.ndarray
        The 1D input acceleration time-series (typically in g).
    t : np.ndarray
        The 1D time vector corresponding to the acceleration series (s).
    knlocs_ai_pct : List[float], optional
        List of percentages of the total Arias Intensity used to dynamically 
        place the internal knots for the spline fit. 
        Default is [1.0, 5.0, 40.0, 75.0, 95.0, 99.0].
    target_disp : float, optional
        The strictly enforced final displacement value for the corrected 
        record. Default is 0.0.

    Returns
    -------
    np.ndarray
        The piecewise baseline-corrected acceleration time-series.
    """
    dt = t[1] - t[0]
    t_end = t[-1]
    
    # 1. Initial Integration 
    raw_vel = integrate.cumulative_trapezoid(accel, dx=dt, initial=0.0)

    # 2. Arias Intensity Calculation
    arias_intensity = integrate.cumulative_trapezoid(accel**2, dx=dt, initial=0.0)
    if arias_intensity[-1] == 0: return accel
    arias_norm = arias_intensity / arias_intensity[-1]

    # 3. Dynamic Knot Mapping
    valid_knots = [k for k in [t[np.argmax(arias_norm >= (p / 100.0))] for p in knlocs_ai_pct] if t[0] < k < t_end]
    if not valid_knots: return accel

    # 4. Spline Fitting with Boundary Pinning
    weights = np.ones_like(t)
    weights[0], weights[-1] = 1e6, 1e6
    spline_vel = LSQUnivariateSpline(t, raw_vel, t=valid_knots, k=3, w=weights)
    
    # 5. Trend Extraction
    accel_trend = spline_vel.derivative(n=1)(t)
    corr_accel = accel - accel_trend
    
    # 6. Target Displacement Closure Check
    corr_vel = integrate.cumulative_trapezoid(corr_accel, dx=dt, initial=0.0)
    corr_disp = integrate.cumulative_trapezoid(corr_vel, dx=dt, initial=0.0)
    
    d_error = corr_disp[-1] - target_disp
    
    if abs(d_error) > 1e-8:
        A = (6.0 * d_error) / (t_end**3)
        delta_accel = A * (t_end - 2.0 * t)
        corr_accel = corr_accel - delta_accel

    return corr_accel

def baselinecorrect(
    sc: np.ndarray,
    t: np.ndarray,
    porder: int = -1,
    imax: int = 80,
    tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """
    Performs robust, iterative baseline correction to force final velocity and 
    displacement to zero by calling `_basecorr`.

    Methodology:
    - Initial Detrending: Optionally fits and subtracts a low-order polynomial 
      (defined by `porder`) from the entire acceleration record before detailed correction.
    - Iterative Scaling (`_basecorr`): The core algorithm (Suarez & Montejo, 2007) 
      attempts to eliminate residual velocity and displacement by applying localized, 
      multiplicative weighting factors. It scales the amplitudes of the acceleration 
      points strictly at the beginning and end portions of the record (the "correction 
      time window", `CT`), leaving the central strong motion segment completely untouched.
    - Window Expansion: Starts with a small correction window (e.g., 1/20th of 
      the total duration). If the internal algorithm fails to converge within `imax` 
      iterations at the specified tolerance (`tol`), this function automatically 
      increases the window size and retries.
    - Failsafe Mechanism: If the required correction window grows to exceed half the 
      total duration of the record (indicating a heavily corrupted signal), the 
      algorithm aborts and safely returns the uncorrected acceleration along with its 
      raw, uncorrected velocity and displacement integrations.

    Parameters
    ----------
    sc : np.ndarray
        The 1D uncorrected acceleration time-series (g).
    t : np.ndarray
        The 1D time vector corresponding to the acceleration series (s).
    porder : int, optional
        Order of the polynomial used for initial overall detrending. 
        Default is -1 (no initial detrending).
    imax : int, optional
        Maximum number of iterations allowed for the correction algorithm 
        within each window attempt. Default is 80.
    tol : float, optional
        Tolerance criterion (expressed as a percentage of the maximum absolute 
        value of the uncorrected velocity and displacement) to determine if 
        the record has converged to zero at the end. Default is 0.01 (1%).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - ccs (np.ndarray): The baseline-corrected acceleration time-series (g).
        - cvel (np.ndarray): The corresponding corrected velocity time history (units like g*s).
        - cdespl (np.ndarray): The corresponding corrected displacement time history (units like g*s^2).

    References
    ----------
    .. [4] Suarez, L. E., & Montejo, L. A. (2007). Applications of the 
           wavelet transform in the analysis and generation of artificial 
           earthquakes. Engineering Structures, 29(8), 1730-1746.
    """
    
    CT = max(1.0, t[-1] / 20.0) # Initial time window for correction
    log.info(f'Attempting baseline correction using first/last {CT:.1f} seconds.')
    vel_orig, despl_orig, ccs, cvel, cdespl = _basecorr(t, sc, CT, porder=porder, imax=imax, tol=tol)

    kka = 1
    # Check if correction failed (indicated by NaNs)
    while np.isnan(ccs).any():
        kka += 1
        CTn = kka * CT # Increase correction window
        log.warning(f'Correction failed with CT={CT*(kka-1):.1f}s. Retrying with {CTn:.1f}s.')
        if CTn >= t[len(t)//2]: # Stop if window exceeds half the record length
            warnings.warn("Baseline correction failed repeatedly. Returning uncorrected record.")
            # Return original integrated values if correction fails completely
            vel_fail = integrate.cumulative_trapezoid(sc, x=t, initial=0)
            despl_fail = integrate.cumulative_trapezoid(vel_fail, x=t, initial=0)
            return sc, vel_fail, despl_fail

        # Try again with the larger window
        _, _, ccs, cvel, cdespl = _basecorr(t, sc, CTn, porder=porder, imax=imax, tol=tol)

    log.info("Baseline correction successful.")
    return ccs, cvel, cdespl

def SignificantDuration(
    s: np.ndarray, 
    t: np.ndarray, 
    ival: float = 5, 
    fval: float = 75
    ) -> Tuple[float, np.ndarray, float, float, float]:
    
    """
    Calculates the significant duration and Arias Intensity of a time series.

    Methodology:
    - Energy Integration: Uses the cumulative trapezoidal rule to integrate the 
      squared acceleration history over time, calculating the raw, unscaled 
      cumulative Arias Intensity buildup ($AI(t) = \\int_0^t a(\\tau)^2 d\\tau$).
    - Normalization: Normalizes the cumulative energy buildup by its maximum 
      final value so that it scales perfectly from 0.0 to 1.0 (0% to 100%).
    - Duration Extraction: Uses boolean masking to isolate the portion of the 
      time vector where the normalized energy falls exactly between the user-defined 
      initial (`ival`) and final (`fval`) percentage thresholds. 
    - Time Identification: The start time (`t1`) and end time (`t2`) are extracted 
      from this isolated window, and the significant duration is defined simply as 
      the difference ($D = t_2 - t_1$).

    Parameters
    ----------
    s : np.ndarray
        Acceleration time-history as a 1D NumPy array (typically in g).
    t : np.ndarray
        Time vector corresponding to the acceleration data (s). Must be the 
        same length as `s`.
    ival : float, optional
        Initial percentage of Arias Intensity defining the start of the 
        significant duration. Default is 5.
    fval : float, optional
        Final percentage of Arias Intensity defining the end of the 
        significant duration. Default is 75.

    Returns
    -------
    Tuple[float, np.ndarray, float, float, float]
        - sd (float): Significant duration (e.g., D_5-75), calculated as `t2 - t1` (s).
        - AIcumnorm (np.ndarray): The normalized cumulative Arias Intensity, 
          ranging from 0 to 1.
        - AI (float): The total Arias Intensity. Note: This returns the raw integral 
          of the squared acceleration without the standard seismological pre-factor 
          of $\\pi / (2g)$.
        - t1 (float): The exact time (s) when the cumulative energy crosses `ival`%.
        - t2 (float): The exact time (s) when the cumulative energy crosses `fval`%.
    Raises
    ------
    ValueError
        If `ival` or `fval` are outside the [0, 100] range, if `fval` is not 
        greater than `ival`, or if the calculated duration is not a valid 
        positive number.
    """
    # --- (1) Input Validation ---
    if not (0 <= ival <= 100 and 0 <= fval <= 100):
        raise ValueError("`ival` and `fval` must be between 0 and 100.")
    if fval <= ival:
        raise ValueError("`fval` must be greater than `ival`.")

    # Calculate the cumulative Arias Intensity
    AIcum = integrate.cumulative_trapezoid(s**2, t, initial=0)
    AI = AIcum[-1]

    # Handle cases with zero or negative total energy
    if AI <= 0:
        raise ValueError("Total Arias Intensity (AI) must be positive to calculate duration.")

    # Normalize the cumulative Arias Intensity
    AIcumnorm = AIcum / AI

    # Find the times where the normalized AI is within the desired range
    t_strong = t[(AIcumnorm >= ival / 100) & (AIcumnorm <= fval / 100)]

    if t_strong.size == 0:
        raise ValueError(
            f"No data points found between {ival}% and {fval}% of Arias Intensity. "
            "The signal may be too short or have an unusual shape."
        )

    # Get the start and end times of this interval
    t1, t2 = t_strong[0], t_strong[-1]

    # Significant duration is the difference between the end and start times
    sd = t2 - t1
    
    # --- (2) Output Validation ---
    if not np.isfinite(sd) or sd <= 0:
        raise ValueError(f"Calculated significant duration ({sd:.2f}s) is not a valid positive number.")

    return sd, AIcumnorm, AI, t1, t2

def pga_correction(
        targetPGA: float, 
        t: np.ndarray, 
        s: np.ndarray, 
        maxit: int = 1000) -> np.ndarray:
    """
    Performs a localized time-domain correction to force the Peak Ground 
    Acceleration (PGA) to exactly match a specified target.

    Methodology:
    - Pulse Isolation: Locates the single point of absolute maximum acceleration 
      (the current PGA). It then uses a peak-finding algorithm on the inverted 
      absolute signal to locate the adjacent "valleys" (local minima) immediately 
      before and after the peak. This mathematically isolates the specific 
      acceleration pulse responsible for the PGA.
    - Localized Scaling: Computes a target multiplier ratio ($\lambda = PGA_{target} / PGA_{current}$).
    - Bilinear Tapering: Constructs a localized, time-dependent scaling window. 
      This multiplier is exactly 1.0 everywhere outside the isolated pulse, ramps 
      linearly from 1.0 to $\lambda$ at the exact location of the peak, and linearly 
      back down to 1.0 at the end of the pulse. This prevents artificial discontinuities 
      or steps in the time history.
    - Iterative Adjustment: Multiplies the acceleration record by this localized 
      scaling window. Because modifying one peak might inadvertently cause an 
      adjacent peak to become the new global maximum (especially if the target 
      scales down), this process iterates until the absolute maximum of the entire 
      record matches the `targetPGA`, or until `maxit` is reached.

    Parameters
    ----------
    targetPGA : float
        The absolute targeted Peak Ground Acceleration (g).
    t : np.ndarray
        The 1D time vector corresponding to the acceleration series (s).
    s : np.ndarray
        The 1D input acceleration time-series (g).
    maxit : int, optional
        Maximum number of iterations allowed to achieve the target PGA. 
        Default is 1000.

    Returns
    -------
    np.ndarray
        The PGA-corrected acceleration time-series (g).
    """
    
    
    sc = np.copy(s)
    motionPGA = np.max(np.abs(sc))
    cont = 0
    
    # Determine direction of correction
    direction = 1 if motionPGA < targetPGA else -1
    
    while cont < maxit:
        currentPGA = np.max(np.abs(sc))
        
        # Check convergence
        if direction == 1 and currentPGA >= targetPGA: break
        if direction == -1 and currentPGA <= targetPGA: break
        
        motionPGAloc = np.argmax(np.abs(sc))
        lamb = targetPGA / currentPGA
        
        # Find zero crossings or local minima around peak to define the cycle
        # Using simple peak finding on inverted signal to find valleys
        peaks_neg, _ = find_peaks(-np.abs(sc))
        
        # Find bounding peaks
        pre_peaks = peaks_neg[peaks_neg < motionPGAloc]
        post_peaks = peaks_neg[peaks_neg > motionPGAloc]
        
        if len(pre_peaks) == 0 or len(post_peaks) == 0:
            break # Cannot isolate cycle
            
        left_p = pre_peaks[-1]
        right_p = post_peaks[0]
        
        # Bilinear scaling function
        mods = [1, 1, lamb, 1, 1]
        tmods = [t[0], t[left_p], t[motionPGAloc], t[right_p], t[-1]]
        mod = np.interp(t, tmods, mods)
        
        sc = mod * sc
        cont += 1
        
    return sc

def log_interp(
    x: np.ndarray,
    xp: np.ndarray,
    yp: np.ndarray,
    epsilon: float = 1e-20,
    out_of_bounds: str = 'nan') -> np.ndarray:
    
    """
    Performs log-linear interpolation in 1D, featuring robust handling of 
    zeros and out-of-bounds queries.

    Methodology:
    - Zero Protection: Since the natural logarithm of zero is mathematically 
      undefined (resulting in -infinity), a tiny offset (`epsilon`) is first 
      added to all input y-coordinates (`yp`) to ensure numerical stability.
    - Logarithmic Transformation: Computes the natural logarithm of the protected 
      y-values ($y_{log} = \\ln(y_p + \\epsilon)$), mapping the data into a semi-log 
      space where exponential growth/decay relationships become strictly linear.
    - Linear Interpolation: Uses standard NumPy linear interpolation (`np.interp`) 
      to project the requested x-coordinates onto this log-transformed y-space.
    - Exponential Reversion: Reverts the interpolated values back to the standard 
      linear domain using the exponential function ($y = \\exp(y_{log})$).
    - Boundary Management: Applies post-processing to explicitly handle queried x-values 
      that fall outside the strict bounds of the input `xp` array, applying either 
      `np.nan` (default) or nearest-neighbor clipping to prevent artificial extrapolation.

    Parameters
    ----------
    x : np.ndarray
        The x-coordinates at which to evaluate the interpolated values.
    xp : np.ndarray
        A 1D array of the x-coordinates of the data points. Must be sorted in
        increasing order.
    yp : np.ndarray
        A 1D array of the y-coordinates of the data points. Must be the same
        size as `xp`.
    epsilon : float, optional
        A minuscule scalar added to `yp` to prevent `log(0)` calculation errors 
        when the array contains exact zeros. Default is 1e-20.
    out_of_bounds : {'nan', 'clip'}, optional
        Specifies how to handle `x` values that fall outside the [min(xp), max(xp)] range.
        - 'nan' (default): Out-of-bounds evaluations are cleanly replaced with `np.nan`.
        - 'clip': Evaluates out-of-bounds requests by forcing them to exactly match 
          the nearest valid endpoint from `yp`.

    Returns
    -------
    np.ndarray
        The interpolated y-values. Shape matches the input `x` array.

    Raises
    ------
    ValueError
        If `out_of_bounds` receives a string other than 'nan' or 'clip'.
    """
    # 1. Convert y-values to a logarithmic scale.
    log_yp = np.log(np.array(yp) + epsilon)

    # 2. Perform linear interpolation in log space.
    log_interp_y = np.interp(x, xp, log_yp)

    # 3. Convert the result back to a linear scale.
    interp_y = np.exp(log_interp_y)

    # 4. Handle out-of-bounds values based on the user's choice.
    if out_of_bounds == 'nan':
        out_of_bounds_mask = (x < np.min(xp)) | (x > np.max(xp))
        interp_y[out_of_bounds_mask] = np.nan
    elif out_of_bounds != 'clip':
        raise ValueError(
            f"Invalid value for out_of_bounds: '{out_of_bounds}'. "
            "Must be 'clip' or 'nan'."
        )

    return interp_y

def load_PEERNGA_record(filepath: str) -> Tuple[np.ndarray, float, int, str]:
    """
    Load record in .at2 format (PEER NGA Databases).

    Parameters
    ----------
    filepath : str
        Path to the .at2 file.

    Returns
    -------
    Tuple[np.ndarray, float, int, str]
        - acc (np.ndarray): Acceleration time series (g).
        - dt (float): Time step (s).
        - npts (int): Number of points in the record.
        - eqname (str): Identifier string (Year_Name_Station_Component).

    Raises
    ------
    FileNotFoundError
        If the specified filepath does not exist.
    ValueError
        If the file format is not as expected.
    """
    try:
        with open(filepath, 'r') as fp:
            next(fp) # Skip header line 1
            line2 = next(fp).strip().split(',')
            if len(line2) < 4:
                raise ValueError("Line 2 format incorrect. Expected Name, Date, Station, Component.")
            date_parts = line2[1].strip().split('/')
            if len(date_parts) < 3:
                raise ValueError("Date format incorrect on Line 2. Expected MM/DD/YYYY.")
            year = date_parts[2]
            eqname = (f"{year}_{line2[0].strip()}_{line2[2].strip()}_comp_{line2[3].strip()}")

            next(fp) # Skip header line 3
            line4 = next(fp).strip().split(',')
            if len(line4) < 2 or 'NPTS=' not in line4[0] or 'DT=' not in line4[1]:
                 raise ValueError("Line 4 format incorrect. Expected NPTS=..., DT=...")
            try:
                npts_str = line4[0].split('=')[1].strip()
                npts = int(npts_str)
                dt_str = line4[1].split('=')[1].split()[0] # Handle potential extra text
                dt = float(dt_str)
            except (IndexError, ValueError) as e:
                raise ValueError(f"Could not parse NPTS or DT from Line 4: {e}")

            # Read acceleration data efficiently
            acc_flat = [float(p) for line in fp for p in line.split()]
            acc = np.array(acc_flat)

            if len(acc) != npts:
                warnings.warn(f"Warning: Number of data points read ({len(acc)}) "
                            f"does not match NPTS specified in header ({npts}). Using read data.")
                npts = len(acc) # Update npts to actual data length

    except FileNotFoundError:
        log.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        log.error(f"Error parsing file {filepath}: {e}")
        raise ValueError(f"Error parsing file {filepath}: {e}")

    return acc, dt, npts, eqname

def get_log_freqs(
        fmin: float = 0.1, 
        fmax: float = 100.0, 
        pts_per_decade: int = 100) -> np.ndarray:
    """
    Generates a logarithmically spaced array of frequencies.

    Methodology:
    - Decade Calculation: Calculates the total number of logarithmic decades 
      between the specified minimum and maximum frequencies using the base-10 
      logarithm ($\log_{10}(f_{max} / f_{min})$).
    - Point Determination: Computes the exact total number of frequency bins 
      required by multiplying the calculated number of decades by the specified 
      `pts_per_decade`. It explicitly adds 1 to ensure that both the start and 
      end frequencies are inclusively captured.
    - Geometric Progression: Utilizes `np.geomspace` to construct the final 
      1D array. This generates a true geometric progression where the multiplicative 
      ratio between consecutive frequency bins is perfectly constant, ensuring 
      even spacing when plotted or evaluated on a logarithmic scale.

    Parameters
    ----------
    fmin : float, optional
        The minimum frequency value in the array (Hz). Must be strictly positive.
        Default is 0.1.
    fmax : float, optional
        The maximum frequency value in the array (Hz). Must be strictly greater 
        than `fmin`. Default is 100.0.
    pts_per_decade : int, optional
        The resolution defined as the number of points to generate for every 
        logarithmic decade (e.g., the number of points between 1 Hz and 10 Hz). 
        Default is 100.

    Returns
    -------
    np.ndarray
        A 1D NumPy array of logarithmically spaced frequency values, starting 
        exactly at `fmin` and ending exactly at `fmax`.
    """
    # Calculate the number of decades the frequency range spans
    num_decades = np.log10(fmax / fmin)

    # Determine the total number of points needed
    total_points = int(pts_per_decade * num_decades) + 1

    # Generate the logarithmically spaced array
    return np.geomspace(fmin, fmax, num=total_points)

def dfactor(
        x: np.ndarray, 
        y: np.ndarray, 
        plot: int = 1) -> Tuple[np.ndarray, np.ndarray, float, float, float]:
    
    """
    Computes the directionality factor of a biaxial ground motion or SDOF response.

    Methodology:
    - Coordinate Conversion: Takes the two orthogonal components of the response 
      (x and y) and converts them into polar coordinates to track the response 
      trajectory's radius and angle over time.
    - Envelope Generation: Uses a Convex Hull algorithm to determine the tightest 
      bounding polygon (envelope) containing the entire 2D response trajectory, and 
      calculates the area of this envelope (Area_hull).
    - Peak Response Area: Identifies the absolute maximum response vector (r_max) 
      and calculates the area of a perfect circle bounded by this maximum radius 
      (Area_circle = pi * r_max^2).
    - Directionality Factor (DF): Calculates the DF as the square root of the ratio 
      of the circular area over the hull area (DF = sqrt(Area_circle / Area_hull)). 
      Values close to 1.0 indicate low directionality (isotropic response), while 
      larger values indicate highly polarized, directional shaking (Rivera & Montejo, 2021).

    Parameters
    ----------
    x : np.ndarray
        The 1D time-series representing the x-coordinate of the response.
    y : np.ndarray
        The 1D time-series representing the y-coordinate of the response.
    plot : int, optional
        Flag (1 or 0) indicating whether to generate a polar plot visualizing 
        the response trajectory, convex hull, and bounding circle. Default is 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, float, float, float]
        - hr (np.ndarray): Polar radii of the convex hull vertices.
        - htheta (np.ndarray): Polar angles (radians) of the convex hull vertices.
        - rmax (float): The maximum response magnitude (radius).
        - thetamax (float): The angle (degrees) at which `rmax` occurs.
        - df (float): The calculated Directionality Factor (DF).

    References
    ----------
    Rivera-Figueroa, A., & Montejo, L. A. (2021). Spectral matching 
    RotD100 target spectra: Effect on records characteristics and 
    seismic response. Earthquake Spectra, 37(4), 1-17.
    """   
    
    n1 = np.size(x); n2 = np.size(y); npo = np.min((n1,n2))
    x = x[:npo]; y = y[:npo]
    
    # Change data to polar coordinates:   
    r = np.sqrt(x**2+y**2)      # radius
    theta = np.arctan2(y,x)     # angle in radians
    
    # Maximum radius and angle of occurence:
    rmax = np.amax(r)
    thetamax = theta[np.argmax(r)]
    
    # Determine the envelope of data points (convex hull):
    points  = np.column_stack((x,y))    # stack data in columns
    hull    = ConvexHull(points)        # convex hull 
    
    # Obtain coordinates of the envelope:
    xh = points[hull.vertices,0]        # x-coordinate of hull
    yh = points[hull.vertices,1]        # y-coorindate of hull
    
    # Change envelope coordinates to polar:
    hr = np.sqrt(xh**2+yh**2)      # radius
    htheta = np.arctan2(yh,xh)     # angle in radians
    
    # Determine area of hull and circle with max radius:
    hArea   = 0.5*np.abs(np.dot(xh,np.roll(yh,1)) 
                         - np.dot(yh,np.roll(xh,1)))    # hull area
    mArea   = np.pi*rmax**2                             # circle area

    # Calculate directionality factor                          
    df = (mArea/hArea)**0.5
    
# =============================================================================
#     PLOT:
# =============================================================================
    
    if plot:
        
        plt.style.use('seaborn-talk')
        plt.figure(figsize=(5,4))
        plt.style.use('seaborn-talk')
        rlimit = rmax*1.05
        ax = plt.subplot(111, projection = 'polar')
        ax.plot([0, thetamax], [0, rmax] ,'-o', color='tab:orange', zorder = 3)
        ax.plot(theta, r, color='tab:blue', zorder = 2, linewidth = 1.5)
        ax.fill(htheta,hr, color='tab:blue',alpha = 0.5, linewidth=3, zorder=2)
        circle = plt.Circle((0, 0), rmax, color="tab:orange", 
                 transform=ax.transData._b, alpha=0.5, linewidth=3, zorder=1)
        ax.add_artist(circle)
        ax.set_facecolor('whitesmoke')
        ax.set_rlabel_position(0)
        ax.set_xlabel('DF = %.2f' %df, labelpad = -10, 
                        bbox=dict(boxstyle='square', fc='whitesmoke', ec = 'k'))
        lines, labels = plt.thetagrids(range(0,360,45),())
        lines, labels = ax.set_rgrids((rlimit*.25, rlimit*.5, rlimit*.75, 
                                            rlimit), fontsize = 8, fmt='%.2f')
        
        # To anotate the max resp and ocurrence angle:
        offset = (0,-30) if thetamax<0 else (0,30)
        if np.abs(thetamax+np.pi/2)<0.001: offset = (-30,0)
        hp = 'left' if thetamax<np.pi/2 and thetamax>-1*np.pi/2 else 'right'
        ax.annotate('(%.1f, %.0fº)'% (rmax, np.degrees(thetamax)), 
                      xy=(thetamax,rmax), 
                      textcoords='offset points', fontsize = 10,
                      xytext=offset, ha=hp, va='center',
                      arrowprops=dict(arrowstyle='->',  
                      connectionstyle="angle3,angleA=0,angleB=90"),
                      bbox = dict(boxstyle='round', fc='0.99'), zorder = 5)
        
        plt.suptitle('Directionality Factor') 
        plt.tight_layout(rect=(0,0,1,1))
    
    return(hr, htheta, rmax, np.degrees(thetamax), df)

def DFSpectra(
    T: np.ndarray, 
    s1: np.ndarray, 
    s2: np.ndarray, 
    z: float, 
    dt: float, 
    theta: np.ndarray, 
    plot: int = 1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Computes the Directionality Response Spectrum (DSA) and RotDnn spectra for 
    biaxial ground motions.

    Methodology:
    - Frequency Domain Solution: Leverages Fast Fourier Transforms (FFT) to solve 
      the SDOF equation of motion for both orthogonal horizontal acceleration 
      components simultaneously across all specified periods. It computes both 
      the displacement response (via receptance) and the absolute acceleration 
      response (via accelerance).
    - RotDnn Spectra: Projects the biaxial displacement response histories across 
      the specified rotation angles (`theta`) to evaluate the period-specific 
      orientation-independent pseudo-spectral accelerations (RotD50 and RotD100).
    - Directionality Spectrum: For every period, it isolates the biaxial absolute 
      acceleration response histories and passes them to the `dfactor` function. 
      This computes the Directionality Factor (DF) at each period, constructing 
      the full Directionality Spectrum of Acceleration (DSA) to quantify how the 
      polarization of the earthquake varies across different resonant frequencies 
      (Rivera & Montejo, 2021).

    Parameters
    ----------
    T : np.ndarray
        Vector of periods at which to calculate the spectra (s).
    s1 : np.ndarray
        Acceleration time-series for the first horizontal component (g).
    s2 : np.ndarray
        Acceleration time-series for the second horizontal component (g).
    z : float
        Damping ratio (e.g., 0.05 for 5%).
    dt : float
        Time step of the acceleration series (s).
    theta : np.ndarray
        Vector of angles (degrees) to evaluate the rotated response.
    plot : int, optional
        Flag (1 or 0) indicating whether to generate a summary figure plotting 
        the RotD50, RotD100, their ratio, and the DSA. Default is 1.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - RotD100 (np.ndarray): The maximum direction pseudo-spectral acceleration (g).
        - RotD50 (np.ndarray): The median direction pseudo-spectral acceleration (g).
        - ratios (np.ndarray): The period-dependent ratio of RotD100 to RotD50.
        - DSA (np.ndarray): The Directionality Spectrum of Acceleration (Directionality Factors).

    References
    ----------
    Rivera-Figueroa, A., & Montejo, L. A. (2021). Spectral matching 
    RotD100 target spectra: Effect on records characteristics and 
    seismic response. Earthquake Spectra, 37(4), 1-17.
    """
    
    percentiles=[50,100]
    pi = np.pi
    theta = theta*pi/180
    
    ntheta = np.size(theta)
    
    n1 = np.size(s1); n2 = np.size(s2); npo = np.min((n1,n2))
    s1 = s1[:npo]; s2 = s2[:npo]
        
    nT  = np.size(T)
    
    SD  = np.zeros((ntheta,nT))
    
    nor = npo
    
    n = int(2**np.ceil(np.log2(npo+10*np.max(T)/dt)))  # add zeros to provide enough quiet time
    fs=1/dt;
    s1 = np.append(s1,np.zeros(n-npo))
    s2 = np.append(s2,np.zeros(n-npo))
    
    fres  = fs/n                            # frequency resolution
    nfrs  = int(np.ceil(n/2))               # number of frequencies
    freqs = fres*np.arange(0,nfrs+1,1)      # vector with frequencies
    ww    = 2*pi*freqs                      # vector with frequencies [rad/s]
    ffts1 = fft(s1)        
    ffts2 = fft(s2) 
    
    DSA = np.zeros(nT)
    
    m = 1
    
    for kk in range(nT):
        w = 2*pi/T[kk] ; k=m*w**2; c = 2*z*m*w
        
        H1 = 1       / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Receptance
        H3 = -ww**2  / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Accelerance
        
        H1 = np.append(H1,np.conj(H1[n//2-1:0:-1]))
        H1[n//2] = np.real(H1[n//2])     # Transfer function (complete) - Receptance
        
        H3 = np.append(H3,np.conj(H3[n//2-1:0:-1]))
        H3[n//2] = np.real(H3[n//2])     # Transfer function (complete) - Accelerance
        
        
        CoFd1 = H1*ffts1   # frequency domain convolution
        d1 = ifft(CoFd1)   # go back to the time domain (displacement)
        d1 = np.real(d1[:nor])
        
        CoFd2 = H1*ffts2   # frequency domain convolution
        d2 = ifft(CoFd2)   # go back to the time domain (displacement)
        d2 = np.real(d2[:nor])
        
        Md1,Mtheta = np.meshgrid(d1,theta,sparse=True, copy=False)
        Md2,_      = np.meshgrid(d2,theta,sparse=True, copy=False)
                
        drot = Md1*np.cos(Mtheta)+Md2*np.sin(Mtheta)
        
        SD[:,kk] = np.max(np.abs(drot),axis=1)
              
        CoFa1 = H3*ffts1   # frequency domain convolution
        a1 = np.real(ifft(CoFa1))   # go back to the time domain 
        a1 = a1 - s1
        a1 = a1[:nor]
        
        CoFa2 = H3*ffts2   # frequency domain convolution
        a2 = np.real(ifft(CoFa2))   # go back to the time domain 
        a2 = a2 - s2
        a2 = a2[:nor]
        
        _,_,_,_,DSA[kk] = dfactor(a1,a2, plot=0)
        
    
    PSA180 = (2*pi/T)**2 * SD
    n = len(percentiles)
    PSArotnn = np.zeros((nT,n))
    
    for k in range(n):
        PSArotnn[:,k] = np.percentile(PSA180,percentiles[k],axis=0)
    
    RotD50 = PSArotnn[:,0]
    RotD100 = PSArotnn[:,1]
    ratios = RotD100/RotD50
    
    if plot:
        
        plt.figure(figsize=(6.5,9))
        plt.subplot(311)
        plt.semilogx(T,RotD50,'-b')
        plt.semilogx(T,RotD100,'-g')
        plt.legend((': RotD50',': RotD100'))
        plt.ylabel('PSA [g]')
        plt.grid(which='both',color='lavender', linestyle='--', linewidth=1)
        plt.subplot(312)
        plt.semilogx(T,ratios,'-k')
        plt.ylabel('RotD100/RotD50')
        plt.ylim((0.9,1.1*np.max(ratios)))
        plt.xlabel('T [s]')
        plt.grid(which='both',color='lavender', linestyle='--', linewidth=1)
        plt.subplot(313)
        plt.semilogx(T,DSA,'-k')
        plt.xlabel('T [s]')
        plt.ylabel('Directionality Factor')
        plt.ylim((0.9,1.1*np.max(DSA)))
        plt.grid(which='both',color='lavender', linestyle='--', linewidth=1)
    
    return RotD100, RotD50, ratios, DSA

def plot_single_component_results(
    results: Dict[str, Any],
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    T1PSA: float = 0.02,
    T2PSA: float = 5.0,
    zi: float = 0.05,
    units: str = 'g') -> Tuple[plt.Figure, plt.Figure, Dict[str, Any]]:
    
    """
    Generates verification plots and metrics for single-component spectral matching results.

    Methodology:
    - Spectral Comparison (Figure 1): Generates a semi-log plot comparing the Target 
      Pseudo-Spectral Acceleration (PSA), the initial Scaled PSA, and the final 
      Matched PSA. A lower sub-panel plots the ratio of the Matched PSA to the 
      Target PSA, highlighting the matching domain (defined by `T1PSA` and `T2PSA`) 
      and verifying compliance with the defined tolerance boundaries (`targetPSAlimits`).
    - Time-Domain Comparison (Figure 2): Creates a three-panel figure displaying the 
      acceleration, velocity, and displacement time histories. The original scaled 
      and final matched records are overlaid. On secondary y-axes, it plots the 
      normalized energy buildup (e.g., normalized Arias Intensity) for each kinematic 
      domain to ensure the temporal energy distribution of the seed record remains intact.
    - Deviation Tracking: Computes the maximum absolute difference between the normalized 
      energy buildups of the matched and original records. This metric quantifies how 
      much the matching process altered the temporal envelope of the seed motion.

    Parameters
    ----------
    results : Dict[str, Any]
        The comprehensive output dictionary generated by the single-component matching 
        function (e.g., `generate_psa_compatible_record`), containing frequency/period 
        arrays, PSA spectra, time vectors, and kinematic histories.
    targetPSAlimits : Tuple[float, float], optional
        The lower and upper tolerance bounds for the PSA matching ratio. Used to plot 
        the limit lines in the ratio sub-panel. Default is (0.9, 1.3).
    T1PSA : float, optional
        The start of the period range (s) over which the matching was performed. Used 
        to shade the target matching region on the plots. Default is 0.02.
    T2PSA : float, optional
        The end of the period range (s) over which the matching was performed. Used 
        to shade the target matching region on the plots. Default is 5.0.
    zi : float, optional
        The damping ratio (as a decimal) associated with the target response spectrum. 
        Used for plot labeling (e.g., 0.05 for 5% damping). Default is 0.05.
    units : str, optional
        The string representation of the acceleration units (e.g., 'g', 'm/s²'). 
        Used for plot labeling. Default is 'g'.

    Returns
    -------
    Tuple[plt.Figure, plt.Figure, Dict[str, list]]
        - fig1 (plt.Figure): The matplotlib figure object containing the spectral 
          comparison and ratio plots.
        - fig2 (plt.Figure): The matplotlib figure object containing the kinematic time 
          histories and normalized energy buildup plots.
        - max_deltas_dict (Dict[str, list]): A dictionary containing the maximum 
          absolute deviations in normalized energy buildup. The exact keys are 
          `'Acc'`, `'Vel'`, and `'Disp'`.
    """    
    mpl.rcParams['font.size'] = 9 
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['font.family'] = 'serif'
    
    LINEWIDTH_MAIN = 1.0
    C_TARGET = 'k'; C_SCALED = 'dimgray'; C_MATCHED = 'salmon'
    COLOR_SHADE = 'steelblue'; ALPHA_SHADE = 0.1

    periods = results['periods']; t = results['t']
    idx_p = np.argsort(periods); p_plot = periods[idx_p]

    # =========================================================================
    # FIGURE 1: SPECTRA
    # =========================================================================
    fig1 = plt.figure(figsize=(6.0, 4.5))
    gs = GridSpec(2, 1, figure=fig1, height_ratios=[7, 3], hspace=0.12)
    ax_top = fig1.add_subplot(gs[0, 0]); ax_bot = fig1.add_subplot(gs[1, 0])
    
    ax_top.axvspan(T1PSA, T2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)
    ax_bot.axvspan(T1PSA, T2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

    target = results['target_psa'][idx_p]
    ax_top.semilogx(p_plot, target, color=C_TARGET, lw=LINEWIDTH_MAIN*2, zorder=5)
    
    if not np.isnan(targetPSAlimits[0]):
        ax_top.semilogx(p_plot, target * targetPSAlimits[0], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
        ax_bot.axhline(targetPSAlimits[0], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)
    if not np.isnan(targetPSAlimits[1]):
        ax_top.semilogx(p_plot, target * targetPSAlimits[1], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
        ax_bot.axhline(targetPSAlimits[1], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)

    ax_top.semilogx(p_plot, results["psa_s"][idx_p], color=C_SCALED, lw=LINEWIDTH_MAIN, label='Scaled')
    ax_top.semilogx(p_plot, results["psa_sc"][idx_p], color=C_MATCHED, lw=LINEWIDTH_MAIN, label='Matched')
    
    mask = (p_plot >= T1PSA) & (p_plot <= T2PSA)
    ax_bot.semilogx(p_plot, np.where(mask, results["psa_s"][idx_p]/target, np.nan), color=C_SCALED, lw=LINEWIDTH_MAIN)
    ax_bot.semilogx(p_plot, np.where(mask, results["psa_sc"][idx_p]/target, np.nan), color=C_MATCHED, lw=LINEWIDTH_MAIN)
        
    ax_top.set_ylabel(f'PSA ({units})'); ax_top.set_xlim(0.01, 10); ax_top.set_xticklabels([])
    ax_top.set_ylim(bottom=np.nanmin(target*0.5), top=np.nanmax(target*1.2))
    ax_bot.set_xlim(0.01, 10); ax_bot.set_ylim(0.5, 1.5); ax_bot.set_ylabel('Ratio'); ax_bot.set_xlabel('Period (s)')
    
    handles = [
        Line2D([0],[0], color=C_SCALED, lw=LINEWIDTH_MAIN, label='Scaled'),
        Line2D([0],[0], color=C_MATCHED, lw=LINEWIDTH_MAIN, label='Matched'),
        Line2D([0],[0], color=C_TARGET, lw=1.5, label='Target')
    ]
    fig1.legend(handles=handles, loc='upper center', ncol=3, bbox_to_anchor=(0.5, 0.99), frameon=False)
    fig1.tight_layout(rect=[0, 0, 1, 0.88]); fig1.subplots_adjust(top=0.93)

    # =========================================================================
    # FIGURE 2: TIME HISTORIES
    # =========================================================================
    if units == 'g': conv_vel = 980.665; conv_disp = 980.665; u_vel = 'cm/s'; u_disp = 'cm'
    else: conv_vel = 1.0; conv_disp = 1.0; u_vel = f'{units}-s'; u_disp = f'{units}-s^2'

    # Single column layout since it's just one component
    fig2, axs = plt.subplots(3, 1, figsize=(7.5, 6), sharex=True)
    trace_colors = [C_SCALED, C_MATCHED]
    max_deltas_dict = {}
    
    th_list = [results['s_scaled'], results['sc']]
    v_list = [results['vel_s'] * conv_vel, results['vel_sc'] * conv_vel]
    d_list = [results['disp_s'] * conv_disp, results['disp_sc'] * conv_disp]
    nai_list = [results['ai_s'], results['ai_sc']]
    ncsv_list = [results['csv_s'], results['csv_sc']]
    ncsd_list = [results['csd_s'], results['csd_sc']]

    groups = [(th_list, nai_list, 'Acc', units, 'Norm. CSA'), (v_list, ncsv_list, 'Vel', u_vel, 'Norm. CSV'), (d_list, ncsd_list, 'Disp', u_disp, 'Norm. CSD')]
    y_pos = 0.05; x_anchor = 0.98

    for row, (th, norm, name, unit, norm_lbl) in enumerate(groups):
        axL = axs[row]; axR = axL.twinx()
        for d, c in zip(th, trace_colors): 
            if d is not None and not np.all(np.isnan(d)): axL.plot(t, d, color=c, lw=0.5, alpha=0.6)
        for n, c in zip(norm, trace_colors): 
            if n is not None and not np.all(n == 0.0): axR.plot(t, n, color=c, lw=1.0)
        
        deltas = [np.max(np.abs(n - norm[0])) for n in norm[1:]]
        max_deltas_dict[name] = deltas
        
        if not np.all(norm[0] == 0.0):
            offset_val = 0.00; step = 0.12
            for i, delta in enumerate(reversed(deltas)):
                c_idx = len(deltas) - i 
                comma = ", " if i > 0 else ""
                axL.text(x_anchor - offset_val, y_pos, f"{delta:.2f}{comma}", transform=axL.transAxes, fontsize=8, color=trace_colors[c_idx], va='bottom', ha='right', fontweight='bold')
                offset_val += step
            axL.text(x_anchor - offset_val, y_pos, r'max $\Delta$:', transform=axL.transAxes, fontsize=8, color='black', va='bottom', ha='right')

        axL.set_ylabel(f'{name} [{unit}]'); axR.set_ylabel(norm_lbl, color='k', fontsize=9); axR.set_ylim(-0.05, 1.05)
        all_data = np.concatenate([d for d in th if d is not None and not np.all(np.isnan(d))]); mx = np.nanmax(np.abs(all_data)); limit = 1.05 * mx if mx !=0 and np.isfinite(mx) else 1.0
        axL.set_ylim(-limit, limit); axL.grid(False); axR.grid(False)

    axs[2].set_xlabel('Time [s]')
    
    # Use only 2 handles (Scaled, Matched) for the time history legend
    fig2.legend(handles=handles[:2], loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=9)
    fig2.tight_layout(rect=[0, 0, 1, 0.95]); fig2.subplots_adjust(top=0.92) 

    return fig1, fig2, max_deltas_dict

def plot_rotdnn_results(
    results: Dict[str, Any],
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    T1PSA: float = 0.02,
    T2PSA: float = 5.0,
    zi: float = 0.05,
    units: str = 'g',
    plot_directionality: bool = False,
    polar_freqs: List[float] = [0.5, 1, 2, 4, 8, 12, 16, 20]) -> Union[Tuple['plt.Figure', 'plt.Figure', Dict], Tuple['plt.Figure', 'plt.Figure', 'plt.Figure', 'plt.Figure', Dict]]:
    
    """
    Generates verification plots and metrics for biaxial RotDnn spectral matching results.

    Methodology:
    - Spectral Comparison (Figure 1): Generates a semi-log plot comparing the orientation-independent 
      Target Pseudo-Spectral Acceleration (RotDnn PSA) against the initial Scaled RotDnn PSA and the 
      final Matched RotDnn PSA. Sub-panels plot the ratio of the Matched PSA to the Target PSA, 
      highlighting the matching domain (defined by `T1PSA` and `T2PSA`) and verifying compliance 
      with the defined tolerance boundaries (`targetPSAlimits`).
    - Time-Domain Comparison (Figure 2): Creates a 3x2 grid of subplots displaying the acceleration, 
      velocity, and displacement time histories for both horizontal components (Component 1 on the left, 
      Component 2 on the right). It overlays the original scaled and final matched records. On secondary 
      y-axes, it plots the normalized energy buildup for each kinematic domain to ensure the temporal 
      envelope of the seed record is preserved.
    - Deviation Tracking: Computes the maximum absolute difference between the normalized energy buildups 
      of the matched and original records for both components independently, quantifying temporal alteration.
    - Directionality (Optional - Figures 3 & 4): If `plot_directionality` is activated, the function 
      solves the biaxial SDOF response at various resonant frequencies. It plots polar representations 
      of the response trajectories (Figure 3) at specific sample frequencies (`polar_freqs`), and constructs 
      the period-dependent RotD100/RotD50 ratios alongside the Directionality Spectrum of Acceleration (DSA) 
      using the `dfactor` routine (Figure 4) to verify that isotropic/polarized characteristics are maintained.

    Parameters
    ----------
    results : Dict[str, Any]
        The comprehensive output dictionary generated by the RotDnn matching function 
        (e.g., `generate_rotdnn_psa_compatible_record`), containing frequency/period arrays, 
        PSA spectra, time vectors, and kinematic histories for both components.
    targetPSAlimits : Tuple[float, float], optional
        The lower and upper tolerance bounds for the PSA matching ratio. Used to plot 
        the limit lines in the ratio sub-panels. Default is (0.9, 1.3).
    T1PSA : float, optional
        The start of the period range (s) over which the matching was performed. Used 
        to shade the target matching region on the plots. Default is 0.02.
    T2PSA : float, optional
        The end of the period range (s) over which the matching was performed. Used 
        to shade the target matching region on the plots. Default is 5.0.
    zi : float, optional
        The damping ratio (as a decimal) associated with the target response spectrum. 
        Used for plot labeling (e.g., 0.05 for 5% damping). Default is 0.05.
    units : str, optional
        The string representation of the acceleration units (e.g., 'g', 'm/s²'). 
        Used for plot labeling. Default is 'g'.
    plot_directionality : bool, optional
        If True, executes the necessary SDOF simulations to compute and plot the biaxial 
        polar response trajectories and the Directionality Spectrum of Acceleration (DSA). 
        Default is False.
    polar_freqs : List[float], optional
        A list of specific frequencies (Hz) at which to plot the polar response trajectories 
        in Figure 3 (only used if `plot_directionality` is True). 
        Default is [0.5, 1, 2, 4, 8, 12, 16, 20].

    Returns
    -------
    Union[Tuple[plt.Figure, plt.Figure, Dict[str, list]], Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure, Dict[str, list]]]
        If `plot_directionality` is False:
        - fig1 (plt.Figure): The spectral comparison and ratio plots.
        - fig2 (plt.Figure): The kinematic time histories and energy buildup plots.
        - max_deltas_dict (Dict[str, list]): The maximum absolute deviations in normalized energy 
          buildup. Keys are `'Acc1'`, `'Acc2'`, `'Vel1'`, `'Vel2'`, `'Disp1'`, and `'Disp2'`.

        If `plot_directionality` is True, returns all of the above, plus:
        - fig3 (plt.Figure): The polar plots of biaxial response trajectories.
        - fig4 (plt.Figure): The RotD100/RotD50 ratios and the Directionality Spectrum of Acceleration.
    """
    
    mpl.rcParams['font.size'] = 9 
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['font.family'] = 'serif'
    
    LINEWIDTH_MAIN = 1.0
    C_TARGET = 'k'; C_SCALED = 'dimgray'; C_PSA = 'cornflowerblue'
    COLOR_SHADE = 'steelblue'; ALPHA_SHADE = 0.1

    periods = results['periods']; t = results['t']; nn = results.get('nn', 50)
    sf = results.get('scale_factor', 1.0) 
    
    idx_p = np.argsort(periods)
    p_plot = periods[idx_p]

    # =========================================================================
    # FIGURE 1: SPECTRA (1x3 Subplot Grid)
    # =========================================================================
    fig1 = plt.figure(figsize=(10.0, 4.5))
    gs = GridSpec(2, 3, figure=fig1, height_ratios=[7, 3], hspace=0.12, wspace=0.15)

    target = results['target_psa'][idx_p]
    
    p_s1 = results.get('psa_s1'); p_sc1 = results.get('psa_sc1')
    p_s2 = results.get('psa_s2'); p_sc2 = results.get('psa_sc2')
    p_data_s = results.get("psa_s"); p_data_sc = results.get("psa_sc")

    # Define the data sets for each of the 3 columns
    columns_data = [
        ('Component 1', p_s1, p_sc1),
        ('Component 2', p_s2, p_sc2),
        (f'RotD{nn}', p_data_s, p_data_sc)
    ]

    for i, (title, d_s, d_sc) in enumerate(columns_data):
        ax_top = fig1.add_subplot(gs[0, i])
        ax_bot = fig1.add_subplot(gs[1, i])

        # Shaded regions in all 3 subplots
        ax_top.axvspan(T1PSA, T2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)
        ax_bot.axvspan(T1PSA, T2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

        # Target and Limits
        ax_top.semilogx(p_plot, target, color=C_TARGET, lw=LINEWIDTH_MAIN*2, zorder=5)
        if not np.isnan(targetPSAlimits[0]):
            ax_top.semilogx(p_plot, target * targetPSAlimits[0], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
            ax_bot.axhline(targetPSAlimits[0], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)
        if not np.isnan(targetPSAlimits[1]):
            ax_top.semilogx(p_plot, target * targetPSAlimits[1], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
            ax_bot.axhline(targetPSAlimits[1], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)

        # Scaled and Matched Traces
        if d_s is not None:
            data_s_sorted = d_s[idx_p]
            ax_top.semilogx(p_plot, data_s_sorted, color=C_SCALED, lw=LINEWIDTH_MAIN)
        if d_sc is not None:
            data_sc_sorted = d_sc[idx_p]
            ax_top.semilogx(p_plot, data_sc_sorted, color=C_PSA, lw=LINEWIDTH_MAIN)
            
        # Ratio plotting
        mask = (p_plot >= T1PSA) & (p_plot <= T2PSA)
        if d_s is not None:
            ax_bot.semilogx(p_plot, np.where(mask, data_s_sorted/target, np.nan), color=C_SCALED, lw=LINEWIDTH_MAIN)
        if d_sc is not None:
            ax_bot.semilogx(p_plot, np.where(mask, data_sc_sorted/target, np.nan), color=C_PSA, lw=LINEWIDTH_MAIN)
            
        # Formatting per column
        ax_top.set_title(title, fontsize=10, fontweight='bold')
        ax_top.set_xlim(0.01, 10); ax_top.set_xticklabels([])
        ax_bot.set_xlim(0.01, 10); ax_bot.set_ylim(0.5, 1.5)
        ax_bot.set_xlabel('Period (s)')
        ax_bot.axhline(1.0, color='k', lw=LINEWIDTH_MAIN, zorder=5)
        
        mask_plot = (p_plot >= 0.01) & (p_plot <= 10)
        y_max = np.nanmax((target * targetPSAlimits[1])[mask_plot]) * 1.1
        y_min = np.nanmin((target * targetPSAlimits[0])[mask_plot]) * 0.7
        ax_top.set_ylim(bottom=y_min, top=y_max)
        
        if i == 0:
            ax_top.set_ylabel(f'PSA ({units})')
            ax_bot.set_ylabel('Ratio')
        else:
            ax_top.set_yticklabels([])
            ax_bot.set_yticklabels([])

    handles = [
        Line2D([0],[0], color=C_SCALED, lw=LINEWIDTH_MAIN, label='Scaled'),
        Line2D([0],[0], color=C_PSA, lw=LINEWIDTH_MAIN, label='Matched'),
        Line2D([0],[0], color=C_TARGET, lw=LINEWIDTH_MAIN*2, label='Target'),
        Line2D([0],[0], color=C_TARGET, lw=LINEWIDTH_MAIN, ls='--', label='Limits')
    ]
    
    fig1.legend(handles=handles, loc='upper center', ncol=4, bbox_to_anchor=(0.5, 1.02), frameon=False, columnspacing=1.5)
    fig1.tight_layout()
    fig1.subplots_adjust(top=0.88, bottom=0.12)

    # =========================================================================
    # FIGURE 2: TIME HISTORIES (Original Version)
    # =========================================================================
    if units == 'g': conv_vel = 980.665; conv_disp = 980.665; u_vel = 'cm/s'; u_disp = 'cm'
    else: conv_vel = 1.0; conv_disp = 1.0; u_vel = f'{units}-s'; u_disp = f'{units}-s^2'

    fig2, axs = plt.subplots(3, 2, figsize=(7.5, 6), sharex=True)
    trace_colors = [C_SCALED, C_PSA]
    max_deltas_dict = {}
    
    for comp in [1, 2]:
        th_list = [results[f's{comp}_scaled'], results[f'sc{comp}']]
        v_list = [results[f'vel{comp}_s'] * conv_vel, results[f'vel{comp}_sc'] * conv_vel]
        d_list = [results[f'disp{comp}_s'] * conv_disp, results[f'disp{comp}_sc'] * conv_disp]
        nai_list = [results[f'ai{comp}_s'], results[f'ai{comp}_sc']]
        ncsv_list = [results[f'csv{comp}_s'], results[f'csv{comp}_sc']]
        ncsd_list = [results[f'csd{comp}_s'], results[f'csd{comp}_sc']]

        groups = [
            (th_list, nai_list, 'Acc', units, 'Norm. CSA'),
            (v_list, ncsv_list, 'Vel', u_vel, 'Norm. CSV'),  
            (d_list, ncsd_list, 'Disp', u_disp, 'Norm. CSD') 
        ]

        y_pos = 0.05; x_anchor = 0.98; col = comp - 1

        for row, (th, norm, name, unit, norm_lbl) in enumerate(groups):
            axL = axs[row, col]; axR = axL.twinx()
            for d, c in zip(th, trace_colors): 
                if d is not None and not np.all(np.isnan(d)): axL.plot(t, d, color=c, lw=0.5, alpha=0.6)
            for n, c in zip(norm, trace_colors): 
                if n is not None and not np.all(n == 0.0): axR.plot(t, n, color=c, lw=1.0)
            
            deltas = [np.max(np.abs(n - norm[0])) for n in norm[1:]]
            max_deltas_dict[f"{name}{comp}"] = deltas
            
            if not np.all(norm[0] == 0.0):
                offset_val = 0.00; step = 0.12
                for i, delta in enumerate(reversed(deltas)):
                    c_idx = len(deltas) - i 
                    comma = ", " if i > 0 else ""
                    axL.text(x_anchor - offset_val, y_pos, f"{delta:.2f}{comma}", transform=axL.transAxes, fontsize=8, color=trace_colors[c_idx], va='bottom', ha='right', fontweight='bold')
                    offset_val += step
                axL.text(x_anchor - offset_val, y_pos, r'max $\Delta$:', transform=axL.transAxes, fontsize=8, color='black', va='bottom', ha='right')

            axL.set_ylabel(f'{name} [{unit}]' if col == 0 else '')
            axR.set_ylabel(norm_lbl if col == 1 else '', color='k', fontsize=9)
            axR.set_ylim(-0.05, 1.05)
            
            all_data = np.concatenate([d for d in th if d is not None and not np.all(np.isnan(d))])
            mx = np.nanmax(np.abs(all_data)); limit = 1.05 * mx if mx !=0 and np.isfinite(mx) else 1.0
            axL.set_ylim(-limit, limit); axL.grid(False); axR.grid(False)
            
            if col == 0: axR.set_yticks([])
            if col == 1: axL.set_yticks([])

    axs[2, 0].set_xlabel('Time [s]'); axs[2, 1].set_xlabel('Time [s]')
    
    fig2.legend(handles=handles[:2], loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=2, fontsize=9)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.subplots_adjust(top=0.92) 

    if not plot_directionality: return fig1, fig2, max_deltas_dict

    # =========================================================================
    # FIGURE 3: POLAR SPECTRA (Original Version)
    # =========================================================================
    target_Ts = [1.0 / f for f in polar_freqs]
    indices = [np.argmin(np.abs(periods - pt)) for pt in target_Ts]
    actual_freqs = [1.0 / periods[i] for i in indices]
    
    psa_180_list = [results.get('psa_180_seed', np.zeros((1,len(periods)))) * sf, results.get('psa_180_sc', np.zeros((1,len(periods))))]
        
    n_plots = len(polar_freqs)
    ncols = 4
    nrows = int(np.ceil(n_plots / ncols))
    fig_height = 1.8 * nrows 
    
    fig3, axes = plt.subplots(nrows, ncols, subplot_kw={'projection': 'polar'}, figsize=(7.5, fig_height))
    axes_flat = axes.flatten() if n_plots > 1 else [axes]
    theta_rad = np.linspace(0, 2*np.pi, 360, endpoint=False)
    
    for i, ax in enumerate(axes_flat):
        if i < n_plots:
            pidx = indices[i]; f_val = actual_freqs[i]
            for mat, lbl, col in zip(psa_180_list, ['Scaled', 'PSA matched'], trace_colors):
                if mat is not None and mat.size > 1:
                    data = mat[:, pidx]; mx = np.max(data)
                    if mx > 0: data = data / mx
                    data_wrap = np.concatenate([data, data]); l_str = lbl if i == (n_plots - 1) else ""
                    ax.plot(theta_rad, data_wrap, color=col, linewidth=1.0, label=l_str)
            ax.set_title(f"f = {f_val:.0f} Hz", fontsize=8, pad=2); ax.set_xticks([]); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels([]); ax.set_ylim(0, 1.1)
            ax.grid(True, linestyle=':', alpha=0.5); ax.set_theta_zero_location("E")
        else: ax.axis('off')
    
    fig3.legend(loc='lower center', ncol=len(trace_colors), bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False, columnspacing=1.0)
    plt.tight_layout(rect=(0, 0.06, 1, 1.0))

    # =========================================================================
    # FIGURE 4: DIRECTIONALITY ANALYSIS (Original Version)
    # =========================================================================
    fig4, (ax_ratio, ax_dsa) = plt.subplots(1, 2, figsize=(7.5, 4.0))
    
    def calc_dsa_curve_fd(a1, a2, dt_rec, T_vec, damping):
        if a1 is None or a2 is None: return np.full_like(T_vec, np.nan)
        npts = len(a1); N = 2**int(np.ceil(np.log2(npts)))
        A1_w = np.fft.rfft(a1, n=N); A2_w = np.fft.rfft(a2, n=N)
        freqs_fft = np.fft.rfftfreq(N, d=dt_rec); omega = 2 * np.pi * freqs_fft
        dsa_list = []
        for T in T_vec:
            wn = 2 * np.pi / T
            with np.errstate(divide='ignore', invalid='ignore'): H = (wn**2 + 2j*damping*wn*omega) / (wn**2 - omega**2 + 2j*damping*wn*omega)
            H[0] = 1.0; R1_w = A1_w * H; R2_w = A2_w * H
            r1 = np.fft.irfft(R1_w, n=N)[:npts]; r2 = np.fft.irfft(R2_w, n=N)[:npts]
            d_out = dfactor(r1, r2, plot=0); dsa_list.append(d_out[4]) 
        return np.array(dsa_list)

    data_groups = [
        (results.get('psa_180_seed', np.zeros((1,len(periods))))*sf, results['s1_scaled'], results['s2_scaled'], C_SCALED, 'Scaled'),
        (results.get('psa_180_sc', np.zeros((1,len(periods)))), results['sc1'], results['sc2'], C_PSA, 'PSA matched')
    ]

    for (mat, acc_x, acc_y, col, lbl) in data_groups:
        if mat is not None and mat.size > 1:
            rotd100 = np.max(mat, axis=0)[idx_p]
            rotd50 = np.percentile(mat, 50, axis=0)[idx_p]
            ratio_curve = rotd100 / rotd50
            ax_ratio.semilogx(p_plot, ratio_curve, color=col, lw=1.0, label=lbl)
            
            dsa_curve = calc_dsa_curve_fd(acc_x, acc_y, results['dt'], periods, zi)[idx_p]
            ax_dsa.semilogx(p_plot, dsa_curve, color=col, lw=1.0, label=lbl)

    ax_ratio.set_xlabel('Period (s)'); ax_ratio.set_ylabel('RotD100 / RotD50'); ax_ratio.set_xlim(0.01, 10)
    ax_ratio.grid(True, which='both', linestyle=':', alpha=0.5); ax_ratio.axvspan(T1PSA, T2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

    ax_dsa.set_xlabel('Period (s)'); ax_dsa.set_ylabel('Directionality Factor'); ax_dsa.set_xlim(0.01, 10)
    ax_dsa.grid(True, which='both', linestyle=':', alpha=0.5); ax_dsa.axvspan(T1PSA, T2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

    legend_handles_4 = [Line2D([0], [0], color=C_SCALED, lw=1.5, label='Scaled'), Line2D([0], [0], color=C_PSA, lw=1.5, label='PSA matched')]
    fig4.legend(handles=legend_handles_4, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False, columnspacing=1.0)
    plt.tight_layout(rect=(0, 0.1, 1, 1.0))

    return fig1, fig2, fig3, fig4, max_deltas_dict

def plot_psa_psd_fas_results(
    results: Dict[str, Any],
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    PSDreduction: float = 1.0,
    FASreduction: float = 1.0,
    F1PSA: float = 0.2,
    F2PSA: float = 50.0,
    F1Check: float = 0.3,
    F2Check: float = 30.0,
    zi: float = 0.05,
    units: str = 'g') -> Tuple[plt.Figure, plt.Figure, Dict[str, Any]]:
    
    """
    Generates verification plots and metrics for advanced single-component spectral 
    matching results (PSA, FAS, and PSD domains).

    Methodology:
    - Frequency Domain Comparison (Figure 1): Generates a dynamically sized figure with 
      up to three sub-panels depending on which matching stages were activated. 
      - The **PSA panel** compares the initial Scaled PSA and the sequentially matched 
        PSAs against the Target PSA. A lower sub-plot displays the ratio to the target, 
        highlighting the primary matching domain (`F1PSA` to `F2PSA`) and the target limits.
      - The **FAS panel** (if applicable) compares the smoothed FAS of the records against 
        the smoothed Minimum Target FAS, plotting the threshold (`FASreduction`) used during matching.
      - The **PSD panel** (if applicable) compares the one-sided PSD of the records against 
        the target PSD, plotting the threshold (`PSDreduction`) used during matching. 
        Shaded regions highlight the frequency range (`F1Check` to `F2Check`) used for the 
        FAS and PSD compliance checks.
    - Time-Domain Comparison (Figure 2): Creates a three-panel figure displaying the 
      acceleration, velocity, and displacement time histories, overlaying the scaled 
      seed and all successfully matched stages. On secondary y-axes, it plots the 
      normalized energy buildup (e.g., Normalized CSA/CSV/CSD) for each kinematic domain.
    - Deviation Tracking: Computes the maximum absolute difference between the normalized 
      energy buildups of each matched stage and the original scaled record, printing 
      the maximum deltas directly onto the time-domain plots to track temporal alteration.

    Parameters
    ----------
    results : Dict[str, Any]
        The comprehensive output dictionary generated by the advanced matching function 
        (e.g., `generate_psa_psd_fas_compatible_record`), containing frequency/period 
        arrays, spectra, time vectors, and kinematic histories for each activated stage.
    targetPSAlimits : Tuple[float, float], optional
        The lower and upper tolerance bounds for the PSA matching ratio. Used to plot 
        the limit lines in the ratio sub-panel. Default is (0.9, 1.3).
    PSDreduction : float, optional
        The scaling factor applied to the target PSD during the matching algorithm 
        to define the minimum required threshold. Used for plotting the limit. Default is 1.0.
    FASreduction : float, optional
        The scaling factor applied to the target FAS during the matching algorithm 
        to define the minimum required threshold. Used for plotting the limit. Default is 1.0.
    F1PSA : float, optional
        The minimum frequency (Hz) for the primary PSA matching domain. Default is 0.2.
    F2PSA : float, optional
        The maximum frequency (Hz) for the primary PSA matching domain. Default is 50.0.
    F1Check : float, optional
        The minimum frequency (Hz) evaluated during the FAS and PSD compliance checks. 
        Used to shade the verification region. Default is 0.3.
    F2Check : float, optional
        The maximum frequency (Hz) evaluated during the FAS and PSD compliance checks. 
        Used to shade the verification region. Default is 30.0.
    zi : float, optional
        The damping ratio (as a decimal) associated with the target response spectrum. 
        Used for plot labeling. Default is 0.05.
    units : str, optional
        The string representation of the acceleration units (e.g., 'g', 'm/s²'). 
        Used for plot labeling. Default is 'g'.

    Returns
    -------
    Tuple[plt.Figure, plt.Figure, Dict[str, list]]
        - fig1 (plt.Figure): The dynamically generated frequency domain plots (PSA, FAS, PSD).
        - fig2 (plt.Figure): The kinematic time histories and normalized energy buildup plots.
        - max_deltas_dict (Dict[str, list]): A dictionary containing lists of the maximum 
          absolute deviations in normalized energy buildup for each matched stage relative 
          to the scaled seed. The exact keys are `'Acc'`, `'Vel'`, and `'Disp'`.
    """
    
    mpl.rcParams['font.size'] = 9 
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['font.family'] = 'serif'
    
    LINEWIDTH_MAIN = 1.0
    C_TARGET = 'k'; C_SCALED = 'dimgray'; C_PSA = 'cornflowerblue'; C_FAS = 'blueviolet'; C_PSD = 'salmon'
    COLOR_SHADE = 'steelblue'; ALPHA_SHADE = 0.1

    freqs = results['freqs']; t = results['t']
    
    # --- Detect Which Modes Were Run ---
    do_fas = not np.all(np.isnan(results.get('target_fas', [np.nan])))
    do_psd = not np.all(np.isnan(results.get('target_psd', [np.nan])))

    # =========================================================================
    # FIGURE 1: SPECTRA (Dynamic Grid)
    # =========================================================================
    plot_configs = [('PSA', results['target_psa'], targetPSAlimits, (F1PSA, F2PSA), f'PSA ({units})', False)]
    if do_fas: plot_configs.append(('FAS', results['target_fas'], (FASreduction, np.nan), (F1Check, F2Check), f'FAS ({units}-s)', True))
    if do_psd: plot_configs.append(('PSD', results['target_psd'], (PSDreduction, np.nan), (F1Check, F2Check), fr'PSD ({units}$^2$/Hz)', True))

    num_cols = len(plot_configs)
    fig1 = plt.figure(figsize=(2.5 * num_cols, 4))
    gs = GridSpec(2, num_cols, figure=fig1, height_ratios=[7, 3], hspace=0.12, wspace=0.45)

    # Build active traces list dynamically
    active_keys = ['sc']; active_colors = [C_PSA]; active_labels = ['PSA matched']
    if do_fas: active_keys.append('sc_fas'); active_colors.append(C_FAS); active_labels.append('FAS adj')
    if do_psd: active_keys.append('sca'); active_colors.append(C_PSD); active_labels.append('PSD adj')

    for i, (spec_type, target, lims, mask_range, ylabel, is_loglog) in enumerate(plot_configs):
        ax_top = fig1.add_subplot(gs[0, i])
        ax_bot = fig1.add_subplot(gs[1, i])

        ax_top.axvspan(mask_range[0], mask_range[1], color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)
        ax_bot.axvspan(mask_range[0], mask_range[1], color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

        plot_cmd_top = ax_top.loglog if is_loglog else ax_top.semilogx
        plot_cmd_bot = ax_bot.semilogx
        
        plot_cmd_top(freqs, target, color=C_TARGET, lw=LINEWIDTH_MAIN*2, zorder=5)

        if not np.isnan(lims[0]):
            plot_cmd_top(freqs, target * lims[0], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
            ax_bot.axhline(lims[0], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)
        if not np.isnan(lims[1]):
            plot_cmd_top(freqs, target * lims[1], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
            ax_bot.axhline(lims[1], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)

        p_data_s = results[f"{spec_type.lower()}_s"]
        plot_cmd_top(freqs, p_data_s, color=C_SCALED, lw=LINEWIDTH_MAIN)
        mask = (freqs >= mask_range[0]) & (freqs <= mask_range[1])
        plot_cmd_bot(freqs, np.where(mask, p_data_s/target, np.nan), color=C_SCALED, lw=LINEWIDTH_MAIN)
        
        for key, color in zip(active_keys, active_colors):
            p_data = results[f"{spec_type.lower()}_{key}"]
            plot_cmd_top(freqs, p_data, color=color, lw=LINEWIDTH_MAIN)
            plot_cmd_bot(freqs, np.where(mask, p_data/target, np.nan), color=color, lw=LINEWIDTH_MAIN)
            
        ax_top.set_ylabel(ylabel); ax_top.set_xlim(0.1, 100); ax_top.set_xticklabels([])
        
        mask_plot = (freqs >= 0.1) & (freqs <= 100)
        if is_loglog:
            y_min = np.nanmin((target * (lims[0] if not np.isnan(lims[0]) else 1.0))[mask_plot]) * 0.7
            y_max = np.nanmax(target[mask_plot]) * 1.5
            ax_top.set_ylim(bottom=y_min, top=y_max)
        else:
            y_max = np.nanmax((target * (lims[1] if not np.isnan(lims[1]) else 1.0))[mask_plot]) * 1.1
            ax_top.set_ylim(bottom=-0.015, top=y_max)
        
        ax_bot.set_xlim(0.1, 100); ax_bot.set_ylim(0.5, 1.5)
        ax_bot.set_ylabel('Ratio'); ax_bot.set_xlabel('Frequency (Hz)')
        ax_bot.axhline(1.0, color='k', lw=LINEWIDTH_MAIN, zorder=5)

    handles = [Line2D([0],[0], color=C_SCALED, lw=LINEWIDTH_MAIN, label='Scaled')]
    for c, l in zip(active_colors, active_labels):
        handles.append(Line2D([0],[0], color=c, lw=LINEWIDTH_MAIN, label=l))
    handles.extend([
        Line2D([0],[0], color=C_TARGET, lw=LINEWIDTH_MAIN*2, label='Target'),
        Line2D([0],[0], color=C_TARGET, lw=LINEWIDTH_MAIN, ls='--', label='Limits')
    ])
    fig1.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 0.99), frameon=False, columnspacing=1.0)
    fig1.subplots_adjust(top=0.91, bottom=0.12, left=0.12, right=0.97)

    # =========================================================================
    # FIGURE 2: TIME HISTORIES (Dynamic Traces)
    # =========================================================================
    if units == 'g': conv_vel = 980.665; conv_disp = 980.665; u_vel = 'cm/s'; u_disp = 'cm'
    else: conv_vel = 1.0; conv_disp = 1.0; u_vel = f'{units}-s'; u_disp = f'{units}-s^2'

    th_list = [results['s_scaled']]
    v_list = [results['vel_s'] * conv_vel]
    d_list = [results['disp_s'] * conv_disp]
    
    n_ai_list = [results['ai_s'] / results['ai_s'][-1]]
    trace_colors = [C_SCALED]
    
    # Correctly compute normalized squared integration for CSV and CSD natively
    def calc_norm_sq(arr):
        if arr is None or np.all(np.isnan(arr)): return np.zeros_like(t)
        sq_int = integrate.cumulative_trapezoid(arr**2, t, initial=0)
        return sq_int / sq_int[-1] if sq_int[-1] != 0 else np.zeros_like(t)

    n_csv_list = [calc_norm_sq(results['vel_s'])]
    n_csd_list = [calc_norm_sq(results['disp_s'])]

    for key, color in zip(active_keys, active_colors):
        th_list.append(results[key])
        
        v_arr = results.get(f"vel_{key}")
        d_arr = results.get(f"disp_{key}")
        v_list.append(v_arr * conv_vel if v_arr is not None else None)
        d_list.append(d_arr * conv_disp if d_arr is not None else None)
        
        ai_arr = results[f"ai_{key}"]
        n_ai_list.append(ai_arr / ai_arr[-1] if (ai_arr is not None and ai_arr[-1] != 0) else np.zeros_like(t))
        
        n_csv_list.append(calc_norm_sq(v_arr))
        n_csd_list.append(calc_norm_sq(d_arr))
        
        trace_colors.append(color)

    groups = [
        (th_list, n_ai_list, 'Acc', units, 'Norm. CSA'),
        (v_list, n_csv_list, 'Vel', u_vel, 'Norm. CSV'),  
        (d_list, n_csd_list, 'Disp', u_disp, 'Norm. CSD') 
    ]

    fig2, axs = plt.subplots(3, 1, figsize=(7.5, 6), sharex=True)
    max_deltas_dict = {}
    y_pos = 0.05; x_anchor = 0.98

    for row, (th, norm, name, unit, norm_lbl) in enumerate(groups):
        axL = axs[row]; axR = axL.twinx()
        
        for d, c in zip(th, trace_colors): 
            if d is not None and not np.all(np.isnan(d)): axL.plot(t, d, color=c, lw=0.5, alpha=0.6)
            
        for n, c in zip(norm, trace_colors): 
            if n is not None and not np.all(n == 0.0): axR.plot(t, n, color=c, lw=1.0)
            
        deltas = [np.max(np.abs(n - norm[0])) for n in norm[1:]]
        max_deltas_dict[name] = deltas
        
        # Dynamic Delta Labeling (right-to-left placement based on number of active curves)
        if not np.all(norm[0] == 0.0):
            offset_val = 0.00
            step = 0.12 # Increased spacing so numbers don't crowd
            for i, delta in enumerate(reversed(deltas)):
                c_idx = len(deltas) - i 
                # Add comma only if it is NOT the right-most value (i=0)
                comma = ", " if i > 0 else ""
                axL.text(x_anchor - offset_val, y_pos, f"{delta:.2f}{comma}", transform=axL.transAxes, fontsize=8, color=trace_colors[c_idx], va='bottom', ha='right', fontweight='bold')
                offset_val += step
            axL.text(x_anchor - offset_val, y_pos, r'max $\Delta$:', transform=axL.transAxes, fontsize=8, color='black', va='bottom', ha='right')

        axL.set_ylabel(f'{name} [{unit}]')
        axR.set_ylabel(norm_lbl, color='k', fontsize=9)
        axR.set_ylim(-0.05, 1.05)
        
        all_data = np.concatenate([d for d in th if d is not None and not np.all(np.isnan(d))])
        mx = np.nanmax(np.abs(all_data)); limit = 1.05 * mx if mx !=0 and np.isfinite(mx) else 1.0
        axL.set_ylim(-limit, limit)

    axs[2].set_xlabel('Time [s]')
    
    # Place legend and restrict tight_layout from overlapping into the top 5%
    fig2.legend(handles=handles[:-2], loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(handles)-2, fontsize=9)
    fig2.tight_layout(rect=[0, 0, 1, 0.95])
    fig2.subplots_adjust(top=0.92) # Safety margin

    return fig1, fig2, max_deltas_dict

def plot_rotdnn_psa_psd_fas_results(
    results: Dict[str, Any],
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    PSDreduction: float = 1.0,
    FASreduction: float = 1.0,
    F1PSA: float = 0.2,
    F2PSA: float = 50.0,
    F1Check: float = 0.3,
    F2Check: float = 30.0,
    zi: float = 0.05,
    units: str = 'g',
    plot_directionality: bool = False,
    polar_freqs: List[float] = [0.5, 1, 2, 4, 8, 12, 16, 20]) -> Union[Tuple[plt.Figure, plt.Figure, Dict], Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure, Dict]]:
    
    """
    Generates verification plots and metrics for advanced biaxial RotDnn spectral 
    matching results (PSA, FAS, and PSD domains).

    Methodology:
    - Frequency Domain Comparison (Figure 1): Generates a dynamically sized figure with 
      up to three sub-panels depending on which matching stages were activated.
      - The **PSA panel** compares the initial Scaled RotDnn PSA and the sequentially matched 
        RotDnn PSAs against the Target PSA. A lower sub-plot displays the ratio to the target, 
        highlighting the primary matching domain (`F1PSA` to `F2PSA`) and the target limits.
      - The **FAS panel** (if applicable) compares the smoothed geometric mean FAS of the 
        biaxial records against the smoothed Minimum Target FAS, plotting the threshold 
        (`FASreduction`) used during matching.
      - The **PSD panel** (if applicable) compares the geometric mean of the one-sided PSDs 
        of the biaxial records against the target PSD, plotting the threshold (`PSDreduction`) 
        used during matching. Shaded regions highlight the frequency range (`F1Check` to `F2Check`) 
        used for the FAS and PSD compliance checks.
    - Time-Domain Comparison (Figure 2): Creates a 3x2 grid of subplots displaying the 
      acceleration, velocity, and displacement time histories for both horizontal components 
      (Component 1 on the left, Component 2 on the right). It overlays the scaled seed and 
      all successfully matched stages. On secondary y-axes, it plots the normalized energy 
      buildup (e.g., Normalized CSA/CSV/CSD) for each kinematic domain.
    - Deviation Tracking: Computes the maximum absolute difference between the normalized 
      energy buildups of each matched stage and the original scaled record for both components, 
      printing the maximum deltas directly onto the time-domain plots to track temporal alteration.
    - Directionality (Optional - Figures 3 & 4): If `plot_directionality` is activated, the 
      function solves the biaxial SDOF response at various resonant frequencies. It plots polar 
      representations of the response trajectories (Figure 3) at specific sample frequencies 
      (`polar_freqs`), and constructs the period-dependent RotD100/RotD50 ratios alongside the 
      Directionality Spectrum of Acceleration (DSA) using the internal `dfactor` routine (Figure 4).

    Parameters
    ----------
    results : Dict[str, Any]
        The comprehensive output dictionary generated by the advanced RotDnn matching function 
        (e.g., `generate_rotdnn_psa_psd_fas_compatible_record`), containing frequency/period 
        arrays, spectra, time vectors, and kinematic histories for both components across all stages.
    targetPSAlimits : Tuple[float, float], optional
        The lower and upper tolerance bounds for the PSA matching ratio. Used to plot 
        the limit lines in the ratio sub-panels. Default is (0.9, 1.3).
    PSDreduction : float, optional
        The scaling factor applied to the target PSD during the matching algorithm 
        to define the minimum required threshold. Used for plotting the limit. Default is 1.0.
    FASreduction : float, optional
        The scaling factor applied to the target FAS during the matching algorithm 
        to define the minimum required threshold. Used for plotting the limit. Default is 1.0.
    F1PSA : float, optional
        The minimum frequency (Hz) for the primary PSA matching domain. Default is 0.2.
    F2PSA : float, optional
        The maximum frequency (Hz) for the primary PSA matching domain. Default is 50.0.
    F1Check : float, optional
        The minimum frequency (Hz) evaluated during the FAS and PSD compliance checks. 
        Used to shade the verification region. Default is 0.3.
    F2Check : float, optional
        The maximum frequency (Hz) evaluated during the FAS and PSD compliance checks. 
        Used to shade the verification region. Default is 30.0.
    zi : float, optional
        The damping ratio (as a decimal) associated with the target response spectrum. 
        Used for plot labeling. Default is 0.05.
    units : str, optional
        The string representation of the acceleration units (e.g., 'g', 'm/s²'). 
        Used for plot labeling. Default is 'g'.
    plot_directionality : bool, optional
        If True, executes the necessary SDOF simulations to compute and plot the biaxial 
        polar response trajectories and the Directionality Spectrum of Acceleration (DSA). 
        Default is False.
    polar_freqs : List[float], optional
        A list of specific frequencies (Hz) at which to plot the polar response trajectories 
        in Figure 3 (only used if `plot_directionality` is True). 
        Default is [0.5, 1, 2, 4, 8, 12, 16, 20].

    Returns
    -------
    Union[Tuple[plt.Figure, plt.Figure, Dict[str, list]], Tuple[plt.Figure, plt.Figure, plt.Figure, plt.Figure, Dict[str, list]]]
        If `plot_directionality` is False:
        - fig1 (plt.Figure): The dynamically generated frequency domain plots (PSA, FAS, PSD).
        - fig2 (plt.Figure): The kinematic time histories and normalized energy buildup plots.
        - max_deltas_dict (Dict[str, list]): The maximum absolute deviations in normalized energy 
          buildup for all stages. Keys are `'Acc1'`, `'Acc2'`, `'Vel1'`, `'Vel2'`, `'Disp1'`, and `'Disp2'`.

        If `plot_directionality` is True, returns all of the above, plus:
        - fig3 (plt.Figure): The polar plots of biaxial response trajectories.
        - fig4 (plt.Figure): The RotD100/RotD50 ratios and the Directionality Spectrum of Acceleration.
    """
   
    mpl.rcParams['font.size'] = 9 
    mpl.rcParams['legend.frameon'] = False
    mpl.rcParams['mathtext.fontset'] = 'dejavuserif'
    mpl.rcParams['font.family'] = 'serif'
    
    LINEWIDTH_MAIN = 1.0
    C_TARGET = 'k'; C_SCALED = 'dimgray'; C_PSA = 'cornflowerblue'; C_FAS = 'blueviolet'; C_PSD = 'salmon'
    COLOR_SHADE = 'steelblue'; ALPHA_SHADE = 0.1

    freqs = results['freqs']; t = results['t']; nn = results.get('nn', 50)
    sf = results.get('scale_factor', 1.0) 
    
    do_fas = not np.all(np.isnan(results.get('target_fas', [np.nan])))
    do_psd = not np.all(np.isnan(results.get('target_psd', [np.nan])))

    # =========================================================================
    # FIGURE 1: SPECTRA
    # =========================================================================
    plot_configs = [('PSA', results['target_psa'], targetPSAlimits, (F1PSA, F2PSA), f'RotD{nn} PSA ({units})', False)]
    if do_fas: plot_configs.append(('FAS', results['target_fas'], (FASreduction, np.nan), (F1Check, F2Check), f'RotD{nn} FAS ({units}-s)', True))
    if do_psd: plot_configs.append(('PSD', results['target_psd'], (PSDreduction, np.nan), (F1Check, F2Check), fr'RotD{nn} PSD ({units}$^2$/Hz)', True))

    num_cols = len(plot_configs)
    fig1 = plt.figure(figsize=(2.5 * num_cols, 4))
    gs = GridSpec(2, num_cols, figure=fig1, height_ratios=[7, 3], hspace=0.12, wspace=0.45)

    active_keys = ['sc']; active_colors = [C_PSA]; active_labels = ['PSA matched']
    if do_fas: active_keys.append('sc_fas'); active_colors.append(C_FAS); active_labels.append('FAS adj')
    if do_psd: active_keys.append('sca'); active_colors.append(C_PSD); active_labels.append('PSD adj')

    for i, (spec_type, target, lims, mask_range, ylabel, is_loglog) in enumerate(plot_configs):
        ax_top = fig1.add_subplot(gs[0, i]); ax_bot = fig1.add_subplot(gs[1, i])
        ax_top.axvspan(mask_range[0], mask_range[1], color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)
        ax_bot.axvspan(mask_range[0], mask_range[1], color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

        plot_cmd_top = ax_top.loglog if is_loglog else ax_top.semilogx
        plot_cmd_bot = ax_bot.semilogx
        plot_cmd_top(freqs, target, color=C_TARGET, lw=LINEWIDTH_MAIN*2, zorder=5)

        if not np.isnan(lims[0]):
            plot_cmd_top(freqs, target * lims[0], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
            ax_bot.axhline(lims[0], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)
        if not np.isnan(lims[1]):
            plot_cmd_top(freqs, target * lims[1], color=C_TARGET, ls='--', lw=LINEWIDTH_MAIN, zorder=5)
            ax_bot.axhline(lims[1], color='k', ls='--', lw=LINEWIDTH_MAIN, zorder=5)

        p_data_s = results[f"{spec_type.lower()}_s"]
        plot_cmd_top(freqs, p_data_s, color=C_SCALED, lw=LINEWIDTH_MAIN)
        mask = (freqs >= mask_range[0]) & (freqs <= mask_range[1])
        plot_cmd_bot(freqs, np.where(mask, p_data_s/target, np.nan), color=C_SCALED, lw=LINEWIDTH_MAIN)
        
        for key, color in zip(active_keys, active_colors):
            p_data = results[f"{spec_type.lower()}_{key}"]
            plot_cmd_top(freqs, p_data, color=color, lw=LINEWIDTH_MAIN)
            plot_cmd_bot(freqs, np.where(mask, p_data/target, np.nan), color=color, lw=LINEWIDTH_MAIN)
            
        ax_top.set_ylabel(ylabel); ax_top.set_xlim(0.1, 100); ax_top.set_xticklabels([])
        mask_plot = (freqs >= 0.1) & (freqs <= 100)
        if is_loglog:
            y_min = np.nanmin((target * (lims[0] if not np.isnan(lims[0]) else 1.0))[mask_plot]) * 0.7
            y_max = np.nanmax(target[mask_plot]) * 1.5
            ax_top.set_ylim(bottom=y_min, top=y_max)
        else:
            y_max = np.nanmax((target * (lims[1] if not np.isnan(lims[1]) else 1.0))[mask_plot]) * 1.1
            ax_top.set_ylim(bottom=-0.015, top=y_max)
        
        ax_bot.set_xlim(0.1, 100); ax_bot.set_ylim(0.5, 1.5); ax_bot.set_ylabel('Ratio'); ax_bot.set_xlabel('Frequency (Hz)')
        ax_bot.axhline(1.0, color='k', lw=LINEWIDTH_MAIN, zorder=5)

    handles = [Line2D([0],[0], color=C_SCALED, lw=LINEWIDTH_MAIN, label='Scaled')]
    for c, l in zip(active_colors, active_labels): handles.append(Line2D([0],[0], color=c, lw=LINEWIDTH_MAIN, label=l))
    handles.extend([Line2D([0],[0], color=C_TARGET, lw=LINEWIDTH_MAIN*2, label='Target'), Line2D([0],[0], color=C_TARGET, lw=LINEWIDTH_MAIN, ls='--', label='Limits')])
    fig1.legend(handles=handles, loc='upper center', ncol=len(handles), bbox_to_anchor=(0.5, 0.99), frameon=False, columnspacing=1.0)
    fig1.subplots_adjust(top=0.91, bottom=0.12, left=0.12, right=0.97)

    # =========================================================================
    # FIGURE 2: TIME HISTORIES
    # =========================================================================
    if units == 'g': conv_vel = 980.665; conv_disp = 980.665; u_vel = 'cm/s'; u_disp = 'cm'
    else: conv_vel = 1.0; conv_disp = 1.0; u_vel = f'{units}-s'; u_disp = f'{units}-s^2'

    fig2, axs = plt.subplots(3, 2, figsize=(7.5, 6), sharex=True)
    trace_colors = [C_SCALED] + active_colors
    max_deltas_dict = {}
    
    for comp in [1, 2]:
        th_list = [results[f's{comp}_scaled']]; v_list = [results[f'vel{comp}_s'] * conv_vel]; d_list = [results[f'disp{comp}_s'] * conv_disp]
        nai_list = [results[f'ai{comp}_s']]; ncsv_list = [results[f'csv{comp}_s']]; ncsd_list = [results[f'csd{comp}_s']]

        for key in active_keys:
            th_key = f'sc{comp}' if key == 'sc' else (f'sc{comp}_fas' if key == 'sc_fas' else f'sca{comp}')
            th_list.append(results[th_key]); v_list.append(results[f"vel{comp}_{key}"] * conv_vel); d_list.append(results[f"disp{comp}_{key}"] * conv_disp)
            nai_list.append(results[f"ai{comp}_{key}"]); ncsv_list.append(results[f"csv{comp}_{key}"]); ncsd_list.append(results[f"csd{comp}_{key}"])

        groups = [(th_list, nai_list, 'Acc', units, 'Norm. CSA'), (v_list, ncsv_list, 'Vel', u_vel, 'Norm. CSV'), (d_list, ncsd_list, 'Disp', u_disp, 'Norm. CSD')]
        y_pos = 0.05; x_anchor = 0.98; col = comp - 1

        for row, (th, norm, name, unit, norm_lbl) in enumerate(groups):
            axL = axs[row, col]; axR = axL.twinx()
            for d, c in zip(th, trace_colors): 
                if d is not None and not np.all(np.isnan(d)): axL.plot(t, d, color=c, lw=0.5, alpha=0.6)
            for n, c in zip(norm, trace_colors): 
                if n is not None and not np.all(n == 0.0): axR.plot(t, n, color=c, lw=1.0)
            
            deltas = [np.max(np.abs(n - norm[0])) for n in norm[1:]]
            max_deltas_dict[f"{name}{comp}"] = deltas
            
            if not np.all(norm[0] == 0.0):
                offset_val = 0.00
                step = 0.12 
                for i, delta in enumerate(reversed(deltas)):
                    c_idx = len(deltas) - i 
                    comma = ", " if i > 0 else ""
                    axL.text(x_anchor - offset_val, y_pos, f"{delta:.2f}{comma}", transform=axL.transAxes, fontsize=8, color=trace_colors[c_idx], va='bottom', ha='right', fontweight='bold')
                    offset_val += step
                axL.text(x_anchor - offset_val, y_pos, r'max $\Delta$:', transform=axL.transAxes, fontsize=8, color='black', va='bottom', ha='right')

            axL.set_ylabel(f'{name} [{unit}]' if col == 0 else ''); axR.set_ylabel(norm_lbl if col == 1 else '', color='k', fontsize=9); axR.set_ylim(-0.05, 1.05)
            all_data = np.concatenate([d for d in th if d is not None and not np.all(np.isnan(d))]); mx = np.nanmax(np.abs(all_data)); limit = 1.05 * mx if mx !=0 and np.isfinite(mx) else 1.0
            axL.set_ylim(-limit, limit)
            if col == 0: axR.set_yticks([])
            if col == 1: axL.set_yticks([])

    axs[2, 0].set_xlabel('Time [s]'); axs[2, 1].set_xlabel('Time [s]')
    axs[0, 0].set_title("Component 1", fontsize=9, fontweight='bold')
    axs[0, 1].set_title("Component 2", fontsize=9, fontweight='bold')
    
    # Restrict tight_layout from using the top 6% of the figure space
    fig2.tight_layout(rect=[0, 0, 1, 0.94])
    
    # Place legend safely in the reserved top space
    fig2.legend(handles=handles[:-2], loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=len(handles)-2, fontsize=9)

    if not plot_directionality: return fig1, fig2, max_deltas_dict

    # =========================================================================
    # FIGURE 3: POLAR SPECTRA
    # =========================================================================
    periods = results['periods']
    target_Ts = [1.0 / f for f in polar_freqs]
    indices = [np.argmin(np.abs(periods - pt)) for pt in target_Ts]
    actual_freqs = [1.0 / periods[i] for i in indices]
    
    psa_180_list = [results.get('psa_180_seed', np.zeros((1,len(periods)))) * sf]
    for key in active_keys:
        p180_key = 'psa_180_fas' if key == 'sc_fas' else ('psa_180_final' if key == 'sca' else f'psa_180_{key}')
        psa_180_list.append(results.get(p180_key, np.zeros((1,len(periods)))))
        
    n_plots = len(polar_freqs)
    ncols = 4
    nrows = int(np.ceil(n_plots / ncols))
    fig_height = 1.8 * nrows 
    
    fig3, axes = plt.subplots(nrows, ncols, subplot_kw={'projection': 'polar'}, figsize=(7.5, fig_height))
    axes_flat = axes.flatten() if n_plots > 1 else [axes]
    theta_rad = np.linspace(0, 2*np.pi, 360, endpoint=False)
    
    for i, ax in enumerate(axes_flat):
        if i < n_plots:
            pidx = indices[i]; f_val = actual_freqs[i]
            for mat, lbl, col in zip(psa_180_list, ['Scaled'] + active_labels, trace_colors):
                if mat is not None and mat.size > 1:
                    data = mat[:, pidx]; mx = np.max(data)
                    if mx > 0: data = data / mx
                    data_wrap = np.concatenate([data, data]); l_str = lbl if i == (n_plots - 1) else ""
                    ax.plot(theta_rad, data_wrap, color=col, linewidth=1.0, label=l_str)
            ax.set_title(f"f = {f_val:.0f} Hz", fontsize=8, pad=2); ax.set_xticks([]); ax.set_yticks([0.5, 1.0]); ax.set_yticklabels([]); ax.set_ylim(0, 1.1)
            ax.grid(True, linestyle=':', alpha=0.5); ax.set_theta_zero_location("E")
        else: ax.axis('off')
    
    fig3.legend(loc='lower center', ncol=len(trace_colors), bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False, columnspacing=1.0)
    plt.tight_layout(rect=(0, 0.06, 1, 1.0))

    # =========================================================================
    # FIGURE 4: DIRECTIONALITY ANALYSIS
    # =========================================================================
    fig4, (ax_ratio, ax_dsa) = plt.subplots(1, 2, figsize=(7.5, 4.0))
    plot_freqs = 1.0 / periods
    
    def calc_dsa_curve_fd(a1, a2, dt_rec, T_vec, damping):
        if a1 is None or a2 is None: return np.full_like(T_vec, np.nan)
        npts = len(a1); N = 2**int(np.ceil(np.log2(npts)))
        A1_w = np.fft.rfft(a1, n=N); A2_w = np.fft.rfft(a2, n=N)
        freqs_fft = np.fft.rfftfreq(N, d=dt_rec); omega = 2 * np.pi * freqs_fft
        dsa_list = []
        for T in T_vec:
            wn = 2 * np.pi / T
            with np.errstate(divide='ignore', invalid='ignore'): H = (wn**2 + 2j*damping*wn*omega) / (wn**2 - omega**2 + 2j*damping*wn*omega)
            H[0] = 1.0; R1_w = A1_w * H; R2_w = A2_w * H
            r1 = np.fft.irfft(R1_w, n=N)[:npts]; r2 = np.fft.irfft(R2_w, n=N)[:npts]
            d_out = dfactor(r1, r2, plot=0); dsa_list.append(d_out[4]) 
        return np.array(dsa_list)

    data_groups = [(results.get('psa_180_seed', np.zeros((1,len(periods))))*sf, results['s1_scaled'], results['s2_scaled'], C_SCALED, 'Scaled')]
    for key, color, label in zip(active_keys, active_colors, active_labels):
        p180_key = 'psa_180_fas' if key == 'sc_fas' else ('psa_180_final' if key == 'sca' else f'psa_180_{key}')
        if key == 'sc': th1_key, th2_key = 'sc1', 'sc2'
        elif key == 'sc_fas': th1_key, th2_key = 'sc1_fas', 'sc2_fas'
        elif key == 'sca': th1_key, th2_key = 'sca1', 'sca2'
        data_groups.append((results.get(p180_key), results.get(th1_key), results.get(th2_key), color, label))

    for (mat, acc_x, acc_y, col, lbl) in data_groups:
        if mat is not None and mat.size > 1:
            rotd100 = np.max(mat, axis=0); rotd50 = np.percentile(mat, 50, axis=0)
            ratio_curve = rotd100 / rotd50
            ax_ratio.semilogx(plot_freqs, ratio_curve, color=col, lw=1.0, label=lbl)
            dsa_curve = calc_dsa_curve_fd(acc_x, acc_y, results['dt'], periods, zi)
            ax_dsa.semilogx(plot_freqs, dsa_curve, color=col, lw=1.0, label=lbl)

    ax_ratio.set_xlabel('Frequency (Hz)'); ax_ratio.set_ylabel('RotD100 / RotD50'); ax_ratio.set_xlim(0.1, 100)
    ax_ratio.grid(True, which='both', linestyle=':', alpha=0.5); ax_ratio.axvspan(F1PSA, F2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

    ax_dsa.set_xlabel('Frequency (Hz)'); ax_dsa.set_ylabel('Directionality Factor'); ax_dsa.set_xlim(0.1, 100)
    ax_dsa.grid(True, which='both', linestyle=':', alpha=0.5); ax_dsa.axvspan(F1PSA, F2PSA, color=COLOR_SHADE, alpha=ALPHA_SHADE, zorder=0)

    legend_handles_4 = [Line2D([0], [0], color=C_SCALED, lw=1.5, label='Scaled')]
    for c, l in zip(active_colors, active_labels): legend_handles_4.append(Line2D([0], [0], color=c, lw=1.5, label=l))
    fig4.legend(handles=legend_handles_4, loc='lower center', ncol=len(legend_handles_4), bbox_to_anchor=(0.5, 0.0), fontsize=8, frameon=False, columnspacing=1.0)
    plt.tight_layout(rect=(0, 0.1, 1, 1.0))

    return fig1, fig2, fig3, fig4, max_deltas_dict

def plot_fas_psd_comparison(
    # Raw FAS data
    freqs_fas_raw: np.ndarray,
    fas_orig_raw: np.ndarray,
    fas_scaled_raw: np.ndarray,
    fas_matched_raw: np.ndarray,
    # Raw PSD data
    freqs_psd_raw: np.ndarray,
    psd_orig_raw: np.ndarray,
    psd_scaled_raw: np.ndarray,
    psd_matched_raw: np.ndarray,
    # Smoothed FAS data
    output_freq_vector: np.ndarray,
    fas_orig_smooth: np.ndarray,
    fas_scaled_smooth: np.ndarray,
    fas_matched_smooth: np.ndarray,
    # Smoothed PSD data
    psd_orig_smooth: np.ndarray,
    psd_scaled_smooth: np.ndarray,
    psd_matched_smooth: np.ndarray,
    # Plotting options
    title_suffix: str = ""
) -> Tuple[plt.Figure, plt.Figure]:
    """
    Plots a comparison of FAS and PSD for three pre-calculated records.

    This function is a helper to visualize the "Original Seed," "Scaled Seed,"
    and "Matched" records, showing both their raw, spiky spectra and their
    smoothed equivalents. It only performs plotting; all data must be
    calculated beforehand.

    Parameters
    ----------
    freqs_fas_raw : np.ndarray
        The x-axis (frequency) for the raw FAS data.
    fas_orig_raw : np.ndarray
        The raw FAS for the original record.
    fas_scaled_raw : np.ndarray
        The raw FAS for the scaled record.
    fas_matched_raw : np.ndarray
        The raw FAS for the matched record.
    freqs_psd_raw : np.ndarray
        The x-axis (frequency) for the raw PSD data.
    psd_orig_raw : np.ndarray
        The raw PSD for the original record.
    psd_scaled_raw : np.ndarray
        The raw PSD for the scaled record.
    psd_matched_raw : np.ndarray
        The raw PSD for the matched record.
    output_freq_vector : np.ndarray
        The x-axis (frequency) for all smoothed data.
    fas_orig_smooth : np.ndarray
        The smoothed FAS for the original record.
    fas_scaled_smooth : np.ndarray
        The smoothed FAS for the scaled record.
    fas_matched_smooth : np.ndarray
        The smoothed FAS for the matched record.
    psd_orig_smooth : np.ndarray
        The smoothed PSD for the original record.
    psd_scaled_smooth : np.ndarray
        The smoothed PSD for the scaled record.
    psd_matched_smooth : np.ndarray
        The smoothed PSD for the matched record.
    title_suffix : str, optional
        A string to append to the plot titles (e.g., "Component 1").

    Returns
    -------
    fig_fas : plt.Figure
        The matplotlib Figure object for the FAS comparison plot.
    fig_psd : plt.Figure
        The matplotlib Figure object for the PSD comparison plot.
    """
    
    if title_suffix:
        title_suffix = f" - {title_suffix}"

    # --- 1. Plot FAS Comparison (Raw vs. Smooth) ---
    fig_fas, ax_fas = plt.subplots(figsize=(6.5, 5))

    ax_fas.loglog(freqs_fas_raw, fas_orig_raw, lw=0.5, color='blueviolet', alpha=0.5)
    ax_fas.loglog(freqs_fas_raw, fas_scaled_raw, lw=0.5, color='dimgray', alpha=0.5)
    ax_fas.loglog(freqs_fas_raw, fas_matched_raw, lw=0.5, color='salmon', alpha=0.5)

    ax_fas.loglog(output_freq_vector, fas_orig_smooth, lw=1.5, color='blueviolet', label='Original Seed (Smooth)')
    ax_fas.loglog(output_freq_vector, fas_scaled_smooth, lw=1.5, color='dimgray', label='Scaled Seed (Smooth)')
    ax_fas.loglog(output_freq_vector, fas_matched_smooth, lw=1.5, color='salmon', label='Matched (Smooth)')

    ax_fas.set_xlabel('Frequency (Hz)')
    ax_fas.set_ylabel('Fourier Amplitude (g-s)')
    ax_fas.set_title(f'Fourier Amplitude Spectrum Comparison{title_suffix}')
    ax_fas.legend()
    ax_fas.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_fas.set_xlim(output_freq_vector.min(), output_freq_vector.max())

    try:
        all_fas_data = np.concatenate([
            fas_orig_smooth[2:-2],
            fas_scaled_smooth[2:-2],
            fas_matched_smooth[2:-2]
        ])
        min_val = np.nanmin(all_fas_data[all_fas_data > 0])
        ax_fas.set_ylim(bottom=min_val * 0.5)
    except Exception:
        pass # Failsafe

    fig_fas.tight_layout()

    # --- 2. Plot PSD Comparison (Raw vs. Smooth) ---
    fig_psd, ax_psd = plt.subplots(figsize=(6.5, 5))

    ax_psd.loglog(freqs_psd_raw, psd_orig_raw, lw=0.5, color='blueviolet', alpha=0.5)
    ax_psd.loglog(freqs_psd_raw, psd_scaled_raw, lw=0.5, color='dimgray', alpha=0.5)
    ax_psd.loglog(freqs_psd_raw, psd_matched_raw, lw=0.5, color='salmon', alpha=0.5)

    ax_psd.loglog(output_freq_vector, psd_orig_smooth, lw=1.5, color='blueviolet', label='Original Seed (Smooth)')
    ax_psd.loglog(output_freq_vector, psd_scaled_smooth, lw=1.5, color='dimgray', label='Scaled Seed (Smooth)')
    ax_psd.loglog(output_freq_vector, psd_matched_smooth, lw=1.5, color='salmon', label='Matched (Smooth)')

    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_ylabel('Power Spectral Density (g²-s)')
    ax_psd.set_title(f'Power Spectral Density Comparison{title_suffix}')
    ax_psd.legend()
    ax_psd.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_psd.set_xlim(output_freq_vector.min(), output_freq_vector.max())

    try:
        all_psd_data = np.concatenate([
            psd_orig_smooth[2:-2],
            psd_scaled_smooth[2:-2],
            psd_matched_smooth[2:-2]
        ])
        min_val = np.nanmin(all_psd_data[all_psd_data > 0])
        ax_psd.set_ylim(bottom=min_val * 0.5)
    except Exception:
        pass # Failsafe

    fig_psd.tight_layout()
    
    return fig_fas, fig_psd

def plot_rotdnn_fas_psd_comparison(
    output_freq_vector: np.ndarray,
    nn: int,
    fas_rotd_raw_data: Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]],
    fas_rotd_smooth_data: Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]],
    psd_rotd_raw_data: Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]],
    psd_rotd_smooth_data: Tuple[Dict[int, np.ndarray], Dict[int, np.ndarray], Dict[int, np.ndarray]],
    title_suffix: str = ""
) -> plt.Figure:
    """
    Plots a comparison of RotDnn FAS and PSD for three record sets.

    This helper plots "Original Seed", "Scaled Seed", and "Matched" records,
    showing both raw percentile and smoothed ("smooth last") percentile spectra.
    All data must be pre-calculated.

    Parameters
    ----------
    output_freq_vector : np.ndarray
        The x-axis (frequency) for all smoothed data.
    nn : int
        The percentile (e.g., 100) to use for plotting and labels.
    fas_rotd_raw_data : tuple
        A tuple of 3 dicts: (raw_orig, raw_scaled, raw_matched) for RotDnn FAS.
    fas_rotd_smooth_data : tuple
        A tuple of 3 dicts: (smooth_orig, smooth_scaled, smooth_matched) for RotDnn FAS.
    psd_rotd_raw_data : tuple
        A tuple of 3 dicts: (raw_orig, raw_scaled, raw_matched) for RotDnn PSD.
    psd_rotd_smooth_data : tuple
        A tuple of 3 dicts: (smooth_orig, smooth_scaled, smooth_matched) for RotDnn PSD.
    title_suffix : str, optional
        A string to append to the plot titles.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object for the RotDnn comparison plots.
    """
    
    fig_rotd, (ax_fas, ax_psd) = plt.subplots(2, 1, figsize=(6.5, 9))
    if title_suffix:
        fig_rotd.suptitle(title_suffix)

    colors = ['blueviolet', 'dimgray', 'cornflowerblue']
    labels = ['Original Seed', 'Scaled Seed', 'Matched']
    
    # --- Plot RotDnn FAS ---
    for i in range(3):
        # Plot RAW data (thin, semi-transparent)
        ax_fas.loglog(output_freq_vector, fas_rotd_raw_data[i][nn], lw=0.5, color=colors[i], alpha=0.5)
        # Plot SMOOTHED data (thicker, solid, with label)
        ax_fas.loglog(output_freq_vector, fas_rotd_smooth_data[i][nn], lw=1.5, color=colors[i], label=f'{labels[i]} (Smooth)')

    ax_fas.set_xlabel('Frequency (Hz)')
    ax_fas.set_ylabel(f'RotD{nn} FAS (g-s)')
    ax_fas.set_title(f'RotD{nn} Fourier Amplitude Spectrum')
    ax_fas.legend()
    ax_fas.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_fas.set_xlim(output_freq_vector.min(), output_freq_vector.max())

    try:
        all_fas_data = np.concatenate([
            fas_rotd_smooth_data[0][nn][2:-2],
            fas_rotd_smooth_data[1][nn][2:-2],
            fas_rotd_smooth_data[2][nn][2:-2]
        ])
        min_val = np.nanmin(all_fas_data[all_fas_data > 0])
        ax_fas.set_ylim(bottom=min_val * 0.5)
    except Exception:
        pass # Failsafe

    # --- Plot RotDnn PSD ---
    for i in range(3):
        # Plot RAW data (thin, semi-transparent)
        ax_psd.loglog(output_freq_vector, psd_rotd_raw_data[i][nn], lw=0.5, color=colors[i], alpha=0.5)
        # Plot SMOOTHED data (thicker, solid, with label)
        ax_psd.loglog(output_freq_vector, psd_rotd_smooth_data[i][nn], lw=1.5, color=colors[i], label=f'{labels[i]} (Smooth)')
        
    ax_psd.set_xlabel('Frequency (Hz)')
    ax_psd.set_ylabel(f'RotD{nn} PSD (g²-s)')
    ax_psd.set_title(f'RotD{nn} Power Spectral Density')
    ax_psd.legend()
    ax_psd.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_psd.set_xlim(output_freq_vector.min(), output_freq_vector.max())

    try:
        all_psd_data = np.concatenate([
            psd_rotd_smooth_data[0][nn][2:-2],
            psd_rotd_smooth_data[1][nn][2:-2],
            psd_rotd_smooth_data[2][nn][2:-2]
        ])
        min_val = np.nanmin(all_psd_data[all_psd_data > 0])
        ax_psd.set_ylim(bottom=min_val * 0.5)
    except Exception:
        pass # Failsafe

    fig_rotd.tight_layout(rect=(0, 0, 1, 0.96))
    return fig_rotd

def plot_effective_fas_psd_comparison(
    output_freq_vector: np.ndarray,
    eas_raw_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    eas_smooth_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    epsd_raw_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    epsd_smooth_data: Tuple[np.ndarray, np.ndarray, np.ndarray],
    title_suffix: str = ""
) -> plt.Figure:
    """
    Plots a comparison of Effective FAS (EAS) and PSD (EPSD) for three record sets.

    This helper plots "Original Seed", "Scaled Seed", and "Matched" records,
    showing both raw and smoothed ("smooth last") spectra.
    All data must be pre-calculated.

    Parameters
    ----------
    output_freq_vector : np.ndarray
        The x-axis (frequency) for all smoothed data.
    eas_raw_data : tuple
        A tuple of 3 arrays: (raw_orig, raw_scaled, raw_matched) for EAS.
    eas_smooth_data : tuple
        A tuple of 3 arrays: (smooth_orig, smooth_scaled, smooth_matched) for EAS.
    epsd_raw_data : tuple
        A tuple of 3 arrays: (raw_orig, raw_scaled, raw_matched) for EPSD.
    epsd_smooth_data : tuple
        A tuple of 3 arrays: (smooth_orig, smooth_scaled, smooth_matched) for EPSD.
    title_suffix : str, optional
        A string to append to the plot titles.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object for the Effective spectra comparison plots.
    """
    
    fig_eff, (ax_eas, ax_epsd) = plt.subplots(2, 1, figsize=(6.5, 9))
    if title_suffix:
        fig_eff.suptitle(title_suffix)
        
    colors = ['blueviolet', 'dimgray', 'cornflowerblue']
    labels = ['Original Seed', 'Scaled Seed', 'Matched']

    # --- Plot Effective FAS (EAS) ---
    for i in range(3):
        # Plot RAW data (thin, semi-transparent)
        ax_eas.loglog(output_freq_vector, eas_raw_data[i], lw=0.5, color=colors[i], alpha=0.5)
        # Plot SMOOTHED data (thicker, solid, with label)
        ax_eas.loglog(output_freq_vector, eas_smooth_data[i], lw=1.5, color=colors[i], label=f'{labels[i]} (Smooth)')

    ax_eas.set_xlabel('Frequency (Hz)')
    ax_eas.set_ylabel('EAS (g-s)')
    ax_eas.set_title('Effective Amplitude Spectrum (EAS)')
    ax_eas.legend()
    ax_eas.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_eas.set_xlim(output_freq_vector.min(), output_freq_vector.max())

    try:
        all_eas_data = np.concatenate([
            eas_smooth_data[0][2:-2],
            eas_smooth_data[1][2:-2],
            eas_smooth_data[2][2:-2]
        ])
        min_val = np.nanmin(all_eas_data[all_eas_data > 0])
        ax_eas.set_ylim(bottom=min_val * 0.5)
    except Exception:
        pass # Failsafe

    # --- Plot Effective PSD (EPSD) ---
    for i in range(3):
        # Plot RAW data (thin, semi-transparent)
        ax_epsd.loglog(output_freq_vector, epsd_raw_data[i], lw=0.5, color=colors[i], alpha=0.5)
        # Plot SMOOTHED data (thicker, solid, with label)
        ax_epsd.loglog(output_freq_vector, epsd_smooth_data[i], lw=1.5, color=colors[i], label=f'{labels[i]} (Smooth)')

    ax_epsd.set_xlabel('Frequency (Hz)')
    ax_epsd.set_ylabel('EPSD (g²-s)')
    ax_epsd.set_title('Effective Power Spectrum (EPSD)')
    ax_epsd.legend()
    ax_epsd.grid(True, which='both', linestyle=':', alpha=0.7)
    ax_epsd.set_xlim(output_freq_vector.min(), output_freq_vector.max())

    try:
        all_epsd_data = np.concatenate([
            epsd_smooth_data[0][2:-2],
            epsd_smooth_data[1][2:-2],
            epsd_smooth_data[2][2:-2]
        ])
        min_val = np.nanmin(all_epsd_data[all_epsd_data > 0])
        ax_epsd.set_ylim(bottom=min_val * 0.5)
    except Exception:
        pass # Failsafe

    fig_eff.tight_layout(rect=(0, 0, 1, 0.96))
    return fig_eff

def save_generation_spectral_outputs(
        results: Dict[str, Any], 
        prefix: str):
    """
    Exports the final matched spectral data (PSA, PSD, and optionally FAS) 
    to comma-separated values (CSV) files for external analysis or reporting.

    The function dynamically checks the contents of the `results` 
      dictionary to determine which matching stages were active. It will always export 
      the Pseudo-Spectral Acceleration (PSA) and Power Spectral Density (PSD) arrays. 
      If the Fourier Amplitude Spectrum (FAS) arrays are present (indicating a PSA+FAS+PSD 
      run rather than just PSA+PSD), it will export those as well.

    Parameters
    ----------
    results : Dict[str, Any]
        The output dictionary from a single-component or biaxial matching run. Must 
        contain at least 'freqs', 'periods', 'target_psa', 'target_psd', and the 
        final matched representations of those spectra.
    prefix : str
        The base string used to name the output files. The function will append 
        specific suffixes (e.g., '_PSA.csv', '_PSD.csv') to this prefix.

    Returns
    -------
    None
        The function writes directly to the disk and does not return any objects.
    """

    freqs = results['freqs']
    periods = results['periods']
    
    # Save PSA
    if 'psa_sca' in results:
        psa_final = results['psa_sca']
    elif 'psa_rotd_final' in results:
        psa_final = results['psa_rotd_final']
    else:
        psa_final = results.get('psa_fin', np.zeros_like(periods))
        
    psa_data = np.column_stack([periods, results['target_psa'], psa_final])
    np.savetxt(f"{prefix}_PSA.csv", psa_data, delimiter=',', 
               header='Period,Target,Matched', comments='')
    
    # Save PSD
    if 'psd_sca' in results:
        psd_final = results['psd_sca']
    elif 'psd_rotd_final' in results:
        psd_final = results['psd_rotd_final']
    else:
        psd_final = results.get('psd_fin', np.zeros_like(freqs))
        
    psd_data = np.column_stack([freqs, results['target_psd'], psd_final])
    np.savetxt(f"{prefix}_PSD.csv", psd_data, delimiter=',', 
               header='Freq,Target,Matched', comments='')
               
    # Save FAS (Conditional: Only if FAS keys exist in results)
    fas_target = results.get('target_fas')
    if fas_target is not None:
        if 'fas_sca' in results:
            fas_final = results['fas_sca']
        elif 'fas_rotd_final' in results:
            fas_final = results['fas_rotd_final']
        else:
            fas_final = results.get('fas_fin', np.zeros_like(freqs))
            
        fas_data = np.column_stack([freqs, fas_target, fas_final])
        np.savetxt(f"{prefix}_FAS.csv", fas_data, delimiter=',', 
                   header='Freq,Target,Matched', comments='')
        log.info(f"Spectral outputs (PSA, PSD, FAS) saved with prefix: {prefix}")
    else:
        log.info(f"Spectral outputs (PSA, PSD) saved with prefix: {prefix}")
        
def save_results_as_at2(
    results: Dict[str, Any],
    filepath: str,
    comp_key: str = 'ccs',
    header_details: Optional[Dict[str, str]] = None) -> None:
    
    """Saves a matched acceleration time series in PEER .AT2 format.

    Parameters
    ----------
    results : Dict[str, Any]
        The results dictionary from a REQPY function (e.g., REQPY_single).
        Must contain 'ccs' (or other comp_key) and 'dt'.
    filepath : str
        The full path (including extension, e.g., "my_record.AT2")
        to save the file.
    comp_key : str, optional
        The key in the results dictionary for the acceleration array
        (e.g., 'ccs' for REQPY_single, 'scc1' for REQPYrotdnn).
        Default is 'ccs'.
    header_details : Optional[Dict[str, str]], optional
        A dictionary providing details for the .AT2 header.
        Keys: 'title', 'date', 'station', 'component'.
        If None, generic defaults are used.
    """
    accel = results.get(comp_key)
    dt = results.get('dt')

    if accel is None or dt is None:
        msg = f"Cannot save .AT2 file: '{comp_key}' or 'dt' not found in results dictionary."
        log.error(msg)
        raise KeyError(msg)

    npts = len(accel)
    
    # Fill header details with defaults if not provided
    if header_details is None:
        header_details = {}
    
    title = header_details.get('title', 'REQPY SPECTRALLY MATCHED RECORD')
    date = header_details.get('date', '01/01/2025')
    station = header_details.get('station', 'REQPY_STATION')
    component = header_details.get('component', f'Matched {comp_key}')

    header_line1 = f"{title}\n"
    header_line2 = f"EARTHQUAKE, {date}, {station}, {component}\n"
    header_line3 = "ACCELERATION IN G\n"
    header_line4 = f"NPTS= {npts}, DT= {dt:.8f} SEC\n"

    try:
        with open(filepath, 'w') as f:
            f.write(header_line1)
            f.write(header_line2)
            f.write(header_line3)
            f.write(header_line4)
            
            # Write data, 8 columns per line
            for i in range(npts):
                f.write(f" {accel[i]: 15.7e}")
                if (i + 1) % 8 == 0 and i != (npts - 1): # Add newline every 8 points
                    f.write("\n")
            f.write("\n") # Final newline
        log.info(f"Successfully saved .AT2 file to: {filepath}")
    except Exception as e:
        log.error(f"Error saving .AT2 file: {e}")

def save_results_as_2col(
    results: Dict[str, Any],
    filepath: str,
    comp_key: str = 'ccs',
    header_str: Optional[str] = None) -> None:
    
    """Saves a matched time series as a 2-column (Time, Value) text file.

    Parameters
    ----------
    results : Dict[str, Any]
        The results dictionary from a REQPY function.
        Must contain 'dt' and the specified `comp_key`.
    filepath : str
        The full path to save the file.
    comp_key : str, optional
        The key in the results dictionary for the data array
        (e.g., 'ccs', 'cvel', 'cdisp'). Default is 'ccs'.
    header_str : Optional[str], optional
        A string to write as the header. If None, a default
        header is generated.
    """
    data = results.get(comp_key)
    dt = results.get('dt')

    if data is None or dt is None:
        msg = f"Cannot save as 2-col: '{comp_key}' or 'dt' not found in results dictionary."
        log.error(msg)
        raise KeyError(msg)

    npts = len(data)
    t = np.linspace(0, (npts - 1) * dt, npts)
    
    # Stack time and data as columns
    data_to_save = np.stack((t, data), axis=1)

    # Create default header if none provided
    if header_str is None:
        header_str = (f"REQPY Matched Time Series\n"
                      f"Data key: '{comp_key}'\n"
                      f"Time Step (dt): {dt:.8f} s\n"
                      f"Time (s), Value (units vary)")

    try:
        np.savetxt(filepath, data_to_save, header=header_str, fmt='%.8e', delimiter=',')
        log.info(f"Successfully saved 2-column file to: {filepath}")
    except Exception as e:
        log.error(f"Error saving 2-column file: {e}")

def save_results_as_1col(
    results: Dict[str, Any],
    filepath: str,
    comp_key: str = 'ccs',
    header_str: Optional[str] = None) -> None:
    
    """Saves a matched time series as a single-column (Value) text file.

    Parameters
    ----------
    results : Dict[str, Any]
        The results dictionary from a REQPY function.
        Must contain 'dt' and the specified `comp_key`.
    filepath : str
        The full path to save the file.
    comp_key : str, optional
        The key in the results dictionary for the data array
        (e.g., 'ccs', 'cvel', 'cdisp'). Default is 'ccs'.
    header_str : Optional[str], optional
        A string to write as the header. If None, a default
        header is generated.
    """
    data = results.get(comp_key)
    dt = results.get('dt')

    if data is None or dt is None:
        msg = f"Cannot save as 1-col: '{comp_key}' or 'dt' not found in results dictionary."
        log.error(msg)
        raise KeyError(msg)

    # Create default header if none provided
    if header_str is None:
        header_str = (f"REQPY Matched Time Series\n"
                      f"Data key: '{comp_key}'\n"
                      f"Time Step (dt): {dt:.8f} s\n"
                      f"Data points follow:")

    try:
        np.savetxt(filepath, data, header=header_str, fmt='%.8e')
        log.info(f"Successfully saved 1-column file to: {filepath}")
    except Exception as e:
        log.error(f"Error saving 1-column file: {e}")

# =============================================================================
# INTERNAL (HELPER) FUNCTIONS
# =============================================================================

def _zumontw(t: np.ndarray, omega: float, zeta: float) -> np.ndarray:
    """Generates the Suarez-Montejo Wavelet function [5]_.

    Internal helper function used in CWT calculations. This represents the
    'mother wavelet' function.

    Parameters
    ----------
    t : np.ndarray
        Time vector relative to the wavelet center (s).
    omega : float
        Central frequency parameter of the wavelet (rad/s). Typically pi.
    zeta : float
        Damping parameter controlling the wavelet decay. Typically 0.05.

    Returns
    -------
    np.ndarray
        The wavelet function evaluated at times `t`.

    References
    ----------
    .. [5] Suarez, L. E., & Montejo, L. A. (2005). Generation of artificial
           earthquakes...
    """
    # Ensure zeta is non-negative
    zeta = abs(zeta)
    wv = np.exp(-zeta * omega * np.abs(t)) * np.sin(omega * t)
    return wv

def _cwtzm(s: np.ndarray, fs: float, scales: np.ndarray, omega: float, zeta: float) -> np.ndarray:
    """Performs CWT using Suarez-Montejo wavelet via FFT convolution [3]_.

    Internal helper function. Computes the wavelet coefficients C(scale, time)
    by convolving the input signal with scaled versions of the mother wavelet.
    Includes dt scaling consistent with Ref [5] Eq. 16.

    Parameters
    ----------
    s : np.ndarray
        Input signal (e.g., acceleration time series in g's). 1D array.
    fs : float
        Sampling frequency of the input signal `s` (Hz).
    scales : np.ndarray
        1D array of scales at which to compute the CWT. Scales relate to the
        dilation of the mother wavelet and inversely to frequency.
    omega : float
        Central frequency parameter passed to `_zumontw` (rad/s). Typically pi.
    zeta : float
        Damping parameter passed to `_zumontw`. Typically 0.05.

    Returns
    -------
    np.ndarray
        2D array of wavelet coefficients, with shape (len(scales), len(s)).
        Each row corresponds to a scale, each column to a time point.

    Notes
    -----
    - Uses `scipy.signal.fftconvolve` with `mode='same'` for efficient computation
      of the convolution sum at each scale.
    - Applies energy normalization (1/sqrt(scale)) to the wavelet during convolution.
    - Multiplies the result of the convolution sum by `dt` (time step) to approximate
      the CWT integral definition, following Eq. 16 in Ref [5].

    References
    ----------
    .. [3] Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm...
           [cite: 2013 Montejo&Suarez an improved CWT-based algorithm...]
    .. [5] Suarez, L. E., & Montejo, L. A. (2005). Generation of artificial earthquakes...
           See Eq. 16. [cite: 2005 - Suarez and Montejo - Generation of artificial earthquakes..., 5910]
    """
    nf = len(scales)
    dt = 1 / fs
    n = len(s)
    t = np.linspace(0, (n - 1) * dt, n)
    # Center wavelet in time domain for convolution 'same' mode
    centertime = np.median(t)
    # Initialize array for coefficients (should be real for this wavelet)
    coefs = np.zeros((nf, n))

    for k in range(nf):
        # Time vector scaled relative to the current wavelet scale
        wv_t = (t - centertime) / scales[k]
        # Generate the scaled mother wavelet, applying energy normalization (1/sqrt(scale))
        # The function _zumontw generates the base wavelet shape
        wv = _zumontw(wv_t, omega, zeta) / np.sqrt(scales[k])

        # Compute the convolution sum using fftconvolve for efficiency
        # mode='same' ensures the output has the same length as the input signal `s`
        conv_sum = signal.fftconvolve(s, wv, mode='same')

        # Apply dt scaling to approximate the integral definition of CWT
        # as shown in the discrete approximation Eq. 16 of Ref [5].
        coefs[k, :] = conv_sum * dt

    return coefs

def _getdetails(t: np.ndarray, s: np.ndarray, C: np.ndarray, scales: np.ndarray, omega: float, zeta: float) -> Tuple[np.ndarray, np.ndarray]:
    """Generates detail functions D(scale, time) from CWT coefficients [3]_, [5]_.

    Internal helper function. Reconstructs the components of the signal
    associated with each scale. Uses K_psi constant and disables empirical scaling.

    Parameters
    ----------
    t : np.ndarray
        Time vector (s).
    s : np.ndarray
        Original signal (used ONLY if amplitude rescaling were active).
    C : np.ndarray
        2D array of wavelet coefficients (num_scales x num_time_points).
    scales : np.ndarray
        Array of scales corresponding to the rows of `C`.
    omega : float
        Central frequency parameter passed to `_zumontw`. Typically pi.
    zeta : float
        Damping parameter passed to `_zumontw`. Typically 0.05.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - D (np.ndarray): 2D array of detail functions (num_scales x num_time_points).
        - sr (np.ndarray): Signal reconstructed by integrating details over scales.

    Notes
    -----
    Uses the theoretical reconstruction constant K_psi in the scaling factor.
    The empirical final amplitude rescaling step is currently DISABLED.

    References
    ----------
    .. [3] Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm... [cite: 2013 Montejo&Suarez an improved CWT-based algorithm...]
    .. [5] Suarez, L. E., & Montejo, L. A. (2005). Generation of artificial earthquakes... [cite: 2005 - Suarez and Montejo - Generation of artificial earthquakes...]
    """
    NS, n = C.shape
    D = np.zeros((NS, n))
    centertime = np.median(t)
    dt = t[1] - t[0] # Time step for scaling factor

    # Calculate K_psi analytically (for zeta=0.05, omega=pi)
    # Could pre-calculate or pass as argument if parameters change
    if abs(zeta - 0.05) < 1e-6 and abs(omega - np.pi) < 1e-6:
        K_psi = 3.18242642 # Use pre-calculated value
    else:
        # Calculate K_psi using the full formula if params differ
        zeta2 = zeta**2
        zeta2_plus_1_sq = (zeta2 + 1)**2
        # Use np.arctan2 for potentially better numerical stability for tan_inv arg
        tan_inv_arg = 1 / (2 * zeta) - zeta / 2
        term_tan_inv = np.arctan(tan_inv_arg) # arctan handles large values correctly
        K_psi_num = (-4 * zeta * (zeta2 - 1)
                     + np.pi * zeta2_plus_1_sq
                     + 2 * zeta2_plus_1_sq * term_tan_inv)
        K_psi_den = 4 * zeta * zeta2_plus_1_sq * omega**2
        K_psi = K_psi_num / K_psi_den if abs(K_psi_den) > 1e-12 else 1.0 # Avoid division by zero
        log.info(f"Calculated K_psi = {K_psi:.8f} for zeta={zeta}, omega={omega}")

    if abs(K_psi) < 1e-9:
        warnings.warn("K_psi is near zero. Reconstruction might be unstable. Setting K_psi=1.")
        K_psi = 1.0

    for k in range(NS):
        wv_t = (t - centertime) / scales[k]
        # Wavelet used in reconstruction integral (MotherPsi, no 1/sqrt(scale))
        wv = _zumontw(wv_t, omega, zeta)

        # Convolution C(s,p) with MotherPsi((t-p)/s)
        # Assumes input C = C_true * dt (from _cwtzm)
        detail_k_sum = signal.fftconvolve(C[k, :], wv, mode='same')

        # Apply scaling factor: -dt / (K_psi * s^(5/2)) from Ref [5] Eq. 14 derivation.
        # Add negative sign from empirical testing.
        # Incorporates K_psi theoretically.
        scaling_factor = -dt / (K_psi * scales[k]**(5/2))
        D[k, :] = detail_k_sum * scaling_factor

    # Reconstruct signal by integrating details over scales.
    # Integral D(s,t) ds
    sr = np.trapezoid(D, scales, axis=0) # Integrate along scale axis

    # --- EMPIRICAL AMPLITUDE SCALING DISABLED ---
    # # Rescale reconstructed signal amplitude (empirical step)
    # max_abs_s = np.max(np.abs(s))
    # max_abs_sr = np.max(np.abs(sr))
    # if max_abs_sr > 1e-9: # Avoid division by zero
    #     ff = max_abs_s / max_abs_sr
    #     log.debug(f"Detail reconstruction amplitude correction factor (DISABLED): {ff:.4f}")
    #     # sr *= ff
    #     # D *= ff # Also scale the details matrix consistently
    # else:
    #     log.warning("Reconstructed signal from details has near-zero amplitude (before scaling).")
    # --- END ---

    return D, sr

def _CheckPeriodRange(T1: float, T2: float, To: np.ndarray, FF1: float, FF2: float) -> Tuple[float, float, float]:
    """Verifies matching period range against target spectrum and record limits.

    Internal helper function. Adjusts T1, T2, and FF1 if necessary.

    Parameters
    ----------
    T1 : float
        Requested lower bound of matching period range (s). 0 means use min of To.
    T2 : float
        Requested upper bound of matching period range (s). 0 means use max of To.
    To : np.ndarray
        Periods of the target spectrum (s). Assumed sorted.
    FF1 : float
        Current minimum frequency for CWT decomposition (Hz), defines max T record limit.
    FF2 : float
        Current maximum frequency for CWT decomposition (Hz), defines min T record limit.

    Returns
    -------
    Tuple[float, float, float]
        - updated_T1 (float): Adjusted lower period bound (s).
        - updated_T2 (float): Adjusted upper period bound (s).
        - updated_FF1 (float): Adjusted minimum CWT frequency (Hz), if T2 required it.

    Raises
    ------
    ValueError
        If the adjusted range results in T1 >= T2.
    """
    T_min_target, T_max_target = To[0], To[-1]
    T_min_record, T_max_record = 1 / FF2, 1 / FF1

    # Initialize with requested values or defaults
    updated_T1 = T1 if T1 > 1e-9 else T_min_target
    updated_T2 = T2 if T2 > 1e-9 else T_max_target

    if T1 <= 1e-9 and T2 <= 1e-9:
        log.info(f"Matching range defaulted to target spectrum range: [{updated_T1:.3f}s, {updated_T2:.3f}s]")

    # Check against target spectrum limits
    if updated_T1 < T_min_target:
        warnings.warn(f"Specified T1 ({T1:.3f}s) < target spectrum minimum ({T_min_target:.3f}s). Clamping T1.")
        updated_T1 = T_min_target
    if updated_T2 > T_max_target:
        warnings.warn(f"Specified T2 ({T2:.3f}s) > target spectrum maximum ({T_max_target:.3f}s). Clamping T2.")
        updated_T2 = T_max_target

    # Check against record frequency limits
    if updated_T1 < T_min_record:
        warnings.warn(f"Specified/Adjusted T1 ({updated_T1:.3f}s) < record Nyquist limit ({T_min_record:.3f}s). Clamping T1.")
        updated_T1 = T_min_record

    # Adjust CWT low frequency if needed for T2
    updated_FF1 = FF1
    if updated_T2 > T_max_record:
        updated_FF1 = 1 / updated_T2
        log.info(f"Adjusting CWT min frequency (FF1) to {updated_FF1:.3f} Hz to cover requested T2={updated_T2:.3f}s.")

    # Final sanity check
    if updated_T1 >= updated_T2:
        raise ValueError(f"Invalid matching range after adjustments: T1 ({updated_T1:.3f}s) >= T2 ({updated_T2:.3f}s)")

    log.info(f"Final matching period range set to: [{updated_T1:.3f}s, {updated_T2:.3f}s]")
    return updated_T1, updated_T2, updated_FF1
   
@jit(nopython=True, cache=True)
def _RSPW(T: np.ndarray, s: np.ndarray, zeta: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates response spectra (PSA, PSV, SA, SV, SD) using the exact
    solution for piecewise linear excitation (time-domain), strictly assuming underdamping.

    Comprehensive version returning SD, SV, SA, PSA, PSV. Internal helper function.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s). Must contain positive values.
    s : np.ndarray
        Input ground acceleration time series (g). Assumed valid.
    zeta : float
        Damping ratio. Must be >= 0 and < 1 for this function.
    dt : float
        Time step of the acceleration series (s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Pseudo-spectral acceleration (g).
        - PSV (np.ndarray): Pseudo-spectral velocity (units like g*s).
        - SA (np.ndarray): Absolute spectral acceleration (g).
        - SV (np.ndarray): Relative spectral velocity (units like g*s).
        - SD (np.ndarray): Relative spectral displacement (units like g*s^2).
          Returns arrays of NaNs if zeta < 0 or zeta >= 1.

    Notes
    -----
    - Solves the relative motion EOM: u'' + 2ζωn u' + ωn^2 u = -ag(t).
    - Implements the exact solution for underdamped systems (0 <= zeta < 1)
      assuming linear variation of -ag(t) between time steps, using a
      state-space formulation U(t+dt) = A*U(t) + B*P(t).
    - **Strictly requires 0 <= zeta < 1.** Returns NaNs and logs an error
      if zeta is outside this range.
    - Handles T=0 explicitly.
    - Absolute Acceleration SA is calculated as: -2ζωn u'(t) - ωn^2 u(t) + ag(t).
    """
    pi = np.pi
    nper = len(T)
    n = len(s)
    # Initialize output arrays
    SD = np.zeros(nper) # Relative Spectral Displacement
    SV = np.zeros(nper) # Relative Spectral Velocity
    SA = np.zeros(nper) # Absolute Spectral Acceleration
    PSA = np.zeros(nper)
    PSV = np.zeros(nper)

    # --- Validate damping ratio - Return NaNs immediately if invalid ---
    if not 0 <= zeta < 1:
        #log.error(f"Invalid damping ratio zeta={zeta:.4f}. _RSPW requires 0 <= zeta < 1. Returning NaNs.")
        SD[:] = np.nan; SV[:] = np.nan; SA[:] = np.nan
        PSA[:] = np.nan; PSV[:] = np.nan
        return PSA, PSV, SA, SV, SD
    # --- Damping is valid (0 <= zeta < 1) ---

    # Input for relative displacement equation uses negative ground acceleration
    s_input = -s

    # Define tolerance for T=0 check
    small_tolerance = 1e-12
    
    # Explicitly set T=0 values
    mask_T0 = (T <= small_tolerance)
    if np.any(mask_T0):
        #log.debug("Assigning T=0 values (SD=0, PSV=0, PSA=PGA).")
        pga = np.max(np.abs(s))
        SD[mask_T0] = 0.0
        PSV[mask_T0] = 0.0
        PSA[mask_T0] = pga
        SV[mask_T0] = 0.0 
        SA[mask_T0] = pga


    # Loop through strictly positive periods only
    valid_indices = np.where(T > small_tolerance)[0]

    for k in valid_indices:
        period = T[k]
        wn = 2 * pi / period # Natural frequency (rad/s)
        wn_sq = wn**2
        wn_cb = wn_sq * wn # Used in B matrix coeffs

        # State vector: u_state = [displacement, velocity]^T
        u_state = np.zeros((2, n)) # Stores [disp, vel] history
        # Array to store absolute acceleration history ---
        a_abs_hist = np.zeros(n)
        a_abs_hist[0] = s[0] # Initial condition (assuming u=0, u'=0)
        
        # --- Coefficients for state-space matrices A and B (Underdamped Case ONLY) ---
        sqrt_term = np.sqrt(1.0 - zeta**2)
        wd = wn * sqrt_term
        wd_inv = 1.0 / wd
        zeta_term = zeta / sqrt_term

        e_zwt = np.exp(-zeta * wn * dt)
        cos_wdt = np.cos(wd * dt)
        sin_wdt = np.sin(wd * dt)
        
        # Matrix A elements
        _a11 = e_zwt * (cos_wdt + zeta_term * sin_wdt)
        _a12 = e_zwt * wd_inv * sin_wdt
        _a21 = -wn * (1.0/sqrt_term) * e_zwt * sin_wdt
        _a22 = e_zwt * (cos_wdt - zeta_term * sin_wdt)
        
        
        # Matrix B elements
        _b11 = e_zwt * (((2 * zeta**2 - 1) / (wn_sq * dt) + zeta / wn) * wd_inv * sin_wdt +
                       (2 * zeta / (wn_cb * dt) + 1 / wn_sq) * cos_wdt) - 2 * zeta / (wn_cb * dt)
        _b12 = -e_zwt * (((2 * zeta**2 - 1) / (wn_sq * dt)) * wd_inv * sin_wdt +
                        (2 * zeta / (wn_cb * dt)) * cos_wdt) - (1 / wn_sq) + 2 * zeta / (wn_cb * dt)
        _b21 = -((_a11 - 1) / (wn_sq * dt)) - _a12
        _b22 = -_b21 - _a12

        A = np.array([[_a11, _a12], [_a21, _a22]])
        B = np.array([[_b11, _b12], [_b21, _b22]])

        # --- Time stepping using state-space solution ---
        
        c_term = 2.0 * zeta * wn # Damping term coefficient 2*zeta*wn
        k_term = wn_sq          # Stiffness term coefficient wn^2
        

        for q in range(n - 1):
            # U_{q+1} = A * U_q + B * P_q
            u_state[:, q + 1] = A @ u_state[:, q] + B @ np.array([s_input[q], s_input[q + 1]])

            # Calculate absolute acceleration at step q+1 ---
            # a_abs = -c*u' - k*u + ag
            u_rel_next = u_state[0, q + 1]
            v_rel_next = u_state[1, q + 1]
            a_abs_hist[q + 1] = -c_term * v_rel_next - k_term * u_rel_next
            


        # Find maximum absolute values from histories
        SD[k] = np.max(np.abs(u_state[0, :])) # Max relative displacement
        SV[k] = np.max(np.abs(u_state[1, :])) # Max relative velocity
        SA[k] = np.max(np.abs(a_abs_hist))   # Max absolute acceleration
        
    # --- Calculate Pseudo Spectra from SD ---
    mask_Tvalid = (T > small_tolerance)
    omega_n = 2 * pi / T[mask_Tvalid] 
    PSV[mask_Tvalid] = omega_n * SD[mask_Tvalid]
    PSA[mask_Tvalid] = omega_n**2 * SD[mask_Tvalid]
   
    return PSA, PSV, SA, SV, SD

def _RSFD(T: np.ndarray, s: np.ndarray, z: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Calculates response spectra (PSA, PSV, SA, SV, SD) via Frequency Domain.

    Internal helper function. Generally faster for moderate/high damping (z>=2%).

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s : np.ndarray
        Input acceleration time series (g).
    z : float
        Damping ratio.
    dt : float
        Time step (s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Pseudo-spectral acceleration (g).
        - PSV (np.ndarray): Pseudo-spectral velocity (value/g).
        - SA (np.ndarray): Absolute spectral acceleration (g).
        - SV (np.ndarray): Relative spectral velocity (value/g).
        - SD (np.ndarray): Relative spectral displacement (value/g).
    """
    pi = np.pi
    npo = len(s) # Original number of points
    nT = len(T)
    SD = np.zeros(nT); SV = np.zeros(nT); SA = np.zeros(nT)

    # Determine FFT length with zero-padding for sufficient quiet time
    # Pad to at least 10 cycles of the longest period
    n_pad_min = int(10 * np.max(T) / dt if nT > 0 and np.max(T) > 0 else 0)
    n_fft = int(2**np.ceil(np.log2(npo + n_pad_min)))
    s_padded = np.pad(s, (0, n_fft - npo))

    # Frequency vector for rfft
    freqs = np.fft.rfftfreq(n_fft, dt)
    ww = 2 * pi * freqs # Angular frequencies
    ffts = np.fft.rfft(s_padded) # FFT of ground acceleration

    m = 1.0 # Assumed mass
          
    # Loop through strictly positive periods only
    # Define valid_indices based on T > small_tolerance
    small_tolerance = 1e-12
    valid_indices = np.where(T > small_tolerance)[0]

    for kk in valid_indices:

        wn = 2 * pi / T[kk]
        k_stiff = m * wn**2
        c_damp = 2 * z * m * wn

        # Complex frequency response H(w) = 1 / (-mw^2 + i*c*w + k)
        denominator = (-m * ww**2 + k_stiff + 1j * c_damp * ww)
        denominator[np.abs(denominator) < 1e-15] = 1e-15
        # Transfer functions (relative response U(w) / GroundAccel(w))
        # Note: EOM is m*u'' + c*u' + k*u = -m*ag''
        # H_disp_over_accel = (-m) / denominator
        H_disp = (-m) / denominator
        H_vel = H_disp * (1j * ww)
        H_accel_rel = H_disp * (-ww**2)

        # Compute response spectra in frequency domain
        fft_disp = H_disp * ffts
        fft_vel = H_vel * ffts
        fft_accel_rel = H_accel_rel * ffts

        # Inverse FFT to get time domain response
        d = np.fft.irfft(fft_disp, n_fft)
        v = np.fft.irfft(fft_vel, n_fft)
        a_rel = np.fft.irfft(fft_accel_rel, n_fft)

        # Absolute acceleration = relative accel + ground accel
        a_abs = a_rel + s_padded[:n_fft] # Use padded ground accel

        # Find maximum absolute values over original duration
        SD[kk] = np.max(np.abs(d[:npo]))
        SV[kk] = np.max(np.abs(v[:npo]))
        SA[kk] = np.max(np.abs(a_abs[:npo]))

    # Calculate Pseudo Spectra from SD, handle T=0
    with np.errstate(divide='ignore', invalid='ignore'):
        PSV = (2 * pi / T) * SD
        PSA = (2 * pi / T)**2 * SD
        PSV[T <= small_tolerance] = 0.0
        PSA[T <= small_tolerance] = np.max(np.abs(s))
        # Set T=0 values for SV, SD, SA
        SV[T <= small_tolerance] = 0.0
        SD[T <= small_tolerance] = 0.0
        SA[T <= small_tolerance] = np.max(np.abs(s))

    return PSA, PSV, SA, SV, SD

def _basecorr(
        t: np.ndarray, 
        xg: np.ndarray, 
        CT: float, 
        porder: int = -1, 
        imax: int = 80, 
        tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Performs the core iterative baseline correction algorithm (Suarez & Montejo, 2007).

    Internal helper function called iteratively by `baselinecorrect`. 

    Methodology:
    - Window Definition: Isolates two specific segments of the acceleration history: 
      an initial window (length `CT`) and a final window (length `CT`). 
    - Displacement Correction: Integrates the record to find the final displacement error. 
      It computes positive and negative accumulators based on the acceleration values in 
      the *initial* window, weighted by a linearly decaying function. It then multiplies 
      the acceleration points in this initial window by a scaled factor to correct the 
      displacement.
    - Velocity Correction: Integrates the newly adjusted record to find the final 
      velocity error. It computes accumulators in the *final* window, weighted by a 
      linearly increasing function. It then scales the acceleration points in the final 
      window to pull the terminal velocity to zero.
    - Convergence: Checks if the final absolute values of velocity and displacement 
      are less than or equal to the specified `tol` (as a percentage of the peak 
      velocity/displacement). If not, the process loops up to `imax` times.

    Parameters
    ----------
    t : np.ndarray
        Time vector (s).
    xg : np.ndarray
        Input acceleration time series (g).
    CT : float
        Duration of the correction window applied strictly at the start and end 
        of the record (s).
    porder : int, optional
        Order of polynomial for initial global detrending. Default is -1 (none).
    imax : int, optional
        Maximum number of correction iterations. Default is 80.
    tol : float, optional
        Convergence tolerance as a percentage of the maximum response amplitude. 
        Default is 0.01 (1%).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - vel_orig (np.ndarray): Velocity of the initial (potentially detrended) record.
        - despl_orig (np.ndarray): Displacement of the initial record.
        - cxg (np.ndarray): Corrected acceleration time series (g).
        - cvel (np.ndarray): Velocity corresponding to `cxg`.
        - cdespl (np.ndarray): Displacement corresponding to `cxg`.
    """
    # Initial detrending (optional)
    if porder >= 0:
        try:
            coeffs = np.polyfit(t, xg, deg=porder)
            poly_trend = np.polyval(coeffs, t)
            xg_detrended = xg - poly_trend
            log.debug(f"Applied polynomial detrending of order {porder}.")
        except Exception as e:
            warnings.warn(f"Polynomial detrending failed: {e}. Proceeding without detrending.")
            xg_detrended = np.copy(xg)
    else:
        xg_detrended = np.copy(xg)

    n = len(xg_detrended)
    cxg = np.copy(xg_detrended) # Corrected acceleration starts here
    dt = t[1] - t[0]

    # Indices for correction windows
    L = max(0, int(np.ceil(CT / dt)) - 1) # Number of points (0 to L) at start
    M = max(L + 1, n - L - 1)              # Start index for end correction (M to n-1)

    if L == 0 or M >= n-1:
        warnings.warn(f"Correction time CT={CT:.2f}s is too small or large for record "
                    f"length {t[-1]:.2f}s. Skipping correction iteration.")
        # Return integrated original detrended record
        vel_orig = integrate.cumulative_trapezoid(xg_detrended, x=t, initial=0)
        despl_orig = integrate.cumulative_trapezoid(vel_orig, x=t, initial=0)
        return vel_orig, despl_orig, cxg, vel_orig, despl_orig # Return non-corrected


    log.debug(f"Baseline correction using L={L}, M={M}")

    for q in range(imax):
        # --- Correct for final displacement ---
        dU, ap, an = 0.0, 1e-15, -1e-15 # Add epsilon to avoid division by zero

        # Calculate displacement error dU = sum[(tf - ti) * acc(i) * dt]
        # Using numerical integration for potentially better accuracy
        current_vel = integrate.cumulative_trapezoid(cxg, x=t, initial=0)
        current_disp = integrate.cumulative_trapezoid(current_vel, x=t, initial=0)
        dU = current_disp[-1] # Target is 0, so error is the final value

        # Calculate accumulators ap, an for displacement correction (Eq. 27, 28)
        indices_start = np.arange(L + 1) # 0 to L
        time_factor = t[-1] - t[indices_start]
        weight_factor = (L - indices_start) / L if L > 0 else 0.0
        aux = weight_factor * time_factor * cxg[indices_start] * dt

        ap += np.sum(aux[aux >= 0])
        an += np.sum(aux[aux < 0])

        alfap = -dU / (2 * ap)
        alfan = -dU / (2 * an)

        # Apply displacement correction factors
        correction_start = np.zeros_like(cxg)
        correction_start[indices_start] = np.where(cxg[indices_start] > 0,
                                                   alfap * weight_factor,
                                                   alfan * weight_factor)
        cxg *= (1 + correction_start)

        # --- Correct for final velocity ---
        dV, vp, vn = 0.0, 1e-15, -1e-15 # Add epsilon

        # Calculate velocity error dV = sum[acc(i) * dt]
        current_vel_after_disp_corr = integrate.cumulative_trapezoid(cxg, x=t, initial=0)
        dV = current_vel_after_disp_corr[-1] # Target is 0

        # Calculate accumulators vp, vn for velocity correction (Eq. 34, 35)
        indices_end = np.arange(M, n) # M to n-1
        num_pts_end_window = n - M
        weight_factor_end = (indices_end - M) / (num_pts_end_window -1) if num_pts_end_window > 1 else 0.0
        auxv = weight_factor_end * cxg[indices_end] * dt

        vp += np.sum(auxv[auxv >= 0])
        vn += np.sum(auxv[auxv < 0])

        valfap = -dV / (2 * vp)
        valfan = -dV / (2 * vn)

        # Apply velocity correction factors
        correction_end = np.zeros_like(cxg)
        correction_end[indices_end] = np.where(cxg[indices_end] > 0,
                                               valfap * weight_factor_end,
                                               valfan * weight_factor_end)
        cxg *= (1 + correction_end)

        # --- Check convergence ---
        cvel_iter = integrate.cumulative_trapezoid(cxg, x=t, initial=0)
        cdespl_iter = integrate.cumulative_trapezoid(cvel_iter, x=t, initial=0)

        max_abs_vel = np.max(np.abs(cvel_iter))
        max_abs_disp = np.max(np.abs(cdespl_iter))

        # Relative error check
        errv = np.abs(cvel_iter[-1]) / max_abs_vel if max_abs_vel > 1e-12 else 0.0
        errd = np.abs(cdespl_iter[-1]) / max_abs_disp if max_abs_disp > 1e-12 else 0.0

        log.debug(f"_basecorr iter {q+1}: ErrV={errv*100:.3f}%, ErrD={errd*100:.3f}%")

        if errv * 100 <= tol and errd * 100 <= tol:
            log.debug(f"_basecorr converged after {q+1} iterations.")
            break
    else: # Loop finished without break
        warnings.warn(f"_basecorr did not converge within {imax} iterations.")

    # Final integration results
    cvel = integrate.cumulative_trapezoid(cxg, x=t, initial=0)
    cdespl = integrate.cumulative_trapezoid(cvel, x=t, initial=0)
    # Original integrated results (from potentially detrended input)
    vel_orig = integrate.cumulative_trapezoid(xg_detrended, x=t, initial=0)
    despl_orig = integrate.cumulative_trapezoid(vel_orig, x=t, initial=0)

    return vel_orig, despl_orig, cxg, cvel, cdespl

@jit(nopython=True, cache=True)
def _smooth_boxcar_variable(
    freqs_out: np.ndarray,
    freqs_in: np.ndarray,
    spectrum_in: np.ndarray,
    percentage: float = 20) -> np.ndarray:
    
    """
    Smooths a spectrum using a constant percentage (boxcar) window.

    This function applies a smoothing algorithm where, for each frequency in
    `freqs_out`, it averages all points from `spectrum_in` that fall within a
    frequency window. The window's width is a constant percentage of its
    center frequency, making it wider for higher frequencies.

    Parameters
    ----------
    freqs_out : numpy.ndarray
        1D array of frequencies at which the smoothed spectrum will be
        calculated.
    freqs_in : numpy.ndarray
        1D array of frequencies corresponding to the input spectrum.
    spectrum_in : numpy.ndarray
        1D array of amplitude values of the input spectrum. Must be the same
        length as `freqs_in`.
    percentage : float, optional
        The half-width of the smoothing window, expressed as percent of the
        center frequency. The default is 20, meaning the window for a
        center frequency `f` will span from `f * (1 - 0.2)` to `f * (1 + 0.2)`.

    Returns
    -------
    numpy.ndarray
        A 1D array containing the smoothed spectrum values, corresponding to the
        frequencies in `freqs_out`.

    Notes
    -----
    - This function is decorated with ``@numba.jit(nopython=True)``, which
      compiles it to fast machine code for significantly improved performance.
      The first call to the function will have a slight overhead due to this
      compilation step.
    - If no points from `freqs_in` fall within the smoothing window, the
      function falls back to **logarithmic interpolation** to estimate the
      value. This is more suitable for spectral data than linear interpolation.
    - For the logarithmic interpolation to work, all values in `freqs_in`,
      `freqs_out`, and `spectrum_in` must be positive.

    """
    n_out = len(freqs_out)
    n_in = len(freqs_in)
    smoothed_spectrum = np.zeros(n_out)
    percentage = percentage/100

    for i in range(n_out):
        f_center = freqs_out[i]
        f_lower = f_center * (1 - percentage)
        f_upper = f_center * (1 + percentage)

        current_sum = 0.0
        count = 0
        for j in range(n_in):
            # Check if the input frequency is within the window
            if freqs_in[j] >= f_lower and freqs_in[j] <= f_upper:
                current_sum += spectrum_in[j]
                count += 1

        if count > 0:
            # Average the values found within the window
            smoothed_spectrum[i] = current_sum / count
        else:
            # Fallback to logarithmic interpolation if no points are in the window
            log_x_target = np.log(f_center)
            log_x_known = np.log(freqs_in)
            log_y_known = np.log(spectrum_in)
            interp_log_y = np.interp(log_x_target, log_x_known, log_y_known)
            smoothed_spectrum[i] = np.exp(interp_log_y)

    return smoothed_spectrum

@jit(nopython=True, cache=True)
def _konno_ohmachi_1998_downsample(
    freqs_out: np.ndarray,
    freqs_in: np.ndarray,
    spectrum_in: np.ndarray,
    b: float) -> np.ndarray:
    
    """
    Applies Konno-Ohmachi smoothing to a spectrum.

    This function implements the widely used Konno-Ohmachi 1998 smoothing
    algorithm, which uses a spectral window that is constant on a logarithmic
    frequency scale. It is highly efficient due to Numba's JIT compilation.

    Parameters
    ----------
    freqs_out : numpy.ndarray
        1D array of frequencies at which the smoothed spectrum will be
        calculated.
    freqs_in : numpy.ndarray
        1D array of frequencies of the input spectrum.
    spectrum_in : numpy.ndarray
        1D array of amplitudes of the input spectrum. Must be the same length
        as `freqs_in`.
    b : float
        The smoothing coefficient, which controls the bandwidth of the
        smoothing window. Common values range from 13.5 (heavy smoothing)
        to 188.5 (moderate smoothing).

    Returns
    -------
    numpy.ndarray
        A 1D array of the smoothed spectrum values, corresponding to the
        frequencies in `freqs_out`.

    Notes
    -----
    - The weighting function is given by $W(x) = (\sin(x) / x)^4$, where
      $x = b \cdot \log_{10}(f / f_c)$.
    - This function is decorated with ``@numba.jit(nopython=True)``, which
      compiles it to fast machine code. The first call will have a slight
      overhead due to this compilation.
    - If no input frequencies are close enough to an output frequency to
      contribute significant weight, the function falls back to **logarithmic
      interpolation** for robustness.
    - The fallback interpolation requires all input frequency and spectrum
      values to be positive.

    References
    ----------
    .. [1] Konno, K. and Ohmachi, T. (1998). Ground-motion characteristics
           estimated from spectral ratio between horizontal and vertical
           components of microtremor. Bulletin of the Seismological Society
           of America, 88(1), pp.228-241.

    
    """
    n_out = len(freqs_out)
    n_in = len(freqs_in)
    smoothed_spectrum = np.zeros(n_out)

    for i in range(n_out):
        f_center = freqs_out[i]

        # Handle DC component (0 Hz) with simple linear interpolation
        if f_center == 0:
            smoothed_spectrum[i] = np.interp(0.0, freqs_in, spectrum_in)
            continue

        weighted_sum = 0.0
        total_weight = 0.0
        for j in range(n_in):
            f_current = freqs_in[j]
            if f_current == 0:
                continue

            # Calculate the weighting function argument
            x = b * np.log10(f_current / f_center)

            # Weight is (sin(x)/x)^4; handle x=0 case to avoid division by zero
            weight = (np.sin(x) / x)**4 if x != 0 else 1.0

            # Optimization: ignore negligible weights
            if weight > 1e-6:
                weighted_sum += spectrum_in[j] * weight
                total_weight += weight

        if total_weight > 0:
            # Calculate the weighted average
            smoothed_spectrum[i] = weighted_sum / total_weight
        else:
            # Fallback to logarithmic interpolation if no points have weight
            log_x_target = np.log(f_center)
            log_x_known = np.log(freqs_in)
            log_y_known = np.log(spectrum_in)
            interp_log_y = np.interp(log_x_target, log_x_known, log_y_known)
            smoothed_spectrum[i] = np.exp(interp_log_y)

    return smoothed_spectrum

def _konno_ohmachi_1998_sparse_matrix(
    freqs_out: np.ndarray,
    freqs_in: np.ndarray,
    b: float) -> "csr_matrix":
    
    """
    Builds a sparse matrix representation of the Konno-Ohmachi 1998 smoother.

    This function creates a linear operator in the form of a sparse matrix that,
    when multiplied with a spectrum, applies Konno-Ohmachi smoothing. This
    vectorized approach is highly efficient for applying the same smoothing
    to multiple spectra (e.g., rotated components).

    Parameters
    ----------
    freqs_out : numpy.ndarray
        1D array of frequencies at which the smoothed spectrum will be
        calculated. These define the rows of the output matrix.
    freqs_in : numpy.ndarray
        1D array of frequencies of the input spectrum. These define the
        columns of the output matrix.
    b : float
        The smoothing coefficient, which controls the bandwidth of the
        smoothing window. Common values range from 13.5 (heavy smoothing)
        to 188.5 (moderate smoothing).

    Returns
    -------
    scipy.sparse.csr_matrix
        A sparse matrix of shape `(len(freqs_out), len(freqs_in))` where each
        row represents the smoothing weights for a corresponding output frequency.
        Multiplying this matrix by an input spectrum vector yields the
        smoothed spectrum.

    Notes
    -----
    - The core calculation is vectorized using NumPy broadcasting to compute a
      dense matrix of weights, which is then sparsified.
    - To prevent `log(0)` errors, any zero-frequency components (DC offset) in
      the input arrays are temporarily replaced with a small epsilon (`1e-9`).
    - An optimization is applied to create the sparse structure: weights are
      only calculated for frequencies where the argument `x` is within `[-2π, 2π]`,
      as the `(sin(x)/x)^4` window function has its first nulls at these points
      and is negligible beyond them.
    - Each row of the final matrix is normalized by its sum to ensure that the
      smoothing operation conserves energy for a flat input spectrum.

    """
    n_out, n_in = len(freqs_out), len(freqs_in)

    # Replace zeros with a small number to avoid log(0) errors
    freqs_out_safe = np.copy(freqs_out)
    freqs_out_safe[freqs_out_safe == 0] = 1e-9
    freqs_in_safe = np.copy(freqs_in)
    freqs_in_safe[freqs_in_safe == 0] = 1e-9

    # Create a broadcasted matrix of log frequency ratios
    log_ratio_matrix = np.log10(freqs_in_safe / freqs_out_safe[:, np.newaxis])
    x = b * log_ratio_matrix

    # Calculate the window function values, handling the x=0 case
    W_unnormalized = np.ones_like(x)
    non_zero_mask = (x != 0)
    W_unnormalized[non_zero_mask] = (np.sin(x[non_zero_mask]) / x[non_zero_mask])**4

    # Sparsify the matrix by keeping only the central lobe of the window
    sparse_mask = np.abs(x) < (2 * np.pi)
    W_masked = W_unnormalized * sparse_mask

    # Normalize each row by its sum to create the final weights
    row_sums = W_masked.sum(axis=1)
    # Avoid division by zero for rows that have no contributing weights
    row_sums[row_sums == 0] = 1.0

    # Build the sparse matrix from the non-zero elements
    row_indices, col_indices = W_masked.nonzero()
    normalized_data = W_masked[row_indices, col_indices] / row_sums[row_indices]

    return scipy.sparse.coo_matrix(
        (normalized_data, (row_indices, col_indices)), shape=(n_out, n_in)
    ).tocsr()

