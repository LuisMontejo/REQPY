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
3.  **Advanced Matching (New in v0.3.0):** Generate records compatible with 
    PSA, minimum PSD, and/or minimum FAS requirements.
4.  **Signal Analysis (New in v0.3.0):** Compute FAS, PSD, RotDnn spectra, 
    Effective Amplitude Spectra (EAS), and Effective Power Spectra (EPSD) 
    with various smoothing options (including Konno-Ohmachi).
5.  **Correction Routines:** Baseline correction and localized time-domain 
    PGA correction.

===============================================================================
REFERENCES
===============================================================================

[1] Montejo, L. A. (2025). "Generation of Response Spectrum Compatible Records 
    Satisfying a Minimum Power Spectral Density Function." 
    Earthquake Engineering and Resilience. https://doi.org/10.1002/eer2.70008

[2] Montejo, L. A. (2024). "Strong-Motion-Duration-Dependent Power Spectral 
    Density Functions Compatible with Design Response Spectra." 
    Geotechnics 4(4), 1048-1064. https://doi.org/10.3390/geotechnics4040053

[3] Montejo, L. A. (2021). "Response spectral matching of horizontal ground 
    motion components to an orientation-independent spectrum (RotDnn)."
    Earthquake Spectra, 37(2), 1127-1144.

[4] Montejo, L. A., & Suarez, L. E. (2013). "An improved CWT-based algorithm 
    for the generation of spectrum-compatible records."
    International Journal of Advanced Structural Engineering, 5(1), 26.

[5] Suarez, L. E., & Montejo, L. A. (2007). "Applications of the wavelet 
    transform in the generation and analysis of spectrum-compatible records."
    Structural Engineering and Mechanics, 27(2), 173-197.

[6] Suarez, L. E., & Montejo, L. A. (2005). "Generation of artificial
    earthquakes via the wavelet transform." 
    Int. Journal of Solids and Structures, 42(21-22), 5905-5919.

===============================================================================
CHANGELOG
===============================================================================

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
__version__ = "0.3.0"
__email__ = "luis.montejo@upr.edu"

# =============================================================================
# IMPORTS
# =============================================================================
import logging
import numpy as np
from numba import jit
import scipy
from scipy import integrate, signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List, Optional, Union, Literal, Dict, Any
import warnings

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

def generate_psa_psd_fas_compatible_record(
    s: np.ndarray,
    fs: float,
    f_PSA: np.ndarray,
    targetPSA: np.ndarray,
    f_PSD: np.ndarray,
    targetPSD: np.ndarray,
    f_FAS: np.ndarray,
    targetFAS: np.ndarray,
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    PSDreduction: float = 1.0,
    FASreduction: float = 1.0,
    targetPGA: float = -1,
    F1PSA: float = 0.2,
    F2PSA: float = 50.0,
    F1Check: float = 0.3,
    F2Check: float = 30.0,
    zi: float = 0.05,
    BLcorr: bool = True,
    PSA_poly_order: int = 4,
    PSAPSD_poly_order: int = 4,
    NS: int = 300,
    nit: int = 30,
    maxit: int = 1000,
    localized: bool = True,
    smoothing_method: str = 'konno_ohmachi',
    smoothing_coeff: float = 20.0,
    prefer_pykooh: bool = True) -> Dict[str, Any]:
    
    """
    Generates a record compatible with Target PSA, minimum Target FAS, 
    and minimum Target PSD.

    This function extends the standard spectral matching workflow by adding two 
    subsequent adjustment stages. First, the seed record is modified using a 
    CWT-based iterative procedure to match the target PSA. Second, the record's 
    wavelet details are uniformly scaled to satisfy the minimum FAS requirement. 
    Third, localized adjustments are applied to the strong motion portion of 
    the details to satisfy the minimum PSD requirement.

    Parameters
    ----------
    s : numpy.ndarray
        Seed acceleration time-series.
    fs : float
        Sampling frequency of the seed record (Hz).
    f_PSA : numpy.ndarray
        Array of frequencies (Hz) at which the target PSA is defined.
    targetPSA : numpy.ndarray
        Target Pseudo-Spectral Acceleration amplitudes. Units must be consistent 
        with `s` (e.g., g).
    f_PSD : numpy.ndarray
        Array of frequencies (Hz) at which the target PSD is defined.
    targetPSD : numpy.ndarray
        Target Power Spectral Density amplitudes. Units must be consistent 
        with `s` squared per Hz (e.g., g^2/Hz).
    f_FAS : numpy.ndarray
        Array of frequencies (Hz) at which the target FAS is defined.
    targetFAS : numpy.ndarray
        Target Fourier Amplitude Spectrum amplitudes. Units must be consistent 
        with `s` times seconds (e.g., g-s).
    targetPSAlimits : tuple of float, optional
        Acceptable matching limits (min, max) for the PSA ratio (Record/Target).
        Default is (0.9, 1.3).
    PSDreduction : float, optional
        Factor to define the minimum required PSD level relative to the target 
        (e.g., 0.7 means the record PSD must be at least 70% of `targetPSD`). 
        Default is 1.0.
    FASreduction : float, optional
        Factor to define the minimum required FAS level relative to the target.
        Default is 1.0.
    targetPGA : float, optional
        Target Peak Ground Acceleration. If not -1, a PGA correction is performed 
        on the final record. Units must match `s`. Default is -1 (no correction).
    F1PSA, F2PSA : float, optional
        Frequency range (Hz) over which to perform PSA matching. 
        Default is 0.2 to 50.0.
    F1Check, F2Check : float, optional
        Frequency range (Hz) over which to check and adjust both the FAS and PSD. 
        Default is 0.3 to 30.0.
    zi : float, optional
        Damping ratio for response spectrum calculation. Default is 0.05.
    BLcorr : bool, optional
        If True, performs baseline correction on the final record. Default is True.
    PSA_poly_order : int, optional
        Polynomial order for detrending the PSA-matched record (before FAS/PSD 
        adjustments). Use -1 for no detrending. Default is 4.
    PSAPSD_poly_order : int, optional
        Polynomial order for detrending the final compatible record. 
        Use -1 for no detrending. Default is 4.
    NS : int, optional
        Number of scales/frequencies used for the CWT decomposition. Default is 300.
    nit : int, optional
        Number of iterations for the initial PSA matching phase. Default is 30.
    maxit : int, optional
        Maximum number of iterations for the FAS and PSD adjustment phases. 
        Default is 1000.
    localized : bool, optional
        If True, PSD adjustments are applied only to the strong motion portion 
        (SD5-75) of the record using a windowed approach. FAS adjustments are 
        always applied globally. Default is True.
    smoothing_method : {'variable_window', 'konno_ohmachi', 'none'}, optional
        Method used to smooth the record's spectra (FAS/PSD) for comparison 
        against the targets. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Smoothing coefficient (window width percentage for 'variable_window' or 
        bandwidth b for 'konno_ohmachi'). Default is 20.0.
    prefer_pykooh : bool, optional
        If True, uses the `pykooh` library for Konno-Ohmachi smoothing if installed.
        Default is True.

    Returns
    -------
    dict
        A dictionary containing the results:
        - 'sc' (numpy.ndarray): Record spectrally matched to PSA only.
        - 'sc_fas' (numpy.ndarray): Record matched to PSA and adjusted for FAS.
        - 'sca' (numpy.ndarray): Final record compatible with PSA, FAS, and PSD.
        - 't', 'dt': Time vector and time step.
        - 's_scaled': Seed record scaled to the target PSA.
        - 'psa_s', 'psa_sc', 'psa_sc_fas', 'psa_sca': PSA metrics for all stages.
        - 'fas_s', 'fas_sc', 'fas_sc_fas', 'fas_sca': FAS metrics for all stages.
        - 'psd_s', 'psd_sc', 'psd_sc_fas', 'psd_sca': PSD metrics for all stages.
        - 'target_psa', 'target_psd', 'target_fas': Interpolated targets.

    Notes
    -----
    This function is unit-agnostic. The units of the inputs must be consistent.
    For example, if `s` is in [g], `targetPSD` must be in [g^2/Hz] and 
    `targetFAS` in [g-s].
    """
    
    dt = 1/fs
    nt = len(s)
    if nt % 2 != 0: 
        s = np.append(s, 0)
        nt += 1
    
    t = np.linspace(0, (nt-1)*dt, nt)
    
    idx_f = np.argsort(f_PSA)
    f_PSA = f_PSA[idx_f]; targetPSA = targetPSA[idx_f]
    To_target = 1.0 / f_PSA
    idx_T = np.argsort(To_target)
    To_target = To_target[idx_T]; targetPSA_T = targetPSA[idx_T]
    
    idx_psd = np.argsort(f_PSD)
    f_PSD = f_PSD[idx_psd]; targetPSD = targetPSD[idx_psd]

    idx_fas = np.argsort(f_FAS)
    f_FAS = f_FAS[idx_fas]; targetFAS = targetFAS[idx_fas]

    T1PSA, T2PSA = 1.0/F2PSA, 1.0/F1PSA
    T1Check, T2Check = 1.0/F2Check, 1.0/F1Check
    
    FF1 = min(4/(nt*dt), 0.1); FF2 = 1/(2*dt)
    
    T1PSA, T2PSA, FF1 = _CheckPeriodRange(T1PSA, T2PSA, To_target, FF1, FF2)
    
    if T1Check < T1PSA or T2Check > T2PSA:
        log.warning("FAS/PSD matching range is outside PSA matching range.")

    omega = np.pi; zeta = 0.05
    freqs = np.geomspace(FF2, FF1, NS) 
    T = 1/freqs
    scales = omega / (2*np.pi*freqs)
    
    log.info("Performing CWT decomposition...")
    C = _cwtzm(s, fs, scales, omega, zeta)
    D, sr = _getdetails(t, s, C, scales, omega, zeta)
    
    ds = log_interp(T, To_target, targetPSA_T)
    TargetPSD_int = log_interp(freqs, f_PSD, targetPSD)
    TargetFAS_int = log_interp(freqs, f_FAS, targetFAS)
    
    TlocsPSA = np.where((T >= T1PSA) & (T <= T2PSA))[0]
    TlocsCheck = np.where((T >= T1Check) & (T <= T2Check))[0] 
    nTlocsPSA = len(TlocsPSA)

    PSAsr, _, _ = compute_spectrum(T, sr, zi, dt)
    sf = np.sum(ds[TlocsPSA]) / np.sum(PSAsr[TlocsPSA])
    
    sr = sf * sr; D = sf * D
    
    hPSAbc = np.zeros((NS, nit+1)); ns = np.zeros((nt, nit+1)); DN = np.zeros((NS, nt, nit+1))
    hPSAbc[:, 0] = sf * PSAsr; ns[:, 0] = sr; DN[:, :, 0] = D
    factorPSA = np.ones((NS, 1))
    
    log.info(f"Starting PSA matching ({nit} iterations)...")
    for qq in range(1, nit+1):
        factorPSA[TlocsPSA, 0] = ds[TlocsPSA] / hPSAbc[TlocsPSA, qq-1]
        DN[:, :, qq] = factorPSA * DN[:, :, qq-1]
        ns[:, qq] = np.trapz(DN[:, :, qq].T, scales)
        hPSAbc[:, qq], _, _ = compute_spectrum(T, ns[:, qq], zi, dt)
        
    brloc = np.argmin(np.linalg.norm(np.abs(hPSAbc[TlocsPSA, :] - ds[TlocsPSA][:, None]) / ds[TlocsPSA][:, None], axis=0))
    sc = ns[:, brloc]; Dc = DN[:, :, brloc]
    log.info(f"Best PSA match at it {brloc}")

    # --- FAS Adjustment ---
    sc_fas = np.copy(sc)
    Dc_fas = np.copy(Dc)

    def _calc_fas_internal(acc_series):
        _, _, _, fas_smooth = calculate_earthquake_fas(
            acc_series, fs, tukey_alpha=0.1, nfft_method='same',
            detrend_method='linear', smoothing_method=smoothing_method,
            smoothing_coeff=smoothing_coeff, prefer_pykooh=prefer_pykooh,
            downsample_freqs=None 
        )
        return fas_smooth

    fft_freqs = np.fft.rfftfreq(len(sc_fas), 1/fs)
    FAS_smooth_sc = _calc_fas_internal(sc_fas)
    MotionFAS_int = log_interp(freqs, fft_freqs, FAS_smooth_sc)

    ratioFAS = MotionFAS_int / TargetFAS_int
    minratioFAS = np.min(ratioFAS[TlocsCheck])

    if minratioFAS >= FASreduction:
        log.info("Minimum FAS requirement already satisfied.")
    else:
        log.info(f"Starting FAS adjustment. Initial min ratio: {minratioFAS:.2f}")
        cont = 0
        while minratioFAS < FASreduction and cont < maxit:
            crit_idx = np.argmin(ratioFAS[TlocsCheck])
            crit_neg_loc = TlocsCheck[crit_idx]
            factorn = 1.01 * (TargetFAS_int[crit_neg_loc] / MotionFAS_int[crit_neg_loc])
            
            Dc_fas[crit_neg_loc, :] = factorn * Dc_fas[crit_neg_loc, :]
            sc_fas = np.trapz(Dc_fas.T, scales)
            
            FAS_smooth_sc = _calc_fas_internal(sc_fas)
            MotionFAS_int = log_interp(freqs, fft_freqs, FAS_smooth_sc)
            ratioFAS = MotionFAS_int / TargetFAS_int
            minratioFAS = np.min(ratioFAS[TlocsCheck])
            cont += 1
        log.info(f"FAS adjustment finished at it {cont}. Final min ratio: {minratioFAS:.3f}")

    # --- PSD Adjustment ---
    # sc_fas is the input, but we do NOT apply corrections yet.
    sca = np.copy(sc_fas)
    Dca = np.copy(Dc_fas)
    
    def _calc_psd_internal(acc_series):
        ret = calculate_earthquake_psd(
            acc_series, fs, 
            tukey_alpha=0.1, duration_percent=(5,75), nfft_method='same',
            detrend_method='linear',
            smoothing_method=smoothing_method,
            smoothing_coeff=smoothing_coeff,
            prefer_pykooh=prefer_pykooh
        )
        psd_smooth = ret[5]
        _, _, _, t1, t2 = SignificantDuration(acc_series, t, 5, 75)
        return psd_smooth, t1, t2

    PSDavg_sca, t1sca, t2sca = _calc_psd_internal(sca)
    MotionPSD_int = log_interp(freqs, fft_freqs, PSDavg_sca)

    ratioPSD = MotionPSD_int / TargetPSD_int
    minratioPSD = np.min(ratioPSD[TlocsCheck])
    
    if minratioPSD >= PSDreduction:
        log.info("Minimum PSD requirement already satisfied.")
    else:
        log.info(f"Starting PSD adjustment. Initial min ratio: {minratioPSD:.2f}")
        cont = 0
        while minratioPSD < PSDreduction and cont < maxit:
            if localized:
                alphaw = 0.1
                locs = np.where((t >= t1sca - dt/2) & (t <= t2sca + dt/2))[0]
                if len(locs) > 0:
                    windowshort = signal.windows.tukey(len(locs), alphaw)
                    window = np.zeros(nt); window[locs] = windowshort; unos = np.ones(nt)
                    crit_idx = np.argmin(ratioPSD[TlocsCheck])
                    crit_neg_loc = TlocsCheck[crit_idx]
                    factorn = 1.01 * (TargetPSD_int[crit_neg_loc] / MotionPSD_int[crit_neg_loc])**0.5
                    scaling_vector = np.maximum(factorn * window, unos)
                    Dca[crit_neg_loc, :] = scaling_vector * Dca[crit_neg_loc, :]
            else:
                crit_idx = np.argmin(ratioPSD[TlocsCheck])
                crit_neg_loc = TlocsCheck[crit_idx]
                scaling_vector = 1.01 * (TargetPSD_int[crit_neg_loc] / MotionPSD_int[crit_neg_loc])**0.5
                Dca[crit_neg_loc, :] = scaling_vector * Dca[crit_neg_loc, :]

            sca = np.trapz(Dca.T, scales)
            PSDavg_sca, t1sca, t2sca = _calc_psd_internal(sca)
            MotionPSD_int = log_interp(freqs, fft_freqs, PSDavg_sca)
            ratioPSD = MotionPSD_int / TargetPSD_int
            minratioPSD = np.min(ratioPSD[TlocsCheck])
            cont += 1
                
        log.info(f"PSD adjustment finished at it {cont}. Final min ratio: {minratioPSD:.3f}")

    # --- 7. Final Corrections (PGA & Baseline) ---
    if targetPGA != -1:
        log.info(f"Correcting PGA to {targetPGA}g for both FAS-only and FAS+PSD records...")
        sca = pga_correction(targetPGA, t, sca)
        sc_fas = pga_correction(targetPGA, t, sc_fas)
        
    if BLcorr:
        log.info("Performing final baseline correction...")
        sca, _, _ = baselinecorrect(sca, t, porder=PSAPSD_poly_order)
        sc_fas, _, _ = baselinecorrect(sc_fas, t, porder=PSAPSD_poly_order)
        sc, _, _ = baselinecorrect(sc, t, porder=PSA_poly_order)

    # --- 8. Final Scaling Checks (Independent) ---
    log.info("Performing final scaling checks...")
    
    freqs_check = get_log_freqs(0.1, fs/2, 100); T_check = 1/freqs_check
    TargetPSD_check = log_interp(freqs_check, f_PSD, targetPSD)
    TargetFAS_check = log_interp(freqs_check, f_FAS, targetFAS)
    TargetPSA_check = log_interp(T_check, To_target, targetPSA_T)
    
    TlocsPSA_check = np.where((T_check >= T1PSA) & (T_check <= T2PSA))[0]
    FlocsCheck_check = np.where((freqs_check >= F1Check) & (freqs_check <= F2Check))[0]

    # --- Check Record: PSA + FAS + PSD (sca) ---
    # Calc Spectra for sca
    PSAsca, _, _ = compute_spectrum(T_check, sca, zi, dt)
    
    ret_psd_sca = calculate_earthquake_psd(
        sca, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, 
        nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check
    )
    PSDavg_sca_check = ret_psd_sca[5]
    if smoothing_method == 'variable_window':
        PSDavg_sca_check = log_interp(freqs_check, fft_freqs, PSDavg_sca_check)

    _, _, _, FASavg_sca_check = calculate_earthquake_fas(
        sca, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff,
        nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check
    )
    
    # Calc Factors for sca
    factorPSA_sca = np.ones(len(T_check)); factorPSD_sca = np.ones(len(freqs_check)); factorFAS_sca = np.ones(len(freqs_check))
    
    if len(TlocsPSA_check) > 0:
        factorPSA_sca[TlocsPSA_check] = targetPSAlimits[0] * TargetPSA_check[TlocsPSA_check] / PSAsca[TlocsPSA_check]
    if len(FlocsCheck_check) > 0:
        factorPSD_sca[FlocsCheck_check] = (PSDreduction * TargetPSD_check[FlocsCheck_check] / PSDavg_sca_check[FlocsCheck_check])**0.5
        factorFAS_sca[FlocsCheck_check] = (FASreduction * TargetFAS_check[FlocsCheck_check] / FASavg_sca_check[FlocsCheck_check])
    
    final_factor_sca = np.max(np.hstack((factorPSA_sca, factorPSD_sca, factorFAS_sca)))
    
    if final_factor_sca > 1.0:
        log.info(f"Final scaling required for PSA+PSD+FAS record (sca): {final_factor_sca:.3f}")
        sca = final_factor_sca * sca

    # --- Check Record: PSA + FAS (sc_fas) ---
    # Calc Spectra for sc_fas (Independent check)
    PSAsc_fas, _, _ = compute_spectrum(T_check, sc_fas, zi, dt)
    
    _, _, _, FASavg_sc_fas_check = calculate_earthquake_fas(
        sc_fas, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff,
        nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check
    )
    
    # Calc Factors for sc_fas (PSA and FAS only)
    factorPSA_fas = np.ones(len(T_check)); factorFAS_fas = np.ones(len(freqs_check))
    
    if len(TlocsPSA_check) > 0:
        factorPSA_fas[TlocsPSA_check] = targetPSAlimits[0] * TargetPSA_check[TlocsPSA_check] / PSAsc_fas[TlocsPSA_check]
    if len(FlocsCheck_check) > 0:
        factorFAS_fas[FlocsCheck_check] = (FASreduction * TargetFAS_check[FlocsCheck_check] / FASavg_sc_fas_check[FlocsCheck_check])
        
    final_factor_fas = np.max(np.hstack((factorPSA_fas, factorFAS_fas)))

    if final_factor_fas > 1.0:
        log.info(f"Final scaling required for PSA+FAS record (sc_fas): {final_factor_fas:.3f}")
        sc_fas = final_factor_fas * sc_fas

    # --- 9. COMPUTE ALL METRICS FOR RETURN ---
    log.info("Calculating final metrics for verification...")
    
    def _calc_metrics(acc):
        vel = integrate.cumulative_trapezoid(acc, t, initial=0)
        disp = integrate.cumulative_trapezoid(vel, t, initial=0)
        ai = integrate.cumulative_trapezoid(acc**2, t, initial=0)
        cav_proxy = integrate.cumulative_trapezoid(np.abs(vel), t, initial=0) 
        
        psa, _, _ = compute_spectrum(T_check, acc, zi, dt)
        
        ret_psd = calculate_earthquake_psd(
            acc, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, 
            nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check 
        )
        psd_raw = ret_psd[5]
        psd_final = log_interp(freqs_check, fft_freqs, psd_raw) if smoothing_method == 'variable_window' else psd_raw
            
        _, _, _, fas_raw = calculate_earthquake_fas(
            acc, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff,
            nfft_method='same', prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check
        )
        fas_final = fas_raw 

        return psa, psd_final, fas_final, vel, disp, ai, cav_proxy

    # Scaled Seed
    idx_seed_match = np.where((T_check >= T1PSA) & (T_check <= T2PSA))[0]
    psa_seed_raw, _, _ = compute_spectrum(T_check, s, zi, dt)
    sf_seed = np.sum(TargetPSA_check[idx_seed_match]) / np.sum(psa_seed_raw[idx_seed_match])
    s_scaled = s * sf_seed
    
    psa_s, psd_s, fas_s, v_s, d_s, ai_s, cav_s = _calc_metrics(s_scaled)
    psa_sc, psd_sc, fas_sc, v_sc, d_sc, ai_sc, cav_sc = _calc_metrics(sc)
    psa_sc_fas, psd_sc_fas, fas_sc_fas, v_sc_fas, d_sc_fas, ai_sc_fas, cav_sc_fas = _calc_metrics(sc_fas)
    psa_sca, psd_sca, fas_sca, v_sca, d_sca, ai_sca, cav_sca = _calc_metrics(sca)

    results = {
        't': t, 'dt': dt, 'freqs': freqs_check, 'periods': T_check,
        's_scaled': s_scaled, 'sc': sc, 'sc_fas': sc_fas, 'sca': sca,
        'psa_s': psa_s, 'psa_sc': psa_sc, 'psa_sc_fas': psa_sc_fas, 'psa_sca': psa_sca,
        'psd_s': psd_s, 'psd_sc': psd_sc, 'psd_sc_fas': psd_sc_fas, 'psd_sca': psd_sca,
        'fas_s': fas_s, 'fas_sc': fas_sc, 'fas_sc_fas': fas_sc_fas, 'fas_sca': fas_sca,
        'vel_s': v_s, 'vel_sc': v_sc, 'vel_sc_fas': v_sc_fas, 'vel_sca': v_sca,
        'disp_s': d_s, 'disp_sc': d_sc, 'disp_sc_fas': d_sc_fas, 'disp_sca': d_sca,
        'ai_s': ai_s, 'ai_sc': ai_sc, 'ai_sc_fas': ai_sc_fas, 'ai_sca': ai_sca,
        'cav_s': cav_s, 'cav_sc': cav_sc, 'cav_sc_fas': cav_sc_fas, 'cav_sca': cav_sca,
        'target_psa': TargetPSA_check, 'target_psd': TargetPSD_check, 'target_fas': TargetFAS_check
    }
    return results

def generate_psa_psd_compatible_record(
    s: np.ndarray,
    fs: float,
    f_PSA: np.ndarray,
    targetPSA: np.ndarray,
    f_PSD: np.ndarray,
    targetPSD: np.ndarray,
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    PSDreduction: float = 1.0,
    targetPGA: float = -1,
    F1PSA: float = 0.2,
    F2PSA: float = 50.0,
    F1PSD: float = 0.3,
    F2PSD: float = 30.0,
    zi: float = 0.05,
    BLcorr: bool = True,
    PSA_poly_order: int = 4,
    PSAPSD_poly_order: int = 4,
    NS: int = 300,
    nit: int = 30,
    maxit: int = 1000,
    localized: bool = True,
    smoothing_method: str = 'variable_window',
    smoothing_coeff: float = 20.0,
    prefer_pykooh: bool = True) -> Dict[str, Any]:
    
    """
    Generates a record compatible with both a target Response Spectrum (PSA)
    and a minimum Target Power Spectral Density (PSD).

    This function modifies an input seed record using a CWT-based iterative 
    procedure to match a target PSA. It then performs further localized 
    adjustments to ensuring the record's PSD meets a specified minimum requirement 
    defined by a target PSD function.

    Parameters
    ----------
    s : numpy.ndarray
        Seed acceleration time-series.
    fs : float
        Sampling frequency of the seed record (Hz).
    f_PSA : numpy.ndarray
        Array of frequencies (Hz) at which the target PSA is defined.
    targetPSA : numpy.ndarray
        Target Pseudo-Spectral Acceleration amplitudes. Units must be consistent 
        with `s` (e.g., g).
    f_PSD : numpy.ndarray
        Array of frequencies (Hz) at which the target PSD is defined.
    targetPSD : numpy.ndarray
        Target Power Spectral Density amplitudes. Units must be consistent 
        with `s` squared per Hz (e.g., g^2/Hz).
    targetPSAlimits : tuple of float, optional
        Acceptable matching limits (min, max) for the PSA ratio (Record/Target).
        Default is (0.9, 1.3), consistent with SRP 3.7.1.
    PSDreduction : float, optional
        Factor to define the minimum required PSD level relative to the target 
        (e.g., 0.7 means the record PSD must be at least 70% of `targetPSD`). 
        Default is 1.0.
    targetPGA : float, optional
        Target Peak Ground Acceleration. If not -1, a PGA correction is performed 
        after matching. Units must match `s`. Default is -1 (no correction).
    F1PSA, F2PSA : float, optional
        Frequency range (Hz) over which to perform PSA matching. 
        Default is 0.2 to 50.0.
    F1PSD, F2PSD : float, optional
        Frequency range (Hz) over which to check and adjust the PSD. 
        Default is 0.3 to 30.0.
    zi : float, optional
        Damping ratio for response spectrum calculation. Default is 0.05.
    BLcorr : bool, optional
        If True, performs baseline correction on the final record. Default is True.
    PSA_poly_order : int, optional
        Polynomial order for detrending the PSA-matched record (before PSD 
        adjustment). Use -1 for no detrending. Default is 4.
    PSAPSD_poly_order : int, optional
        Polynomial order for detrending the final PSA-PSD compatible record. 
        Use -1 for no detrending. Default is 4.
    NS : int, optional
        Number of scales/frequencies used for the CWT decomposition. Default is 300.
    nit : int, optional
        Number of iterations for the initial PSA matching phase. Default is 30.
    maxit : int, optional
        Maximum number of iterations for the PSD adjustment phase. Default is 1000.
    localized : bool, optional
        If True, PSD adjustments are applied only to the strong motion portion 
        (SD5-75) of the record using a windowed approach. Default is True.
    smoothing_method : {'variable_window', 'konno_ohmachi', 'none'}, optional
        Method used to smooth the record's PSD for comparison against the target.
        Default is 'variable_window' (SRP 3.7.1 style).
    smoothing_coeff : float, optional
        Smoothing coefficient. For 'variable_window', this is the window width 
        percentage (e.g., 20.0 for +/- 20%). For 'konno_ohmachi', this is the 
        bandwidth parameter b. Default is 20.0.
    prefer_pykooh : bool, optional
        If True, uses the `pykooh` library for Konno-Ohmachi smoothing if installed.
        Default is True.

    Returns
    -------
    dict
        A dictionary containing the results:
        - 'sc' (numpy.ndarray): Record spectrally matched to PSA only.
        - 'sca' (numpy.ndarray): Final record compatible with PSA and minimum PSD.
        - 't' (numpy.ndarray): Time vector corresponding to the records.
        - 'dt' (float): Time step.
        - 's_scaled' (numpy.ndarray): Seed record scaled to the target PSA.
        - 'psa_s', 'psa_sc', 'psa_sca' (tuple): PSA metrics (PSA, PSD, vel, etc.) 
          for the scaled seed, PSA-matched, and final records, respectively.
        - 'target_psa' (numpy.ndarray): Interpolated target PSA.
        - 'target_psd' (numpy.ndarray): Interpolated target PSD.

    Notes
    -----
    This function is unit-agnostic. The units of the input `targetPSD` must be 
    consistent with the units of the seed record `s`. For example, if `s` is in [g], 
    `targetPSD` must be provided in [g^2/Hz]. The internal calculations do not 
    perform unit conversion.

    References
    ----------
    .. [1] Montejo, L.A. (2025). "Generation of Response Spectrum Compatible 
           Records Satisfying a Minimum Power Spectral Density Function." 
           Earthquake Engineering and Resilience.
    """
    dt = 1/fs
    nt = len(s)
    if nt % 2 != 0: 
        s = np.append(s, 0)
        nt += 1
    
    t = np.linspace(0, (nt-1)*dt, nt)
    
    # 1. Sort and Prepare Targets
    idx_f = np.argsort(f_PSA)
    f_PSA = f_PSA[idx_f]
    targetPSA = targetPSA[idx_f]
    
    To_target = 1.0 / f_PSA
    idx_T = np.argsort(To_target)
    To_target = To_target[idx_T]
    targetPSA_T = targetPSA[idx_T] 
    
    idx_psd = np.argsort(f_PSD)
    f_PSD = f_PSD[idx_psd]
    targetPSD = targetPSD[idx_psd]

    # 2. Setup CWT Parameters
    T1PSA, T2PSA = 1.0/F2PSA, 1.0/F1PSA
    T1PSD, T2PSD = 1.0/F2PSD, 1.0/F1PSD
    
    FF1 = min(4/(nt*dt), 0.1)
    FF2 = 1/(2*dt)
    
    T1PSA, T2PSA, FF1 = _CheckPeriodRange(T1PSA, T2PSA, To_target, FF1, FF2)
    
    if T1PSD < T1PSA or T2PSD > T2PSA:
        log.warning("PSD matching range is outside PSA matching range.")

    omega = np.pi
    zeta = 0.05
    freqs = np.geomspace(FF2, FF1, NS) 
    T = 1/freqs
    scales = omega / (2*np.pi*freqs)
    
    # 3. CWT Decomposition
    log.info("Performing CWT decomposition...")
    C = _cwtzm(s, fs, scales, omega, zeta)
    D, sr = _getdetails(t, s, C, scales, omega, zeta)
    
    ds = log_interp(T, To_target, targetPSA_T)
    TargetPSD_int = log_interp(freqs, f_PSD, targetPSD)
    
    TlocsPSA = np.where((T >= T1PSA) & (T <= T2PSA))[0]
    TlocsPSD = np.where((T >= T1PSD) & (T <= T2PSD))[0]
    nTlocsPSA = len(TlocsPSA)

    # 4. PSA Matching Loop
    PSAsr, _, _ = compute_spectrum(T, sr, zi, dt)
    sf = np.sum(ds[TlocsPSA]) / np.sum(PSAsr[TlocsPSA])
    
    sr = sf * sr
    D = sf * D
    
    hPSAbc = np.zeros((NS, nit+1))
    ns = np.zeros((nt, nit+1))
    DN = np.zeros((NS, nt, nit+1))
    
    hPSAbc[:, 0] = sf * PSAsr
    ns[:, 0] = sr
    DN[:, :, 0] = D
    
    factorPSA = np.ones((NS, 1))
    
    log.info(f"Starting PSA matching ({nit} iterations)...")
    for qq in range(1, nit+1):
        factorPSA[TlocsPSA, 0] = ds[TlocsPSA] / hPSAbc[TlocsPSA, qq-1]
        DN[:, :, qq] = factorPSA * DN[:, :, qq-1]
        ns[:, qq] = np.trapz(DN[:, :, qq].T, scales)
        hPSAbc[:, qq], _, _ = compute_spectrum(T, ns[:, qq], zi, dt)
        
    difPSA = np.abs(hPSAbc[TlocsPSA, :] - ds[TlocsPSA][:, None]) / ds[TlocsPSA][:, None]
    rmsePSA = np.linalg.norm(difPSA, axis=0) / np.sqrt(nTlocsPSA) * 100
    brloc = np.argmin(rmsePSA)
    
    sc = ns[:, brloc]
    Dc = DN[:, :, brloc]
    log.info(f"Best PSA match at it {brloc}, RMSE: {rmsePSA[brloc]:.2f}%")

    # 5. PSD Adjustment Loop
    sca = np.copy(sc)
    Dca = np.copy(Dc)
    
    def _calc_psd_internal(acc_series):
        ret = calculate_earthquake_psd(
            acc_series, fs, 
            tukey_alpha=0.1, duration_percent=(5,75), nfft_method='same',
            detrend_method='linear',
            smoothing_method=smoothing_method,
            smoothing_coeff=smoothing_coeff,
            prefer_pykooh=prefer_pykooh
        )
        psd_smooth = ret[5]
        _, _, _, t1, t2 = SignificantDuration(acc_series, t, 5, 75)
        return psd_smooth, t1, t2

    PSDavg_sca, t1sca, t2sca = _calc_psd_internal(sca)
    
    fft_freqs = np.fft.rfftfreq(len(sca), 1/fs)
    MotionPSD_int = log_interp(freqs, fft_freqs, PSDavg_sca)

    ratio = MotionPSD_int / TargetPSD_int
    minratio = np.min(ratio[TlocsPSD])
    
    if minratio >= PSDreduction:
        log.info("Minimum PSD requirement already satisfied.")
    else:
        log.info(f"Starting PSD adjustment. Initial min ratio: {minratio:.2f} (Target: {PSDreduction})")
        cont = 0
        while minratio < PSDreduction and cont < maxit:
            if localized:
                alphaw = 0.1
                locs = np.where((t >= t1sca - dt/2) & (t <= t2sca + dt/2))[0]
                if len(locs) > 0:
                    windowshort = signal.windows.tukey(len(locs), alphaw)
                    window = np.zeros(nt); window[locs] = windowshort; unos = np.ones(nt)
                    crit_neg_loc = TlocsPSD[0] + np.argmin(ratio[TlocsPSD])
                    scaling_vector = np.maximum(1.01 * (TargetPSD_int[crit_neg_loc] / MotionPSD_int[crit_neg_loc])**0.5 * window, unos)
                    Dca[crit_neg_loc, :] = scaling_vector * Dca[crit_neg_loc, :]
            else:
                crit_neg_loc = TlocsPSD[0] + np.argmin(ratio[TlocsPSD])
                scaling_vector = 1.01 * (TargetPSD_int[crit_neg_loc] / MotionPSD_int[crit_neg_loc])**0.5
                Dca[crit_neg_loc, :] = scaling_vector * Dca[crit_neg_loc, :]

            sca = np.trapz(Dca.T, scales)
            PSDavg_sca, t1sca, t2sca = _calc_psd_internal(sca)
            MotionPSD_int = log_interp(freqs, fft_freqs, PSDavg_sca)
            ratio = MotionPSD_int / TargetPSD_int
            minratio = np.min(ratio[TlocsPSD])
            cont += 1
            if cont % 50 == 0: log.debug(f"PSD Iteration {cont}: Min Ratio = {minratio:.3f}")
                
        log.info(f"PSD adjustment finished at it {cont}. Final min ratio: {minratio:.3f}")

    # 6. PGA Correction
    if targetPGA != -1:
        log.info(f"Correcting PGA to {targetPGA}g...")
        sca = pga_correction(targetPGA, t, sca)

    # 7. Final Scaling & Baseline Correction
    freqs_check = get_log_freqs(0.1, fs/2, 100)
    T_check = 1/freqs_check
    
    PSAsca, _, _ = compute_spectrum(T_check, sca, zi, dt)
    
    ret_final = calculate_earthquake_psd(
        sca, fs, tukey_alpha=0.1, duration_percent=(5,75), nfft_method='same',
        smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff,
        prefer_pykooh=prefer_pykooh, downsample_freqs=freqs_check
    )
    PSDavg_sca_final = ret_final[5]
    
    if smoothing_method == 'variable_window':
        PSDavg_sca_check = log_interp(freqs_check, fft_freqs, PSDavg_sca_final)
    else:
        PSDavg_sca_check = PSDavg_sca_final

    TargetPSD_check = log_interp(freqs_check, f_PSD, targetPSD)
    TargetPSA_check = log_interp(T_check, To_target, targetPSA_T)
    
    TlocsPSA_check = np.where((T_check >= T1PSA) & (T_check <= T2PSA))[0]
    FlocsPSD_check = np.where((freqs_check >= F1PSD) & (freqs_check <= F2PSD))[0]
    
    factorPSA = np.ones(len(T_check))
    factorPSD = np.ones(len(freqs_check))
    
    if len(TlocsPSA_check) > 0:
        factorPSA[TlocsPSA_check] = targetPSAlimits[0] * TargetPSA_check[TlocsPSA_check] / PSAsca[TlocsPSA_check]
    if len(FlocsPSD_check) > 0:
        factorPSD[FlocsPSD_check] = (PSDreduction * TargetPSD_check[FlocsPSD_check] / PSDavg_sca_check[FlocsPSD_check])**0.5
    
    final_factor = np.max(np.hstack((factorPSA, factorPSD)))
    if final_factor > 1.0:
        log.info(f"Final scaling required to meet lower bounds: {final_factor:.3f}")
        sca = final_factor * sca
    
    if BLcorr:
        log.info("Performing baseline correction...")
        sca, _, _ = baselinecorrect(sca, t, porder=PSAPSD_poly_order)
        sc, _, _ = baselinecorrect(sc, t, porder=PSA_poly_order)

    # --- 8. COMPUTE ALL METRICS FOR RETURN ---
    log.info("Calculating final metrics for verification...")
    
    def _calc_metrics(acc):
        vel = integrate.cumulative_trapezoid(acc, t, initial=0)
        disp = integrate.cumulative_trapezoid(vel, t, initial=0)
        ai = integrate.cumulative_trapezoid(acc**2, t, initial=0)
        cav_proxy = integrate.cumulative_trapezoid(np.abs(vel), t, initial=0) 
        
        psa, _, _ = compute_spectrum(T_check, acc, zi, dt)
        
        ret = calculate_earthquake_psd(
            acc, fs, smoothing_method=smoothing_method, smoothing_coeff=smoothing_coeff, 
            nfft_method='same', prefer_pykooh=prefer_pykooh,
            downsample_freqs=freqs_check 
        )
        psd_smooth_raw = ret[5]
        
        if smoothing_method == 'variable_window':
            psd_final = log_interp(freqs_check, fft_freqs, psd_smooth_raw)
        else:
            psd_final = psd_smooth_raw
            
        return psa, psd_final, vel, disp, ai, cav_proxy

    # Scaled Seed
    idx_seed_match = np.where((T_check >= T1PSA) & (T_check <= T2PSA))[0]
    psa_seed_raw, _, _ = compute_spectrum(T_check, s, zi, dt)
    sf_seed = np.sum(TargetPSA_check[idx_seed_match]) / np.sum(psa_seed_raw[idx_seed_match])
    s_scaled = s * sf_seed
    
    psa_s, psd_s, v_s, d_s, ai_s, cav_s = _calc_metrics(s_scaled)
    psa_sc, psd_sc, v_sc, d_sc, ai_sc, cav_sc = _calc_metrics(sc)
    psa_sca, psd_sca, v_sca, d_sca, ai_sca, cav_sca = _calc_metrics(sca)

    results = {
        't': t, 'dt': dt, 'freqs': freqs_check, 'periods': T_check,
        's_scaled': s_scaled, 'sc': sc, 'sca': sca,
        'psa_s': psa_s, 'psa_sc': psa_sc, 'psa_sca': psa_sca,
        'psd_s': psd_s, 'psd_sc': psd_sc, 'psd_sca': psd_sca,
        'vel_s': v_s, 'vel_sc': v_sc, 'vel_sca': v_sca,
        'disp_s': d_s, 'disp_sc': d_sc, 'disp_sca': d_sca,
        'ai_s': ai_s, 'ai_sc': ai_sc, 'ai_sca': ai_sca,
        'cav_s': cav_s, 'cav_sc': cav_sc, 'cav_sca': cav_sca,
        'target_psa': TargetPSA_check, 'target_psd': TargetPSD_check
    }

    return results

def REQPYrotdnn(
    s1: np.ndarray,
    s2: np.ndarray,
    fs: float,
    dso: np.ndarray,
    To: np.ndarray,
    nn: int,
    T1: float = 0.0,
    T2: float = 0.0,
    zi: float = 0.05,
    nit: int = 15,
    NS: int = 100,
    baseline: bool = True,
    porder: int = -1) -> Dict[str, Any]:
    
    """Response spectral matching of horizontal ground motion components
    to an orientation-independent spectrum (RotDnn).

    Modifies two input horizontal acceleration time series (s1, s2) such
    that their resulting orientation-independent response spectrum
    (e.g., RotD100) matches the target spectrum (dso).

    Parameters
    ----------
    s1 : np.ndarray
        Seed record for the first horizontal component (acceleration in g's).
    s2 : np.ndarray
        Seed record for the second horizontal component (acceleration in g's).
    fs : float
        Sampling frequency of the seed records (Hz).
    dso : np.ndarray
        Target/design pseudo-acceleration spectrum ordinates (g).
    To : np.ndarray
        Periods corresponding to the target spectrum ordinates `dso` (s).
        Must be sorted in ascending order.
    nn : int
        Percentile for the orientation-independent spectrum (e.g., 100 for
        RotD100, 50 for RotD50).
    T1 : float, optional
        Lower bound of the period range for spectral matching (s).
        Default is 0.0, which uses the lowest period in `To`.
    T2 : float, optional
        Upper bound of the period range for spectral matching (s).
        Default is 0.0, which uses the highest period in `To`.
    zi : float, optional
        Damping ratio for response spectrum calculation. Default is 0.05 (5%).
    nit : int, optional
        Maximum number of iterations for the matching algorithm. Default is 15.
    NS : int, optional
        Number of frequency/scale values used for the CWT decomposition and
        response spectra calculation. Default is 100.
    baseline : bool, optional
        Flag to perform baseline correction on the matched records.
        Default is True.
    porder : int, optional
        Order of the polynomial used for initial detrending during baseline
        correction. Only used if `baseline=True`. Default is -1 (no detrending).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results:
        - 'scc1' (np.ndarray): Matched acceleration time series for comp 1 (g).
        - 'scc2' (np.ndarray): Matched acceleration time series for comp 2 (g).
        - 'cvel1' (np.ndarray): Velocity time history for matched comp 1 (vel/g).
        - 'cvel2' (np.ndarray): Velocity time history for matched comp 2 (vel/g).
        - 'cdisp1' (np.ndarray): Displacement time history for matched comp 1 (displ/g).
        - 'cdisp2' (np.ndarray): Displacement time history for matched comp 2 (displ/g).
        - 'PSArotnn' (np.ndarray): RotDnn spectrum of the matched components (g).
        - 'PSArotnnor' (np.ndarray): RotDnn spectrum of the original seed components (g).
        - 'T' (np.ndarray): Periods corresponding to the calculated spectra (s).
        - 'meanefin' (float): Final average misfit within the matching range (%).
        - 'rmsefin' (float): Final root mean squared error within the matching range (%).
        - 'sf' (float): Initial scaling factor applied to seed records.
        - 'dt' (float): Time step of the records (s).

    Notes
    -----
    The core algorithm [1] uses the Continuous Wavelet Transform (CWT) with the
    Suarez-Montejo wavelet [5]_ and performs calculations efficiently in the
    frequency domain [3]_. Baseline correction uses the method described in [4]_.
    This function does *not* perform plotting or file saving; use the returned
    dictionary and helper plotting functions for that.

    References
    ----------
    .. [1] Montejo, L. A. (2021). Response spectral matching...
    .. [3] Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm...
    .. [4] Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform...
    .. [5] Suarez, L. E., & Montejo, L. A. (2005). Generation of artificial earthquakes...
    """
    pi = np.pi
    theta = np.arange(0, 180, 1)

    # Ensure equal length
    n1 = np.size(s1); n2 = np.size(s2); n = min(n1, n2)
    if n1 != n2:
        warnings.warn(f"Input records have different lengths ({n1} vs {n2}). Truncating to {n} points.")
    s1 = s1[:n]; s2 = s2[:n]

    dt = 1 / fs                     # time step
    t = np.linspace(0, (n - 1) * dt, n) # time vector
    # Frequency range for CWT decomposition depends on record duration
    FF1 = min(4 / (n * dt), 0.1); FF2 = 1 / (2 * dt)

    # Ensure target spectrum periods are sorted
    Tsortindex = np.argsort(To)
    if not np.all(Tsortindex == np.arange(len(To))):
        log.info("Target spectrum periods were not sorted. Sorting now.")
        To = To[Tsortindex]
        dso = dso[Tsortindex]

    T1, T2, FF1 = _CheckPeriodRange(T1, T2, To, FF1, FF2) # verifies period range

    # Perform Continuous Wavelet Decomposition:
    omega = pi; zeta = 0.05             # wavelet function parameters
    freqs = np.geomspace(FF2, FF1, NS)  # frequencies vector (log spaced)
    T = 1 / freqs                       # periods vector
    scales = omega / (2 * pi * freqs)   # scales vector
    C1 = _cwtzm(s1, fs, scales, omega, zeta)
    C2 = _cwtzm(s2, fs, scales, omega, zeta)
    log.info("Wavelet decomposition performed.")

    # Generate detail functions:
    D1, sr1 = _getdetails(t, s1, C1, scales, omega, zeta)
    D2, sr2 = _getdetails(t, s2, C2, scales, omega, zeta)
    log.info("Detail functions generated.")

    # Interpolate target spectrum to match CWT periods
    ds = np.interp(T, To, dso, left=np.nan, right=np.nan)
    # Define indices within the matching period range
    Tlocs = np.where((T >= T1) & (T <= T2))[0]
    if len(Tlocs) == 0:
        raise ValueError("No target spectrum points found within the specified matching range.")

    # Calculate initial RotDnn spectrum and scaling factor
    PSA180or, _, _ = compute_rotated_spectra(T, s1, s2, zi, dt, theta)
    PSArotnnor = np.percentile(PSA180or, nn, axis=0)

    # Calculate initial scaling factor based on energy in the matching range
    sf = np.sum(ds[Tlocs]) / np.sum(PSArotnnor[Tlocs])
    log.info("Initial scaling factor: %.4f", sf)

    # Apply initial scaling
    sc1 = sf * sr1; D1 = sf * D1
    sc2 = sf * sr2; D2 = sf * D2

    # --- Iterative Process ---
    meane = np.zeros(nit + 1)
    rmse = np.zeros(nit + 1)
    hPSArotnn = np.zeros((NS, nit + 1))
    ns1 = np.zeros((n, nit + 1))
    ns2 = np.zeros((n, nit + 1))

    # Initial state (iteration 0)
    PSA180_iter0, _, _ = compute_rotated_spectra(T, sc1, sc2, zi, dt, theta)
    hPSArotnn[:, 0] = np.percentile(PSA180_iter0, nn, axis=0)
    ns1[:, 0] = sc1
    ns2[:, 0] = sc2

    diff = np.abs(hPSArotnn[Tlocs, 0] - ds[Tlocs]) / ds[Tlocs]
    meane[0] = np.mean(diff) * 100
    rmse[0] = np.linalg.norm(diff) / np.sqrt(len(Tlocs)) * 100
    log.info("Iteration 0: RMSE=%.2f%%, Misfit=%.2f%%", rmse[0], meane[0])

    factor = np.ones(NS)

    for m in range(1, nit + 1):
        log.info("Starting iteration %d of %d...", m, nit)
        # Calculate scaling factors only within the matching range
        factor[Tlocs] = ds[Tlocs] / hPSArotnn[Tlocs, m - 1]

        # Apply factors to details (use broadcasting)
        D1 = D1 * factor[:, np.newaxis]
        D2 = D2 * factor[:, np.newaxis]

        # Reconstruct signals
        # Note: Using trapz might be slow for large NS.
        ns1[:, m] = np.trapz(D1, scales, axis=0)
        ns2[:, m] = np.trapz(D2, scales, axis=0)

        # Calculate RotDnn for the current iteration
        PSA180_iter, _, _ = compute_rotated_spectra(T, ns1[:, m], ns2[:, m], zi, dt, theta)
        hPSArotnn[:, m] = np.percentile(PSA180_iter, nn, axis=0)

        # Calculate error metrics
        diff = np.abs(hPSArotnn[Tlocs, m] - ds[Tlocs]) / ds[Tlocs]
        meane[m] = np.mean(diff) * 100
        rmse[m] = np.linalg.norm(diff) / np.sqrt(len(Tlocs)) * 100
        log.info("Iteration %d: RMSE=%.2f%%, Misfit=%.2f%%", m, rmse[m], meane[m])

    # Select the best result (minimum RMSE)
    brloc = np.argmin(rmse)
    log.info("Best result found at iteration %d.", brloc)
    scc1_unbc = ns1[:, brloc] # Spectrally compatible component 1 (un-baseline-corrected)
    scc2_unbc = ns2[:, brloc] # Spectrally compatible component 2 (un-baseline-corrected)
    meanefin = meane[brloc]
    rmsefin = rmse[brloc]

    # --- Baseline Correction ---
    if baseline:
        scc1, cvel1, cdisp1 = baselinecorrect(scc1_unbc, t, porder=porder)
        scc2, cvel2, cdisp2 = baselinecorrect(scc2_unbc, t, porder=porder)
        # Recalculate final spectrum after correction
        PSA180_final, _, _ = compute_rotated_spectra(T, scc1, scc2, zi, dt, theta)
        PSArotnn = np.percentile(PSA180_final, nn, axis=0)
        # Re-evaluate error after baseline correction
        diff_final = np.abs(PSArotnn[Tlocs] - ds[Tlocs]) / ds[Tlocs]
        meanefin_bc = np.mean(diff_final) * 100
        rmsefin_bc = np.linalg.norm(diff_final) / np.sqrt(len(Tlocs)) * 100
        log.info("After Baseline Correction: RMSE=%.2f%%, Misfit=%.2f%%", rmsefin_bc, meanefin_bc)
        # We report the error *before* baseline correction, as that's what the iteration minimized
    else:
        log.warning("Baseline correction was skipped.")
        scc1, scc2 = scc1_unbc, scc2_unbc
        cvel1 = integrate.cumulative_trapezoid(scc1, x=t, initial=0)
        cdisp1 = integrate.cumulative_trapezoid(cvel1, x=t, initial=0)
        cvel2 = integrate.cumulative_trapezoid(scc2, x=t, initial=0)
        cdisp2 = integrate.cumulative_trapezoid(cvel2, x=t, initial=0)
        PSArotnn = hPSArotnn[:, brloc] # Use the spectrum before potential correction

    log.info("REQPYrotdnn finished. Final RMSE: %.2f%%, Misfit: %.2f%%", rmsefin, meanefin)

    results = {
        'scc1': scc1, 'scc2': scc2,
        'cvel1': cvel1, 'cvel2': cvel2,
        'cdisp1': cdisp1, 'cdisp2': cdisp2,
        'PSArotnn': PSArotnn, 'PSArotnnor': PSArotnnor,
        'T': T, 'meanefin': meanefin, 'rmsefin': rmsefin,
        'sf': sf, 'dt': dt
    }
    return results

def REQPY_single(
    s: np.ndarray,
    fs: float,
    dso: np.ndarray,
    To: np.ndarray,
    T1: float = 0.0,
    T2: float = 0.0,
    zi: float = 0.05,
    nit: int = 30,
    NS: int = 100,
    baseline: bool = True,
    porder: int = -1) -> Dict[str, Any]:
    
    """CWT based modification of a single component to match a target spectrum.

    Based on Montejo & Suarez (2013) [3]_. Modifies a single input
    acceleration time series `s` such that its response spectrum matches the
    target spectrum `dso`.

    Parameters
    ----------
    s : np.ndarray
        Seed record (acceleration time series in g's).
    fs : float
        Seed record sampling frequency (Hz).
    dso : np.ndarray
        Target/design pseudo-acceleration spectrum ordinates (g).
    To : np.ndarray
        Periods corresponding to the target spectrum ordinates `dso` (s).
        Must be sorted in ascending order.
    T1 : float, optional
        Lower bound of the period range for spectral matching (s).
        Default is 0.0, which uses the lowest period in `To`.
    T2 : float, optional
        Upper bound of the period range for spectral matching (s).
        Default is 0.0, which uses the highest period in `To`.
    zi : float, optional
        Damping ratio for response spectrum calculation. Default is 0.05 (5%).
    nit : int, optional
        Maximum number of iterations for the matching algorithm. Default is 30.
    NS : int, optional
        Number of frequency/scale values used for the CWT decomposition and
        response spectra calculation. Default is 100.
    baseline : bool, optional
        Flag to perform baseline correction on the matched record.
        Default is True.
    porder : int, optional
        Order of the polynomial used for initial detrending during baseline
        correction. Only used if `baseline=True`. Default is -1 (no detrending).

    Returns
    -------
    Dict[str, Any]
        A dictionary containing the results:
        - 'ccs' (np.ndarray): Matched acceleration time series (g).
        - 'rmsefin' (float): Final root mean squared error within matching range (%).
        - 'meanefin' (float): Final average misfit within matching range (%).
        - 'cvel' (np.ndarray): Velocity time history for matched record (vel/g).
        - 'cdespl' (np.ndarray): Displacement time history for matched record (displ/g).
        - 'PSAccs' (np.ndarray): PSA spectrum of the matched record (g).
        - 'PSAs' (np.ndarray): PSA spectrum of the original seed record (g).
        - 'T' (np.ndarray): Periods corresponding to the calculated spectra (s).
        - 'sf' (float): Initial scaling factor applied to the seed record.
        - 'dt' (float): Time step of the record (s).

    Notes
    -----
    See `REQPYrotdnn` for algorithm details. This function does *not* perform
    plotting or file saving.

    References
    ----------
    .. [3] Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm...
    """
    
    pi = np.pi
    n = np.size(s)                # number of data points in seed record
    dt = 1 / fs                     # time step
    t = np.linspace(0, (n - 1) * dt, n) # time vector
    # Frequency range for CWT decomposition depends on record duration
    FF1 = min(4 / (n * dt), 0.1); FF2 = 1 / (2 * dt)

    # Ensure target spectrum periods are sorted
    Tsortindex = np.argsort(To)
    if not np.all(Tsortindex == np.arange(len(To))):
        log.info("Target spectrum periods were not sorted. Sorting now.")
        To = To[Tsortindex]
        dso = dso[Tsortindex]

    T1, T2, FF1 = _CheckPeriodRange(T1, T2, To, FF1, FF2) # verifies period range

    # Perform Continuous Wavelet Decomposition:
    omega = pi; zeta = 0.05             # wavelet function parameters
    freqs = np.geomspace(FF2, FF1, NS)  # frequencies vector (log spaced)
    T = 1 / freqs                       # periods vector
    scales = omega / (2 * pi * freqs)   # scales vector
    C = _cwtzm(s, fs, scales, omega, zeta) # performs CWT
    log.info("Wavelet decomposition performed.")

    # Generate detail functions:
    D, sr = _getdetails(t, s, C, scales, omega, zeta) # Matrix D, reconstructed signal sr
    log.info("Detail functions generated.")

    # Response spectra from original and reconstructed signal:
    PSAs, _, _ = compute_spectrum(T, s, zi, dt)
    PSAsr, _, _ = compute_spectrum(T, sr, zi, dt)

    # Interpolate target spectrum and define matching indices:
    ds = np.interp(T, To, dso, left=np.nan, right=np.nan)
    Tlocs = np.where((T >= T1) & (T <= T2))[0]
    if len(Tlocs) == 0:
        raise ValueError("No target spectrum points found within the specified matching range.")

    # Initial scaling of record:
    sf = np.sum(ds[Tlocs]) / np.sum(PSAs[Tlocs]) # initial scaling factor
    log.info("Initial scaling factor: %.4f", sf)
    sr = sf * sr; D = sf * D

    # --- Iterative Process ---
    meane = np.zeros(nit + 1)
    rmse = np.zeros(nit + 1)
    hPSAbc = np.zeros((NS, nit + 1)) # History of PSA Before Correction
    ns = np.zeros((n, nit + 1))      # History of matched signals

    # Initial state (iteration 0)
    hPSAbc[:, 0] = sf * PSAsr # Use spectrum of scaled *reconstructed* signal
    ns[:, 0] = sr             # Store scaled reconstructed signal
    diff = np.abs(hPSAbc[Tlocs, 0] - ds[Tlocs]) / ds[Tlocs]
    meane[0] = np.mean(diff) * 100
    rmse[0] = np.linalg.norm(diff) / np.sqrt(len(Tlocs)) * 100
    log.info("Iteration 0: RMSE=%.2f%%, Misfit=%.2f%%", rmse[0], meane[0])

    factor = np.ones(NS)
    DN = D # Keep track of scaled details

    for m in range(1, nit + 1):
        log.info("Starting iteration %d of %d...", m, nit)
        # Calculate scaling factors only within the matching range
        factor[Tlocs] = ds[Tlocs] / hPSAbc[Tlocs, m - 1]

        # Apply factors to details
        DN = DN * factor[:, np.newaxis]

        # Reconstruct signal
        ns[:, m] = np.trapz(DN, scales, axis=0)

        # Calculate PSA for the current iteration
        hPSAbc[:, m], _, _ = compute_spectrum(T, ns[:, m], zi, dt)

        # Calculate error metrics
        diff = np.abs(hPSAbc[Tlocs, m] - ds[Tlocs]) / ds[Tlocs]
        meane[m] = np.mean(diff) * 100
        rmse[m] = np.linalg.norm(diff) / np.sqrt(len(Tlocs)) * 100
        log.info("Iteration %d: RMSE=%.2f%%, Misfit=%.2f%%", m, rmse[m], meane[m])

    # Select the best result
    brloc = np.argmin(rmse)
    log.info("Best result found at iteration %d.", brloc)
    sc = ns[:, brloc] # Best compatible record (un-baseline-corrected)
    meanefin = meane[brloc]
    rmsefin = rmse[brloc]

    # --- Baseline Correction ---
    if baseline:
        ccs, cvel, cdespl = baselinecorrect(sc, t, porder=porder)
        # Recalculate final spectrum after correction
        PSAccs, _, _ = compute_spectrum(T, ccs, zi, dt)
        # Re-evaluate error
        diff_final = np.abs(PSAccs[Tlocs] - ds[Tlocs]) / ds[Tlocs]
        meanefin_bc = np.mean(diff_final) * 100
        rmsefin_bc = np.linalg.norm(diff_final) / np.sqrt(len(Tlocs)) * 100
        log.info("After Baseline Correction: RMSE=%.2f%%, Misfit=%.2f%%", rmsefin_bc, meanefin_bc)
    else:
        log.warning("Baseline correction was skipped.")
        ccs = sc
        cvel = integrate.cumulative_trapezoid(ccs, x=t, initial=0)
        cdespl = integrate.cumulative_trapezoid(cvel, x=t, initial=0)
        PSAccs = hPSAbc[:, brloc] # Spectrum before potential correction

    log.info("REQPY_single finished. Final RMSE: %.2f%%, Misfit: %.2f%%", rmsefin, meanefin)

    results = {
        'ccs': ccs, 'rmsefin': rmsefin, 'meanefin': meanefin,
        'cvel': cvel, 'cdespl': cdespl,
        'PSAccs': PSAccs, 'PSAs': PSAs, # PSAs is original, unscaled spectrum
        'T': T, 'sf': sf, 'dt': dt
    }
    return results

@jit(nopython=True, cache=True)
def compute_spectrum_pw(T: np.ndarray, s: np.ndarray, zeta: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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

def compute_spectrum_fd(T: np.ndarray, s: np.ndarray, z: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Optimized Response spectra via Frequency Domain (SD, PSA, PSV only).

    Internal helper function.

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
        - PSV (np.ndarray): Pseudo-spectral velocity (value/g).
        - SD (np.ndarray): Relative spectral displacement (value/g).
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

def compute_rotated_spectra_fd(T: np.ndarray, s1: np.ndarray, s2: np.ndarray, zeta: float, dt: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates rotated response spectra via Frequency Domain.

    Internal helper function.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s1 : np.ndarray
        Acceleration time series for component 1 (g).
    s2 : np.ndarray
        Acceleration time series for component 2 (g).
    z : float
        Damping ratio.
    dt : float
        Time step (s).
    theta : np.ndarray
        Vector of angles (degrees) for rotation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Rotated PSA (num_angles x num_periods).
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
def compute_rotated_spectra_pw(T: np.ndarray, s1: np.ndarray, s2: np.ndarray, zeta: float, dt: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculates rotated response spectra via Piecewise (time-domain).

    Internal helper function.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s1 : np.ndarray
        Acceleration time series for component 1 (g).
    s2 : np.ndarray
        Acceleration time series for component 2 (g).
    z : float
        Damping ratio (should be < 1).
    dt : float
        Time step (s).
    theta : np.ndarray
        Vector of angles (degrees) for rotation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Rotated PSA (num_angles x num_periods).
        - PSV (np.ndarray): Rotated PSV.
        - SD (np.ndarray): Rotated SD.
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
    nn: int) -> Tuple[np.ndarray, np.ndarray]:
    
    """Computes rotated and RotDnn (e.g., RotD100) spectra from two components.

    This is a primary analysis function.

    Parameters
    ----------
    s1 : np.ndarray
        Acceleration series in the first orthogonal horizontal direction (g).
    s2 : np.ndarray
        Acceleration series in the second orthogonal horizontal direction (g).
    dt : float
        Time step (s).
    zi : float
        Damping ratio for spectra calculation.
    T : np.ndarray
        Periods at which to calculate the spectra (s).
    nn : int
        Percentile for the RotDnn calculation (e.g., 100 for RotD100).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        - PSArotnn (np.ndarray): Vector containing the PSA RotDnn spectrum (g).
        - PSA180 (np.ndarray): Matrix (num_angles x num_periods) containing
          the PSA spectrum at angles from 0 to 179 degrees (g).
    """
    
    n1 = np.size(s1); n2 = np.size(s2); n = min(n1, n2)
    if n1 != n2:
        warnings.warn(f"Input records for rotdnn have different lengths ({n1} vs {n2}). Truncating.")
    s1 = s1[:n]; s2 = s2[:n]
    theta = np.arange(0, 180, 1) # Angles from 0 to 179 degrees
    PSA180, _, _ = compute_rotated_spectra(T, s1, s2, zi, dt, theta)
    PSArotnn = np.percentile(PSA180, nn, axis=0)
    return PSArotnn, PSA180

def compute_spectrum(T: np.ndarray, s: np.ndarray, z: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatcher: selects response spectrum algorithm based on damping.

    Internal helper function. Uses Frequency Domain for z >= 2%, Piecewise otherwise.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods to calculate spectrum (s).
    s : np.ndarray
        Input acceleration time series (g).
    z : float
        Damping ratio.
    dt : float
        Time step of the acceleration series (s).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Pseudo-spectral acceleration (g).
        - PSV (np.ndarray): Pseudo-spectral velocity (value/g).
        - SD (np.ndarray): Spectral displacement (value/g).
    """
    if z >= 0.03:
        log.debug("Using Frequency Domain (FD) method for spectrum calculation (z>=3%).")
        return compute_spectrum_fd(T, s, z, dt) 
    else:
        log.debug("Using Piecewise (PW) method for spectrum calculation (z<3%).")
        return compute_spectrum_pw(T, s, z, dt)

def compute_rotated_spectra(T: np.ndarray, s1: np.ndarray, s2: np.ndarray, z: float, dt: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Dispatcher for rotated response spectrum calculation.

    Internal helper function. Selects FD or PW based on damping.

    Parameters
    ----------
    T : np.ndarray
        Vector of periods (s).
    s1 : np.ndarray
        Acceleration time series for component 1 (g).
    s2 : np.ndarray
        Acceleration time series for component 2 (g).
    z : float
        Damping ratio.
    dt : float
        Time step (s).
    theta : np.ndarray
        Vector of angles (degrees) for rotation.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - PSA (np.ndarray): Rotated PSA (num_angles x num_periods).
        - PSV (np.ndarray): Rotated PSV.
        - SD (np.ndarray): Rotated SD.
    """
    if z >= 0.03:
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
    Calculates PSD and returns 6 values for backward compatibility.
    Returns: freqs, psd, strong_duration, arias_intensity, freqs_smooth, psd_smooth
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
    
    r"""Calculates the Fourier Amplitude Spectrum (FAS) of an earthquake time-series.

    The FAS is calculated over the entire signal duration after applying
    standard preprocessing steps (detrending, windowing). The result can be
    optionally smoothed and downsampled.

    Parameters
    ----------
    accel_series : numpy.ndarray
        The 1D acceleration time-series data.
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    tukey_alpha : float, optional
        Tukey window shape parameter, between 0 (rectangular) and 1 (Hann).
        Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points (nFFT) for the FFT.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated FAS. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the smoothing algorithm.
    downsample_freqs : numpy.ndarray or None, optional
        An array of frequencies at which to calculate the smoothed spectrum.
    prefer_pykooh : bool, optional
        If True (default) and `smoothing_method` is 'konno_ohmachi', the `pykooh`
        library will be used if installed. If False, the in-house implementation
        will be used instead.

    Returns
    -------
    freqs : numpy.ndarray
        The array of frequencies for the FAS.
    fas : numpy.ndarray
        The one-sided Fourier Amplitude Spectrum.
    freqs_smooth : numpy.ndarray or None
        The frequency vector for the smoothed FAS. None if no smoothing.
    fas_smooth : numpy.ndarray or None
        The smoothed, one-sided FAS. None if no smoothing.
        
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
    
    r"""Calculates the Fourier Amplitude Spectrum (FAS) for rotated horizontal components.

    This function takes two orthogonal horizontal time-series, applies identical
    preprocessing, and then calculates the FAS for a suite of rotation angles in the
    frequency domain.

    Parameters
    ----------
    accel_series_1 : numpy.ndarray
        The 1D acceleration time-series for the first horizontal component.
    accel_series_2 : numpy.ndarray
        The 1D acceleration time-series for the second horizontal component.
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
    freqs : numpy.ndarray
        The 1D array of frequencies for the FAS.
    fas_rotated : numpy.ndarray
        The 2D array of rotated FAS values. Shape is (rotation_angles, num_freqs).
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
    
    r"""Calculates the Power Spectral Density (PSD) for rotated horizontal components.

    This function rotates the two orthogonal time-series through a suite of angles
    and calculates the strong motion duration and PSD for each rotated component
    independently.

    Parameters
    ----------
    accel_series_1 : numpy.ndarray
        The 1D acceleration time-series for the first horizontal component.
    accel_series_2 : numpy.ndarray
        The 1D acceleration time-series for the second horizontal component.
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    rotation_angles : int, optional
        The number of rotation angles to compute, evenly spaced from 0 to 180
        degrees (exclusive of 180). Default is 180.
    duration_percent : tuple(float, float) or None, optional
        The start and end percentage of Arias Intensity defining the strong
        motion duration. If None, the entire signal is used. Default is (5, 75).
    tukey_alpha : float, optional
        Tukey window shape parameter, between 0 (rectangular) and 1 (Hann).
        Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT. The same nFFT
        is used for all angles for consistency. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal. Default is 'linear'.

    Returns
    -------
    freqs : numpy.ndarray
        The 1D array of frequencies for the PSD.
    psd_rotated : numpy.ndarray
        The 2D array of rotated PSD values. Shape is (rotation_angles, num_freqs).
        
    Notes
    -----
    - To capture the azimuthal dependency of ground motion, this function rotates
      the time-series first and then calculates the strong motion duration for
      each angle independently.
    - **Warning**: This approach is computationally intensive as it requires
      recalculating the duration and FFT for every rotation angle. For a large
      number of angles, this can be slow.
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
    """Computes orientation-independent RotDnn FAS spectra for multiple percentiles.

    This function calculates the FAS for a suite of rotation angles and then
    computes spectra for each specified percentile (e.g., RotD50, RotD100). It
    offers two distinct smoothing workflows.

    Parameters
    ----------
    accel_series_1 : numpy.ndarray
        The 1D acceleration time-series for the first horizontal component.
    accel_series_2 : numpy.ndarray
        The 1D acceleration time-series for the second horizontal component.
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    percentiles : list or tuple of float, optional
        A container of percentiles to compute from the rotated spectra.
        Default is (50, 100).
    tukey_alpha : float, optional
        Tukey window shape parameter. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated FAS. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the smoothing algorithm.
    downsample_freqs : numpy.ndarray or None, optional
        An array of frequencies to which the final spectra will be returned.
    smooth_last : bool, optional
        Controls the smoothing order.
        - `True` (default): First calculates the raw RotD spectra, then smooths the results.
        - `False`: First smooths every rotated spectrum, then calculates
          the percentiles from the smoothed results.
    prefer_pykooh : bool, optional
        If `True` (default), the `pykooh` library will be used for Konno-Ohmachi
        smoothing if available.

    Returns
    -------
    output_freqs : numpy.ndarray
        The frequency vector for the output spectra.
    fas_rotdnn_smooth : dict
        A dictionary where keys are the requested percentiles and values are the
        corresponding smoothed RotDnn FAS arrays.
    fas_rotdnn_raw_interp : dict
        A dictionary where keys are the requested percentiles and values are the
        corresponding raw (unsmoothed) RotDnn FAS arrays, interpolated to the
        `output_freqs`.

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).

    Examples
    --------
    >>> # Calculate RotD50 and RotD100 using default settings
    >>> freqs, fas_smooth, fas_raw = calculate_fas_rotDnn(
    ...     s1, s2, sample_rate
    ... )
    >>>
    >>> # Access the results using the percentile keys
    >>> rotd50 = fas_smooth[50]
    >>> rotd100 = fas_smooth[100]
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
    
    r"""Computes orientation-independent RotDnn PSD spectra for multiple percentiles.

    This function calculates the PSD for a suite of rotation angles and then
    computes spectra for each specified percentile (e.g., RotD50, RotD100). It
    offers two distinct smoothing workflows.

    Parameters
    ----------
    accel_series_1 : numpy.ndarray
        The 1D acceleration time-series for the first horizontal component.
    accel_series_2 : numpy.ndarray
        The 1D acceleration time-series for the second horizontal component.
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    percentiles : list or tuple of float, optional
        A container of percentiles to compute from the rotated spectra.
        Default is (50, 100).
    duration_percent : tuple(float, float) or None, optional
        The start and end percentage of Arias Intensity defining the strong
        motion duration. If None, the entire signal is used. Default is (5, 75).
    tukey_alpha : float, optional
        Tukey window shape parameter. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated PSD. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the smoothing algorithm.
    downsample_freqs : numpy.ndarray or None, optional
        An array of frequencies to which the final spectra will be returned.
    smooth_last : bool, optional
        Controls the smoothing order.
        - `True` (default): First calculates the raw RotD spectra, then smooths the results.
        - `False`: First smooths every rotated spectrum, then calculates
          the percentiles from the smoothed results.
    prefer_pykooh : bool, optional
        If `True` (default), the `pykooh` library will be used for Konno-Ohmachi
        smoothing if available.

    Returns
    -------
    output_freqs : numpy.ndarray
        The frequency vector for the output spectra.
    psd_rotdnn_smooth : dict
        A dictionary where keys are the requested percentiles and values are the
        corresponding smoothed RotDnn PSD arrays.
    psd_rotdnn_raw_interp : dict
        A dictionary where keys are the requested percentiles and values are the
        corresponding raw (unsmoothed) RotDnn PSD arrays, interpolated to the
        `output_freqs`.

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).

    Examples
    --------
    >>> # Calculate RotD50 and RotD100 using default settings
    >>> freqs, psd_smooth, psd_raw = calculate_psd_rotDnn(s1, s2, sample_rate)
    >>>
    >>> # Access the results using the percentile keys
    >>> rotd50_psd = psd_smooth[50]
    >>> rotd100_psd = psd_smooth[100]
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
    
    r"""Computes the Effective Amplitude Spectrum (EAS) for two horizontal components.

    This function calculates the FAS for two orthogonal components, then combines
    them into a single EAS. It offers two distinct smoothing workflows.

    Parameters
    ----------
    accel_series_1 : numpy.ndarray
        The 1D acceleration time-series for the first horizontal component.
    accel_series_2 : numpy.ndarray
        The 1D acceleration time-series for the second horizontal component.
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    tukey_alpha : float, optional
        Tukey window shape parameter. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT. Default is 'nextpow2'.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated FAS. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the smoothing algorithm.
    downsample_freqs : numpy.ndarray or None, optional
        An array of frequencies to which the final spectra will be returned.
    smooth_last : bool, optional
        Controls the smoothing order.
        - `True` (default): First calculates the raw EAS, then smooths the result.
        - `False`: First smooths each component's FAS, then calculates
          the EAS from the smoothed results.
    prefer_pykooh : bool, optional
        If `True` (default), the `pykooh` library will be used for Konno-Ohmachi
        smoothing if available.

    Returns
    -------
    output_freqs : numpy.ndarray
        The frequency vector for the output spectra.
    eas_smooth : numpy.ndarray
        The smoothed Effective Amplitude Spectrum.
    eas_raw_interp : numpy.ndarray
        The raw (unsmoothed) EAS, interpolated to the `output_freqs`.

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
    
    r"""Computes the Effective Power Spectrum (EPSD) for two horizontal components.

    This function calculates the PSD for two orthogonal components, then combines
    them into a single EPSD. It offers two distinct smoothing workflows.

    Parameters
    ----------
    accel_series_1 : numpy.ndarray
        The 1D acceleration time-series for the first horizontal component.
    accel_series_2 : numpy.ndarray
        The 1D acceleration time-series for the second horizontal component.
    sample_rate : float
        The sampling frequency of the time-series in Hz.
    duration_percent : tuple(float, float) or None, optional
        The start and end percentage of Arias Intensity defining the strong
        motion duration for each component. Default is (5, 75).
    tukey_alpha : float, optional
        Tukey window shape parameter. Default is 0.1.
    nfft_method : {'nextpow2', 'same'} or int, optional
        Method for determining the number of points for the FFT. Default is 'nextpow2'.
    nfft_base : {'total', 'strong'}, optional
        Determines if nFFT is based on 'total' or 'strong' motion length.
    detrend_method : {'linear', 'constant'} or None, optional
        Method for detrending the signal. Default is 'linear'.
    smoothing_method : {'none', 'konno_ohmachi', 'variable_window'}, optional
        Method for smoothing the calculated PSD. Default is 'konno_ohmachi'.
    smoothing_coeff : float, optional
        Coefficient for the smoothing algorithm.
    downsample_freqs : numpy.ndarray or None, optional
        An array of frequencies to which the final spectra will be returned.
    smooth_last : bool, optional
        Controls the smoothing order.
        - `True` (default): First calculates the raw EPSD, then smooths the result.
        - `False`: First smooths each component's PSD, then calculates
          the EPSD from the smoothed results.
    prefer_pykooh : bool, optional
        If `True` (default), the `pykooh` library will be used for Konno-Ohmachi
        smoothing if available.

    Returns
    -------
    output_freqs : numpy.ndarray
        The frequency vector for the output spectra.
    epsd_smooth : numpy.ndarray
        The smoothed Effective Power Spectrum.
    epsd_raw_interp : numpy.ndarray
        The raw (unsmoothed) EPSD, interpolated to the `output_freqs`.

    Raises
    ------
    ValueError
        If `downsample_freqs` contains values outside the valid range of the
        calculated Fourier frequencies (0 to Nyquist frequency).
        
    Notes
    -----
    The Effective Power Spectrum (EPSD) is calculated as the root mean square
    of the two individual component PSDs: $EPSD = \sqrt{0.5 * (PSD_1^2 + PSD_2^2)}$.
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

def baselinecorrect(
    sc: np.ndarray,
    t: np.ndarray,
    porder: int = -1,
    imax: int = 80,
    tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    
    """Performs baseline correction iteratively calling _basecorr.

    This is the main user-facing baseline correction function. It iteratively
    increases the correction time window if the initial correction fails.

    Parameters
    ----------
    sc : np.ndarray
        Uncorrected acceleration time series (g).
    t : np.ndarray
        Time vector corresponding to `sc` (s).
    porder : int, optional
        Order of the polynomial used for initial detrending.
        Default is -1 (no detrending).
    imax : int, optional
        Maximum number of iterations for the correction algorithm within
        `_basecorr`. Default is 80.
    tol : float, optional
        Tolerance criterion (as a percentage of the maximum absolute value)
        for final velocity and displacement. Default is 0.01 (1%).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        - ccs (np.ndarray): Corrected acceleration time series (g).
        - cvel (np.ndarray): Corresponding velocity time history (vel/g).
        - cdespl (np.ndarray): Corresponding displacement time history (displ/g).

    References
    ----------
    .. [4] Suarez, L. E., & Montejo, L. A. (2007). Applications...
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
    s: np.ndarray, t: np.ndarray, ival: float = 5, fval: float = 75
    ) -> Tuple[float, np.ndarray, float, float, float]:
    
    r"""Calculate the significant duration and Arias Intensity of a time series.

    This function determines the time interval during which the central portion
    of a signal's energy is released. It is based on the cumulative sum of the
    squared acceleration, known as the Arias Intensity (AI).

    Parameters
    ----------
    s : numpy.ndarray
        Acceleration time-history as a 1D NumPy array.
    t : numpy.ndarray
        Time vector corresponding to the acceleration data. Must be the same
        length as `s`.
    ival : float, optional
        Initial percentage of Arias Intensity to define the start of the
        significant duration. The default is 5.
    fval : float, optional
        Final percentage of Arias Intensity to define the end of the
        significant duration. The default is 75.

    Returns
    -------
    sd : float
        Significant duration (e.g., $D_{5-75}$), calculated as `t2 - t1`.
    AIcumnorm : numpy.ndarray
        The normalized cumulative Arias Intensity, ranging from 0 to 1.
    AI : float
        The total Arias Intensity. Note: This is the raw integral of the
        squared acceleration, $\int a(t)^2 dt$, without the standard
        pre-factor of $\pi / (2g)$.
    t1 : float
        The time at which the cumulative Arias Intensity reaches `ival` percent.
    t2 : float
        The time at which the cumulative Arias Intensity reaches `fval` percent.

    Raises
    ------
    ValueError
        If `ival` or `fval` are outside the [0, 100] range, if `fval` is not
        greater than `ival`, or if a valid positive duration cannot be calculated.

    See Also
    --------
    scipy.integrate.cumulative_trapezoid : The function used to compute the
                                           cumulative integral.
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

def pga_correction(targetPGA: float, t: np.ndarray, s: np.ndarray, maxit: int = 1000) -> np.ndarray:
    """
    Performs localized PGA correction by identifying the peak cycle and scaling it.
    Reference: Montejo 2025.
    """
    from scipy.signal import find_peaks
    
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
    
    r"""Performs logarithmic interpolation in 1D with options for out-of-bounds values.

    This function interpolates in log-space, which is equivalent to assuming an
    exponential relationship between points. It is particularly useful for data
    that appears linear on a semi-log plot.

    Parameters
    ----------
    x : numpy.ndarray
        The x-coordinates at which to evaluate the interpolated values.
    xp : numpy.ndarray
        A 1D array of the x-coordinates of the data points. Must be sorted in
        increasing order.
    yp : numpy.ndarray
        A 1D array of the y-coordinates of the data points. Must be the same
        size as `xp`.
    epsilon : float, optional
        A small value added to `yp` to avoid `log(0)` errors when `yp`
        contains zeros. The default is 1e-20.
    out_of_bounds : {'nan', 'clip'}, optional
        Specifies how to handle x-values outside the range of `xp`.
        - 'nan' (default): Out-of-bounds values are replaced with `np.nan`.
        - 'clip': The output is clipped to the nearest endpoint value from `yp`.

    Returns
    -------
    y : numpy.ndarray
        The interpolated values, which has the same shape as `x`.

    Raises
    ------
    ValueError
        If `out_of_bounds` is not one of the allowed strings ('nan' or 'clip').

    See Also
    --------
    numpy.interp : The underlying linear interpolation function.

    Examples
    --------
    >>> xp = np.array([1, 2, 3])
    >>> yp = np.array([10, 100, 1000]) # Exponential growth
    >>> x_eval = np.array([0.5, 1.5, 2.5, 3.5])
    >>>
    >>> # Interpolate with 'nan' for out-of-bounds values
    >>> log_interp(x_eval, xp, yp, out_of_bounds='nan')
    array([         nan,  31.6227766, 316.22776602,          nan])
    >>>
    >>> # Interpolate with 'clip' for out-of-bounds values
    >>> log_interp(x_eval, xp, yp, out_of_bounds='clip')
    array([  10.        ,   31.6227766 ,  316.22776602, 1000.        ])

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

def get_log_freqs(fmin: float = 0.1, fmax: float = 100.0, pts_per_decade: int = 100) -> np.ndarray:
    r"""Generates a logarithmically spaced array of frequencies.

    This function creates a NumPy array of frequency values spaced evenly on a
    logarithmic scale, which is useful for tasks like frequency response
    analysis or plotting on a log axis.

    Parameters
    ----------
    fmin : float, optional
        The minimum frequency value in the array. Must be a positive number.
        Default is 0.1.
    fmax : float, optional
        The maximum frequency value in the array. Must be greater than fmin.
        Default is 100.0.
    pts_per_decade : int, optional
        The number of points to generate for each decade (e.g., from 1 to 10,
        10 to 100). Default is 100.

    Returns
    -------
    numpy.ndarray
        A 1D NumPy array of logarithmically spaced frequency values, including
        both `fmin` and `fmax`.

    Examples
    --------
    >>> get_log_freqs(fmin=1, fmax=100, pts_per_decade=5)
    array([  1.        ,   1.58489319,   2.51188643,   3.98107171,
             6.30957344,  10.        ,  15.84893192,  25.11886432,
            39.81071706,  63.09573445, 100.        ])

    """
    # Calculate the number of decades the frequency range spans
    num_decades = np.log10(fmax / fmin)

    # Determine the total number of points needed
    total_points = int(pts_per_decade * num_decades) + 1

    # Generate the logarithmically spaced array
    return np.geomspace(fmin, fmax, num=total_points)

def plot_single_results(
    results: Dict[str, Any],
    s_orig: np.ndarray,
    target_spec: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    T1: float = 0.0,
    T2: float = 0.0,
    xlim_min: Optional[float] = None,
    xlim_max: Optional[float] = None) -> Tuple[plt.Figure, plt.Figure]:
    
    """Generates standard plots for single-component matching results.

    Parameters
    ----------
    results : Dict[str, Any]
        The dictionary returned by `REQPY_single`.
    s_orig : np.ndarray
        The original (unscaled) seed acceleration time series (g).
    target_spec : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Tuple containing (Target_Periods, Target_PSA) for plotting the target.
        If None, the target spectrum is not plotted. Default is None.
    T1 : float, optional
        Lower bound of the matching period range (for highlighting). Default 0.0.
    T2 : float, optional
        Upper bound of the matching period range (for highlighting). Default 0.0.
    xlim_min : Optional[float], optional
        Minimum period limit for the x-axis of the spectra plot.
        Defaults to the minimum period computed (results['T'].min()).
    xlim_max : Optional[float], optional
        Maximum period limit for the x-axis of the spectra plot.
        Defaults to the maximum period computed (results['T'].max()).

    Returns
    -------
    Tuple[plt.Figure, plt.Figure]
        - fig_hist: Figure object for the time history plots.
        - fig_spec: Figure object for the spectra plots.
    """
    # --- Extract data from results dictionary ---
    ccs = results.get('ccs', np.array([]))
    cvel = results.get('cvel', np.array([]))
    cdespl = results.get('cdespl', np.array([]))
    PSAccs = results.get('PSAccs', np.array([]))
    PSAs_orig = results.get('PSAs', np.array([])) # Original spectrum
    T = results.get('T', np.array([0.01, 10.0])) # Periods for calculated spectra
    sf = results.get('sf', 1.0) # Initial scaling factor
    dt = results.get('dt', 0.01) # Time step
    # Handle potential missing keys gracefully if needed
    if ccs.size == 0 or T.size == 0:
        msg = "Results dictionary is missing essential data ('ccs' or 'T'). Cannot plot."
        log.error(msg)
        raise ValueError(msg)
    # --- End data extraction ---

    n = len(ccs)
    t = np.linspace(0, (n - 1) * dt, n)

    # Calculate scaled original histories for comparison
    s_scaled = s_orig[:n] * sf # Ensure s_orig matches length if needed
    vel_scaled = integrate.cumulative_trapezoid(s_scaled, x=t, initial=0)
    despl_scaled = integrate.cumulative_trapezoid(vel_scaled, x=t, initial=0)
    # Ensure PSAs_orig corresponds to T from results
    if len(PSAs_orig) != len(T):
        warnings.warn("Length mismatch between original spectrum PSAs and periods T. Recalculating original spec.")
        PSAs_orig, _, _ = compute_spectrum(T, s_orig[:n], results.get('zi', 0.05), dt) # Use damping from results if avail.

    PSAs_scaled = PSAs_orig * sf # Scaled original spectrum

    mpl.rcParams['font.size'] = 9
    mpl.rcParams['legend.frameon'] = False

    # --- Time History Plot ---
    # Ensure calculated arrays are not empty before finding max
    acc_vals = np.concatenate(([0], np.abs(s_scaled), np.abs(ccs)))
    vel_vals = np.concatenate(([0], np.abs(vel_scaled), np.abs(cvel)))
    disp_vals = np.concatenate(([0], np.abs(despl_scaled), np.abs(cdespl)))

    alim = 1.05 * np.max(acc_vals)
    vlim = 1.05 * np.max(vel_vals)
    dlim = 1.05 * np.max(disp_vals)

    fig_hist, axs_hist = plt.subplots(3, 1, figsize=(6.5, 6.5), sharex=True)

    axs_hist[0].plot(t, s_scaled, lw=1, color='cornflowerblue', label='Scaled Seed')
    axs_hist[0].plot(t, ccs, lw=1, color='salmon', label='Matched')
    axs_hist[0].set_ylim(-alim, alim)
    axs_hist[0].set_ylabel('Acc. [g]')
    axs_hist[0].legend(loc='upper right')
    axs_hist[0].grid(True, linestyle=':', alpha=0.7)

    axs_hist[1].plot(t, vel_scaled, lw=1, color='cornflowerblue')
    axs_hist[1].plot(t, cvel, lw=1, color='salmon')
    axs_hist[1].set_ylim(-vlim, vlim)
    axs_hist[1].set_ylabel('Vel./g')
    axs_hist[1].grid(True, linestyle=':', alpha=0.7)

    axs_hist[2].plot(t, despl_scaled, lw=1, color='cornflowerblue')
    axs_hist[2].plot(t, cdespl, lw=1, color='salmon')
    axs_hist[2].set_ylabel('Displ./g')
    axs_hist[2].set_xlabel('Time [s]')
    axs_hist[2].set_ylim(-dlim, dlim)
    axs_hist[2].grid(True, linestyle=':', alpha=0.7)

    fig_hist.tight_layout()

    # --- Spectra Plot ---
    fig_spec, ax_spec = plt.subplots(figsize=(6.5, 6.5))

    # Determine y-limit first
    all_psa_vals = np.concatenate(([0], PSAs_scaled, PSAccs))
    if target_spec is not None:
        To_targ, dso_targ = target_spec
        all_psa_vals = np.concatenate((all_psa_vals, dso_targ))
    limy = 1.06 * np.nanmax(all_psa_vals) # Use nanmax

    # Plot target spectrum if provided
    if target_spec is not None:
        ax_spec.semilogx(To_targ, dso_targ, color='darkgray', lw=2, label='Target')

    # Highlight matching range
    # Ensure T1 and T2 are within the bounds of T for plotting range fill
    # Use computed T min/max as fallback for fill range if T1/T2 are defaults
    T_min_comp = T.min()
    T_max_comp = T.max()
    plot_T1_fill = T1 if T1 > T_min_comp else T_min_comp
    plot_T2_fill = T2 if T2 > 0 and T2 < T_max_comp else T_max_comp
    if T1 <= 1e-9 and T2 <= 1e-9: # Default case uses full computed range for fill
        plot_T1_fill, plot_T2_fill = T_min_comp, T_max_comp

    # Check if fill range is valid before plotting
    if plot_T1_fill < plot_T2_fill:
        auxx = [plot_T1_fill, plot_T1_fill, plot_T2_fill, plot_T2_fill, plot_T1_fill]
        auxy = [0, limy, limy, 0, 0]
        ax_spec.fill_between(auxx, auxy, color='silver', alpha=0.4, label='Match Range')
        ax_spec.plot(auxx, auxy, color='silver', alpha=1, lw=0.5) # Border for range
    else:
        warnings.warn("Invalid period range for highlighting (T1 >= T2). Skipping fill.")


    # Plot computed spectra
    ax_spec.semilogx(T, PSAs_orig, color='blueviolet', lw=1, label='Original Seed')
    ax_spec.semilogx(T, PSAs_scaled, color='cornflowerblue', lw=1, label='Scaled Seed')
    ax_spec.semilogx(T, PSAccs, color='salmon', lw=1, label='Matched')

    ax_spec.set_xlabel('Period T [s]')
    ax_spec.set_ylabel('PSA [g]')
    ax_spec.set_ylim(bottom=0, top=limy)
    ax_spec.grid(True, which='both', linestyle=':', alpha=0.7)

    # --- Apply x-axis limits ---
    x_min = xlim_min if xlim_min is not None else T_min_comp
    x_max = xlim_max if xlim_max is not None else T_max_comp
    # Ensure min < max
    if x_min < x_max:
        ax_spec.set_xlim(x_min, x_max)
    else:
        warnings.warn(f"Invalid xlim provided (min={x_min} >= max={x_max}). Using default limits.")
        ax_spec.set_xlim(T_min_comp, T_max_comp)
    # --- End x-axis limits ---

    # Adjust legend position
    ax_spec.legend(ncol=3, bbox_to_anchor=(0.5, 1.02), loc='lower center')
    fig_spec.tight_layout(rect=(0, 0, 1, 0.95)) # Adjust layout

    return fig_hist, fig_spec

def plot_rotdnn_results(
    results: Dict[str, Any],
    s1_orig: np.ndarray,
    s2_orig: np.ndarray,
    target_spec: Optional[Tuple[np.ndarray, np.ndarray]] = None,
    T1: float = 0.0,
    T2: float = 0.0,
    xlim_min: Optional[float] = None,
    xlim_max: Optional[float] = None
) -> Tuple[plt.Figure, plt.Figure]:
    
    """Generates standard plots for RotDnn matching results.

    Parameters
    ----------
    results : Dict[str, Any]
        The dictionary returned by `REQPYrotdnn`.
    s1_orig : np.ndarray
        The original (unscaled) seed acceleration for component 1 (g).
    s2_orig : np.ndarray
        The original (unscaled) seed acceleration for component 2 (g).
    target_spec : Optional[Tuple[np.ndarray, np.ndarray]], optional
        Tuple containing (Target_Periods, Target_PSA) for plotting the target.
        If None, the target spectrum is not plotted. Default is None.
    T1 : float, optional
        Lower bound of the matching period range (for highlighting). Default 0.0.
    T2 : float, optional
        Upper bound of the matching period range (for highlighting). Default 0.0.
    xlim_min : Optional[float], optional
        Minimum period limit for the x-axis of the spectra plot.
        Defaults to the minimum period computed (results['T'].min()).
    xlim_max : Optional[float], optional
        Maximum period limit for the x-axis of the spectra plot.
        Defaults to the maximum period computed (results['T'].max()).

    Returns
    -------
    Tuple[plt.Figure, plt.Figure]
        - fig_hist: Figure object for the time history plots.
        - fig_spec: Figure object for the spectra plots.
    """
    # --- Extract data from results dictionary ---
    scc1 = results.get('scc1', np.array([]))
    scc2 = results.get('scc2', np.array([]))
    cvel1 = results.get('cvel1', np.array([]))
    cvel2 = results.get('cvel2', np.array([]))
    cdisp1 = results.get('cdisp1', np.array([]))
    cdisp2 = results.get('cdisp2', np.array([]))
    PSArotnn = results.get('PSArotnn', np.array([]))
    PSArotnnor = results.get('PSArotnnor', np.array([])) # Original RotDnn
    T = results.get('T', np.array([0.01, 10.0])) # Periods for calculated spectra
    sf = results.get('sf', 1.0) # Initial overall scaling factor
    dt = results.get('dt', 0.01) # Time step
    nn_val = results.get('nn', 'nn') # Get percentile value if present in dict

    # Handle potential missing keys gracefully
    if scc1.size == 0 or scc2.size == 0 or T.size == 0:
        msg = "Results dictionary is missing essential data ('scc1', 'scc2', or 'T'). Cannot plot."
        log.error(msg)
        raise ValueError(msg)
    # --- End data extraction ---

    n = len(scc1) # Length of the matched components
    t = np.linspace(0, (n - 1) * dt, n)

    # Calculate scaled original histories for comparison
    # Ensure original signals are truncated to the same length 'n' used in calcs
    s1_scaled = s1_orig[:n] * sf
    s2_scaled = s2_orig[:n] * sf
    v1_scaled = integrate.cumulative_trapezoid(s1_scaled, x=t, initial=0)
    d1_scaled = integrate.cumulative_trapezoid(v1_scaled, x=t, initial=0)
    v2_scaled = integrate.cumulative_trapezoid(s2_scaled, x=t, initial=0)
    d2_scaled = integrate.cumulative_trapezoid(v2_scaled, x=t, initial=0)
    
    # Ensure original RotDnn corresponds to T from results
    if len(PSArotnnor) != len(T):
        warnings.warn(f"Length mismatch between original RotDnn (len {len(PSArotnnor)}) and periods T (len {len(T)}). Recalculating.")
        # Recalculate original RotDnn using the same T
        PSArotnnor, _ = rotdnn(s1_orig[:n], s2_orig[:n], dt, results.get('zi', 0.05), T, nn_val)

    PSArotnn_scaled = PSArotnnor * sf # Scaled original RotDnn

    mpl.rcParams['font.size'] = 9
    mpl.rcParams['legend.frameon'] = False

    # --- Time History Plot ---
    # (Slightly more robust max calculation)
    acc_vals = np.concatenate(([0], np.abs(s1_scaled), np.abs(scc1), np.abs(s2_scaled), np.abs(scc2)))
    vel_vals = np.concatenate(([0], np.abs(v1_scaled), np.abs(cvel1), np.abs(v2_scaled), np.abs(cvel2)))
    disp_vals = np.concatenate(([0], np.abs(d1_scaled), np.abs(cdisp1), np.abs(d2_scaled), np.abs(cdisp2)))

    alim = 1.05 * np.nanmax(acc_vals) if acc_vals.size > 0 else 1.0
    vlim = 1.05 * np.nanmax(vel_vals) if vel_vals.size > 0 else 1.0
    dlim = 1.05 * np.nanmax(disp_vals) if disp_vals.size > 0 else 1.0

    fig_hist, axs_hist = plt.subplots(3, 2, figsize=(6.5, 5), sharex=True, sharey='row')

    # Component 1
    axs_hist[0, 0].plot(t, s1_scaled, lw=1, color='cornflowerblue')
    axs_hist[0, 0].plot(t, scc1, lw=1, color='salmon')
    axs_hist[0, 0].set_ylim(-alim, alim)
    axs_hist[0, 0].set_ylabel('Acc. [g]')
    axs_hist[0, 0].set_title('Component 1')
    axs_hist[0, 0].grid(True, linestyle=':', alpha=0.7)

    axs_hist[1, 0].plot(t, v1_scaled, lw=1, color='cornflowerblue')
    axs_hist[1, 0].plot(t, cvel1, lw=1, color='salmon')
    axs_hist[1, 0].set_ylim(-vlim, vlim)
    axs_hist[1, 0].set_ylabel('Vel./g')
    axs_hist[1, 0].grid(True, linestyle=':', alpha=0.7)

    axs_hist[2, 0].plot(t, d1_scaled, lw=1, color='cornflowerblue')
    axs_hist[2, 0].plot(t, cdisp1, lw=1, color='salmon')
    axs_hist[2, 0].set_ylim(-dlim, dlim)
    axs_hist[2, 0].set_ylabel('Displ./g')
    axs_hist[2, 0].set_xlabel('Time [s]')
    axs_hist[2, 0].grid(True, linestyle=':', alpha=0.7)

    # Component 2
    axs_hist[0, 1].plot(t, s2_scaled, lw=1, color='cornflowerblue')
    axs_hist[0, 1].plot(t, scc2, lw=1, color='salmon')
    axs_hist[0, 1].set_title('Component 2')
    axs_hist[0, 1].grid(True, linestyle=':', alpha=0.7)

    axs_hist[1, 1].plot(t, v2_scaled, lw=1, color='cornflowerblue')
    axs_hist[1, 1].plot(t, cvel2, lw=1, color='salmon')
    axs_hist[1, 1].grid(True, linestyle=':', alpha=0.7)

    axs_hist[2, 1].plot(t, d2_scaled, lw=1, color='cornflowerblue', label='Scaled Seed')
    axs_hist[2, 1].plot(t, cdisp2, lw=1, color='salmon', label='Matched')
    axs_hist[2, 1].set_xlabel('Time [s]')
    axs_hist[2, 1].grid(True, linestyle=':', alpha=0.7)

    # Add legend below plots
    handles, labels = axs_hist[2, 1].get_legend_handles_labels()
    fig_hist.legend(handles, labels, loc='lower center', ncol=2, bbox_to_anchor=(0.5, 0))
    fig_hist.tight_layout(h_pad=0.3, w_pad=0.3, rect=(0, 0.05, 1, 0.96)) # Adjust rect

    # --- Spectra Plot ---
    fig_spec, ax_spec = plt.subplots(figsize=(6.5, 6.5))

    # Determine y-limit first
    all_psa_vals = np.concatenate(([0], PSArotnn_scaled, PSArotnn))
    if target_spec is not None:
        To_targ, dso_targ = target_spec
        all_psa_vals = np.concatenate((all_psa_vals, dso_targ))
    limy = 1.06 * np.nanmax(all_psa_vals)
    if limy == 0 or not np.isfinite(limy): limy = 1.0 # Handle case where all values are zero/nan

    # Plot target spectrum
    if target_spec is not None:
        To_targ, dso_targ = target_spec
        ax_spec.semilogx(To_targ, dso_targ, color='darkgray', lw=2, label='Target')

    # Highlight matching range
    T_min_comp = T.min()
    T_max_comp = T.max()
    plot_T1_fill = T1 if T1 > 0 else T_min_comp
    plot_T2_fill = T2 if T2 > 0 else T_max_comp
    # Ensure fill range is valid
    if plot_T1_fill < plot_T2_fill:
        auxx = [plot_T1_fill, plot_T1_fill, plot_T2_fill, plot_T2_fill, plot_T1_fill]
        auxy = [0, limy, limy, 0, 0]
        ax_spec.fill_between(auxx, auxy, color='silver', alpha=0.4, label='Match Range')
        ax_spec.plot(auxx, auxy, color='silver', alpha=1, lw=0.5)

    # Plot computed spectra
    ax_spec.semilogx(T, PSArotnnor, color='blueviolet', lw=1, label='Original Seed')
    ax_spec.semilogx(T, PSArotnn_scaled, color='cornflowerblue', lw=1, label='Scaled Seed')
    ax_spec.semilogx(T, PSArotnn, color='salmon', lw=1, label='Matched')

    ax_spec.set_xlabel('Period T [s]')
    ax_spec.set_ylabel(f'PSA RotD{nn_val} [g]') # Use nn from results
    ax_spec.set_ylim(bottom=0, top=limy)
    ax_spec.grid(True, which='both', linestyle=':', alpha=0.7)

    # --- APPLY X-LIMITS (NEW BLOCK) ---
    # Default to min/max of computed periods
    x_min_default = T_min_comp
    x_max_default = T_max_comp
    
    # Override with user values if provided
    x_min = xlim_min if xlim_min is not None else x_min_default
    x_max = xlim_max if xlim_max is not None else x_max_default
    
    # Apply limits, ensuring min < max
    if x_min < x_max:
         ax_spec.set_xlim(x_min, x_max)
    else:
         # Log a warning if limits are invalid and fall back to defaults
         warnings.warn(f"Invalid xlim provided (min={x_min} >= max={x_max}). Using defaults [{x_min_default:.2f}, {x_max_default:.2f}].")
         ax_spec.set_xlim(x_min_default, x_max_default)
    # --- END X-LIMITS BLOCK ---

    # Adjust legend position
    ax_spec.legend(ncol=3, bbox_to_anchor=(0.5, 1.02), loc='lower center')
    fig_spec.tight_layout(rect=(0, 0, 1, 0.95))

    return fig_hist, fig_spec

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
    ax_fas.loglog(freqs_fas_raw, fas_scaled_raw, lw=0.5, color='cornflowerblue', alpha=0.5)
    ax_fas.loglog(freqs_fas_raw, fas_matched_raw, lw=0.5, color='salmon', alpha=0.5)

    ax_fas.loglog(output_freq_vector, fas_orig_smooth, lw=1.5, color='blueviolet', label='Original Seed (Smooth)')
    ax_fas.loglog(output_freq_vector, fas_scaled_smooth, lw=1.5, color='cornflowerblue', label='Scaled Seed (Smooth)')
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
    ax_psd.loglog(freqs_psd_raw, psd_scaled_raw, lw=0.5, color='cornflowerblue', alpha=0.5)
    ax_psd.loglog(freqs_psd_raw, psd_matched_raw, lw=0.5, color='salmon', alpha=0.5)

    ax_psd.loglog(output_freq_vector, psd_orig_smooth, lw=1.5, color='blueviolet', label='Original Seed (Smooth)')
    ax_psd.loglog(output_freq_vector, psd_scaled_smooth, lw=1.5, color='cornflowerblue', label='Scaled Seed (Smooth)')
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

    colors = ['blueviolet', 'cornflowerblue', 'salmon']
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
        
    colors = ['blueviolet', 'cornflowerblue', 'salmon']
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

def plot_psa_psd_results(
    results: Dict[str, Any],
    targetPSAlimits: Tuple[float, float] = (0.9, 1.3),
    PSDreduction: float = 1.0,
    F1PSA: float = 0.2,
    F2PSA: float = 50.0,
    F1PSD: float = 0.3,
    F2PSD: float = 30.0,
    zi: float = 0.05,
    nameOut: str = 'ReqPyOut',
    units: str = 'g') -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates verification plots matching the exact style of ReqPyPSD_Module.py.
    """
    
    # --- Unpack Data ---
    freqs = results['freqs'] # Log spaced frequencies for plotting
    periods = 1/freqs
    t = results['t']
    
    # Spectra
    psa_target = results['target_psa']
    psd_target = results['target_psd']
    psa_s = results['psa_s'] # Scaled seed PSA
    psa_sc = results['psa_sc']; psa_sca = results['psa_sca']
    psd_s = results['psd_s'] # Scaled seed PSD
    psd_sc = results['psd_sc']; psd_sca = results['psd_sca']
    
    # Time Histories
    s_scaled = results['s_scaled']; sc = results['sc']; sca = results['sca']
    v_s = results['vel_s']; v_sc = results['vel_sc']; v_sca = results['vel_sca']
    d_s = results['disp_s']; d_sc = results['disp_sc']; d_sca = results['disp_sca']
    ai_s = results['ai_s']; ai_sc = results['ai_sc']; ai_sca = results['ai_sca']
    cav_s = results['cav_s']; cav_sc = results['cav_sc']; cav_sca = results['cav_sca']

    # Ratios
    ratioPSAsc = psa_sc / psa_target
    ratioPSAsca = psa_sca / psa_target
    ratioPSDsc = psd_sc / psd_target
    ratioPSDsca = psd_sca / psd_target
    
    # Indices for checking ranges
    TlocsPSA = np.where((periods >= 1/F2PSA) & (periods <= 1/F1PSA))
    FlocsPSD = np.where((freqs >= F1PSD) & (freqs <= F2PSD))

    # --- Plot Settings ---
    mpl.rcParams['font.size'] = 9
    mpl.rcParams['legend.frameon'] = False
    
    # Limits and Fills
    PSAylim = (-0.05, 1.03 * np.max(np.hstack((targetPSAlimits[1]*psa_target, psa_sc, psa_sca))))
    PSDylim = (PSDreduction * np.min(psd_target), 1.1 * np.max(np.hstack((psd_sc, psd_sca))))
    
    auxxPSA = [F1PSA,F1PSA,F2PSA,F2PSA,F1PSA]
    auxyPSA = [PSAylim[0],PSAylim[1],PSAylim[1],PSAylim[0],PSAylim[0]]
    
    auxxPSD = [F1PSD,F1PSD,F2PSD,F2PSD,F1PSD]
    auxyPSD = [PSDylim[0],PSDylim[1],PSAylim[1],PSDylim[0],PSDylim[0]]

    # --- FIGURE 1: SPECTRA ---
    fig1 = plt.figure(constrained_layout=True, figsize=(6.5, 4.6))
    gs = fig1.add_gridspec(3, 2)
    
    # Top Left: PSA
    ax1 = fig1.add_subplot(gs[0:2, 0])
    ax1.fill_between( auxxPSA, auxyPSA, color='silver', alpha=0.3)
    ax1.semilogx(freqs, psa_target, label='target', color='black', lw=2)
    ax1.semilogx(freqs, targetPSAlimits[1]*psa_target, '--', label='SRP 3.7.1 limits', color='black', lw=1)
    ax1.semilogx(freqs, targetPSAlimits[0]*psa_target, '--', color='black', lw=1)
    ax1.semilogx(freqs, psa_sc, label='PSA matched', color='cornflowerblue', lw=1)
    ax1.semilogx(freqs, psa_sca, label='PSA matched / PSD adjusted', color='salmon', lw=1)
    
    ax1.set_xticklabels([])
    ax1.set_ylim(PSAylim)
    ax1.set_xlim((0.09, 110))
    ax1.set_ylabel(f'PSA [{units}]')
    
    # Bottom Left: PSA Ratio
    ax2 = fig1.add_subplot(gs[2, 0])
    ax2.semilogx(freqs, np.ones_like(freqs), color='black', lw=1)
    ax2.semilogx(freqs, targetPSAlimits[1]*np.ones_like(freqs), '--', color='black', lw=1)
    ax2.semilogx(freqs, targetPSAlimits[0]*np.ones_like(freqs), '--', color='black', lw=1)
    ax2.semilogx(freqs[TlocsPSA], ratioPSAsc[TlocsPSA], color='cornflowerblue', lw=1)
    ax2.semilogx(freqs[TlocsPSA], ratioPSAsca[TlocsPSA], color='salmon', lw=1)
    
    ax2.set_xlim((0.09, 110))
    ax2.set_xlabel('F [Hz]')
    ax2.set_ylabel('PSA ratio')
    
    # Top Right: PSD
    ax3 = fig1.add_subplot(gs[0:2, 1])
    ax3.fill_between( auxxPSD, auxyPSD, color='silver', alpha=0.3)
    ax3.loglog(freqs, psd_target, color='black', lw=2)
    ax3.loglog(freqs, PSDreduction*psd_target, '--', color='black', lw=1)
    ax3.loglog(freqs, psd_sc, color='cornflowerblue', lw=1)
    ax3.loglog(freqs, psd_sca, color='salmon', lw=1)
    
    ax3.set_xticklabels([])
    ax3.set_ylim(PSDylim)
    ax3.set_xlim((0.09, 110))
    ax3.set_ylabel(fr'PSD [{units}$^2$/Hz]')
    
    # Bottom Right: PSD Ratio
    ax4 = fig1.add_subplot(gs[2, 1])
    ax4.semilogx(freqs, np.ones_like(freqs), color='black', lw=1)
    ax4.semilogx(freqs, PSDreduction*np.ones_like(freqs), '--', color='black', lw=1)
    ax4.semilogx(freqs[FlocsPSD], ratioPSDsc[FlocsPSD], color='cornflowerblue', lw=1)
    ax4.semilogx(freqs[FlocsPSD], ratioPSDsca[FlocsPSD], color='salmon', lw=1)
    
    ax4.set_xlim((0.09, 110))
    ax4.set_xlabel('F [Hz]')
    ax4.set_ylabel('PSD ratio')
    
    # Legend
    fig1.legend(loc='upper center', ncols=4, labelcolor='linecolor')
    fig1.tight_layout(rect=(0, 0, 1, 0.96))
    
    
    # --- FIGURE 2: TIME HISTORIES ---
    
    # Normalize AI
    ai_s_norm = ai_s / ai_s[-1]
    ai_sc_norm = ai_sc / ai_sc[-1]
    ai_sca_norm = ai_sca / ai_sca[-1]
    
    # Determine limits
    alim = 1.05 * np.max(np.abs(np.array([s_scaled, sc, sca])))
    vlim = 1.05 * np.max(np.abs(np.array([v_s, v_sc, v_sca])))
    dlim = 1.05 * np.max(np.abs(np.array([d_s, d_sc, d_sca])))

    fig2 = plt.figure(figsize=(6.5, 6))
    
    # 1. Acceleration
    plt.subplot(511)
    plt.plot(t, s_scaled, linewidth=1, color='darkgray')
    plt.plot(t, sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, sca, linewidth=1, color='salmon')
    plt.ylim(-alim, alim)
    plt.ylabel(f'a [{units}]')
    plt.gca().xaxis.set_ticklabels([])
    
    # 2. Velocity
    plt.subplot(512)
    plt.plot(t, v_s, linewidth=1, color='darkgray')
    plt.plot(t, v_sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, v_sca, linewidth=1, color='salmon')
    plt.ylim(-vlim, vlim)
    plt.ylabel(f'v [{units}-s]')
    plt.gca().xaxis.set_ticklabels([])
    
    # 3. Displacement
    plt.subplot(513)
    plt.plot(t, d_s, linewidth=1, color='darkgray')
    plt.plot(t, d_sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, d_sca, linewidth=1, color='salmon')
    plt.ylim(-dlim, dlim)
    plt.ylabel(f'd [{units}-s$^2$]')
    plt.gca().xaxis.set_ticklabels([])
    
    # 4. CAV
    plt.subplot(514)
    plt.plot(t, cav_s, linewidth=1, color='darkgray')
    plt.plot(t, cav_sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, cav_sca, linewidth=1, color='salmon')
    plt.ylabel(f'CAV [{units}-s]')
    plt.gca().xaxis.set_ticklabels([])
    
    # 5. AI (Normalized)
    plt.subplot(515)
    plt.plot(t, ai_s_norm, linewidth=1, color='darkgray', label='scaled')
    plt.plot(t, ai_sc_norm, linewidth=1, color='cornflowerblue', label='PSA matched')
    plt.plot(t, ai_sca_norm, linewidth=1, color='salmon', label='PSA matched / PSD adjusted')
    plt.ylim(-0.05, 1.05)
    plt.ylabel('AI norm.')
    plt.xlabel('t [s]')
    
    plt.figlegend(loc='upper center', ncols=3, labelcolor='linecolor')
    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.95, bottom=0.08)

    return fig1, fig2

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
    nameOut: str = 'ReqPyOut',
    units: str = 'g') -> Tuple[plt.Figure, plt.Figure]:
    """
    Generates verification plots for PSA+FAS+PSD results.
    """
    freqs = results['freqs']; t = results['t']
    
    psa_target = results['target_psa']; psd_target = results['target_psd']; fas_target = results['target_fas']
    psa_sc = results['psa_sc']; psa_sc_fas = results['psa_sc_fas']; psa_sca = results['psa_sca']
    psd_sc = results['psd_sc']; psd_sc_fas = results['psd_sc_fas']; psd_sca = results['psd_sca']
    fas_sc = results['fas_sc']; fas_sc_fas = results['fas_sc_fas']; fas_sca = results['fas_sca']
    
    s_scaled = results['s_scaled']; sc = results['sc']; sc_fas = results['sc_fas']; sca = results['sca']
    v_s = results['vel_s']; v_sc = results['vel_sc']; v_sc_fas = results['vel_sc_fas']; v_sca = results['vel_sca']
    d_s = results['disp_s']; d_sc = results['disp_sc']; d_sc_fas = results['disp_sc_fas']; d_sca = results['disp_sca']
    ai_s = results['ai_s']; ai_sc = results['ai_sc']; ai_sc_fas = results['ai_sc_fas']; ai_sca = results['ai_sca']
    cav_s = results['cav_s']; cav_sc = results['cav_sc']; cav_sc_fas = results['cav_sc_fas']; cav_sca = results['cav_sca']

    ratioPSAsc = psa_sc / psa_target
    ratioPSAscFas = psa_sc_fas / psa_target
    ratioPSAsca = psa_sca / psa_target
    ratioPSDsc = psd_sc / psd_target
    ratioPSDscFas = psd_sc_fas / psd_target
    ratioPSDsca = psd_sca / psd_target
    ratioFASsc = fas_sc / fas_target
    ratioFASscFas = fas_sc_fas / fas_target
    ratioFASsca = fas_sca / fas_target
    
    mpl.rcParams['font.size'] = 9; mpl.rcParams['legend.frameon'] = False
    
    PSAylim = (-0.05, 1.03 * np.max(np.hstack((targetPSAlimits[1]*psa_target, psa_sc, psa_sca))))
    PSDylim = (PSDreduction * np.min(psd_target), 1.1 * np.max(np.hstack((psd_sc, psd_sca))))
    FASylim = (FASreduction * np.min(fas_target), 1.1 * np.max(np.hstack((fas_sc, fas_sca))))
    
    auxxPSA = [F1PSA, F1PSA, F2PSA, F2PSA, F1PSA]
    auxyPSA = [PSAylim[0], PSAylim[1], PSAylim[1], PSAylim[0], PSAylim[0]]
    auxxCheck = [F1Check, F1Check, F2Check, F2Check, F1Check]
    auxyPSD = [PSDylim[0], PSDylim[1], PSDylim[1], PSDylim[0], PSDylim[0]]
    auxyFAS = [FASylim[0], FASylim[1], FASylim[1], FASylim[0], FASylim[0]]
    
    ratioPSAlim = (0.7, 1.4); ratioPSDlim = (0.0, 2.0); ratioFASlim = (0.0, 2.0)

    fig1 = plt.figure(constrained_layout=True, figsize=(8, 7.5))
    gs = fig1.add_gridspec(3, 2)
    
    ax = fig1.add_subplot(gs[0, 0])
    ax.fill_between(auxxPSA, auxyPSA, color='silver', alpha=0.3)
    ax.semilogx(freqs, psa_target, label='target', color='black', lw=2)
    ax.semilogx(freqs, targetPSAlimits[1]*psa_target, '--', label='limits', color='black', lw=1)
    ax.semilogx(freqs, targetPSAlimits[0]*psa_target, '--', color='black', lw=1)
    ax.semilogx(freqs, psa_sc, label='PSA matched', color='cornflowerblue', lw=1)
    ax.semilogx(freqs, psa_sc_fas, label='PSA/FAS', color='mediumseagreen', lw=1)
    ax.semilogx(freqs, psa_sca, label='PSA/FAS/PSD', color='salmon', lw=1)
    ax.set_xticklabels([]); ax.set_ylim(PSAylim); ax.set_xlim((0.09, 110)); ax.set_ylabel(f'PSA [{units}]')
    
    ax = fig1.add_subplot(gs[0, 1])
    ax.semilogx(freqs, np.ones_like(freqs), color='black', lw=1)
    ax.semilogx(freqs, targetPSAlimits[1]*np.ones_like(freqs), '--', color='black', lw=1)
    ax.semilogx(freqs, targetPSAlimits[0]*np.ones_like(freqs), '--', color='black', lw=1)
    idx_psa = np.where((freqs >= F1PSA) & (freqs <= F2PSA))
    ax.semilogx(freqs[idx_psa], ratioPSAsc[idx_psa], color='cornflowerblue', lw=1)
    ax.semilogx(freqs[idx_psa], ratioPSAscFas[idx_psa], color='mediumseagreen', lw=1)
    ax.semilogx(freqs[idx_psa], ratioPSAsca[idx_psa], color='salmon', lw=1)
    ax.set_xlim((0.09, 110)); ax.set_ylim(ratioPSAlim); ax.set_xticklabels([]); ax.set_ylabel('PSA Ratio')

    ax = fig1.add_subplot(gs[1, 0])
    ax.fill_between(auxxCheck, auxyFAS, color='silver', alpha=0.3)
    ax.loglog(freqs, fas_target, color='black', lw=2)
    ax.loglog(freqs, FASreduction*fas_target, '--', color='black', lw=1)
    ax.loglog(freqs, fas_sc, color='cornflowerblue', lw=1)
    ax.loglog(freqs, fas_sc_fas, color='mediumseagreen', lw=1)
    ax.loglog(freqs, fas_sca, color='salmon', lw=1)
    ax.set_xticklabels([]); ax.set_ylim(FASylim); ax.set_xlim((0.09, 110)); ax.set_ylabel(f'FAS [{units}-s]')
    
    ax = fig1.add_subplot(gs[1, 1])
    ax.semilogx(freqs, np.ones_like(freqs), color='black', lw=1)
    ax.semilogx(freqs, FASreduction*np.ones_like(freqs), '--', color='black', lw=1)
    idx_check = np.where((freqs >= F1Check) & (freqs <= F2Check))
    ax.semilogx(freqs[idx_check], ratioFASsc[idx_check], color='cornflowerblue', lw=1)
    ax.semilogx(freqs[idx_check], ratioFASscFas[idx_check], color='mediumseagreen', lw=1)
    ax.semilogx(freqs[idx_check], ratioFASsca[idx_check], color='salmon', lw=1)
    ax.set_xlim((0.09, 110)); ax.set_ylim(ratioFASlim); ax.set_xticklabels([]); ax.set_ylabel('FAS Ratio')

    ax = fig1.add_subplot(gs[2, 0])
    ax.fill_between(auxxCheck, auxyPSD, color='silver', alpha=0.3)
    ax.loglog(freqs, psd_target, color='black', lw=2)
    ax.loglog(freqs, PSDreduction*psd_target, '--', color='black', lw=1)
    ax.loglog(freqs, psd_sc, color='cornflowerblue', lw=1)
    ax.loglog(freqs, psd_sc_fas, color='mediumseagreen', lw=1)
    ax.loglog(freqs, psd_sca, color='salmon', lw=1)
    ax.set_ylim(PSDylim); ax.set_xlim((0.09, 110)); ax.set_ylabel(fr'PSD [{units}$^2$/Hz]'); ax.set_xlabel('F [Hz]')
    
    ax = fig1.add_subplot(gs[2, 1])
    ax.semilogx(freqs, np.ones_like(freqs), color='black', lw=1)
    ax.semilogx(freqs, PSDreduction*np.ones_like(freqs), '--', color='black', lw=1)
    ax.semilogx(freqs[idx_check], ratioPSDsc[idx_check], color='cornflowerblue', lw=1)
    ax.semilogx(freqs[idx_check], ratioPSDscFas[idx_check], color='mediumseagreen', lw=1)
    ax.semilogx(freqs[idx_check], ratioPSDsca[idx_check], color='salmon', lw=1)
    ax.set_xlim((0.09, 110)); ax.set_ylim(ratioPSDlim); ax.set_xlabel('F [Hz]'); ax.set_ylabel('PSD Ratio')
    
    lines = [
        mpl.lines.Line2D([0], [0], color='cornflowerblue', lw=1, label='PSA matched'),
        mpl.lines.Line2D([0], [0], color='mediumseagreen', lw=1, label='PSA / FAS'),
        mpl.lines.Line2D([0], [0], color='salmon', lw=1, label='PSA / FAS / PSD'),
        mpl.lines.Line2D([0], [0], color='black', lw=2, label='target'),
        mpl.lines.Line2D([0], [0], color='black', linestyle='--', lw=1, label='limits')
    ]
    
    # Legend
    fig1.legend(loc='upper center', ncols=5, labelcolor='linecolor')
    fig1.tight_layout(rect=(0, 0, 1, 0.96))

    ai_s_norm = ai_s / ai_s[-1] if ai_s[-1] != 0 else ai_s
    ai_sc_norm = ai_sc / ai_sc[-1] if ai_sc[-1] != 0 else ai_sc
    ai_sc_fas_norm = ai_sc_fas / ai_sc_fas[-1] if ai_sc_fas[-1] != 0 else ai_sc_fas
    ai_sca_norm = ai_sca / ai_sca[-1] if ai_sca[-1] != 0 else ai_sca
    
    alim = 1.05 * np.max(np.abs(np.array([s_scaled, sc, sc_fas, sca])))
    vlim = 1.05 * np.max(np.abs(np.array([v_s, v_sc, v_sc_fas, v_sca])))
    dlim = 1.05 * np.max(np.abs(np.array([d_s, d_sc, d_sc_fas, d_sca])))

    fig2 = plt.figure(figsize=(6.5, 7))
    
    plt.subplot(511)
    plt.plot(t, s_scaled, linewidth=1, color='darkgray')
    plt.plot(t, sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, sc_fas, linewidth=1, color='mediumseagreen')
    plt.plot(t, sca, linewidth=1, color='salmon')
    plt.ylim(-alim, alim); plt.gca().xaxis.set_ticklabels([]); plt.ylabel(f'a [{units}]')
    
    plt.subplot(512)
    plt.plot(t, v_s, linewidth=1, color='darkgray')
    plt.plot(t, v_sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, v_sc_fas, linewidth=1, color='mediumseagreen')
    plt.plot(t, v_sca, linewidth=1, color='salmon')
    plt.ylim(-vlim, vlim); plt.gca().xaxis.set_ticklabels([]); plt.ylabel(f'v [{units}-s]')
    
    plt.subplot(513)
    plt.plot(t, d_s, linewidth=1, color='darkgray')
    plt.plot(t, d_sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, d_sc_fas, linewidth=1, color='mediumseagreen')
    plt.plot(t, d_sca, linewidth=1, color='salmon')
    plt.ylim(-dlim, dlim); plt.gca().xaxis.set_ticklabels([]); plt.ylabel(f'd [{units}-s$^2$]')
    
    plt.subplot(514)
    plt.plot(t, ai_s_norm, linewidth=1, color='darkgray', label='scaled')
    plt.plot(t, ai_sc_norm, linewidth=1, color='cornflowerblue', label='PSA matched')
    plt.plot(t, ai_sc_fas_norm, linewidth=1, color='mediumseagreen', label='PSA / FAS')
    plt.plot(t, ai_sca_norm, linewidth=1, color='salmon', label='PSA / FAS / PSD')
    plt.ylim(-0.05, 1.05); plt.ylabel('AI norm.'); plt.gca().xaxis.set_ticklabels([])
    
    plt.subplot(515)
    plt.plot(t, cav_s, linewidth=1, color='darkgray')
    plt.plot(t, cav_sc, linewidth=1, color='cornflowerblue')
    plt.plot(t, cav_sc_fas, linewidth=1, color='mediumseagreen')
    plt.plot(t, cav_sca, linewidth=1, color='salmon')
    plt.ylabel(f'CAV [{units}-s]'); plt.xlabel('t [s]')
    
       
    plt.figlegend(loc='upper center', ncols=4, labelcolor='linecolor')
    plt.tight_layout(h_pad=0.1)
    plt.subplots_adjust(top=0.95, bottom=0.08)

    return fig1, fig2

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
    nameOut: str = 'ReqPyOut',
    units: str = 'g') -> Tuple[plt.Figure, plt.Figure]:
    """
    Plots RotDnn matching results matching the style of plot_psa_psd_fas_results.
    Spectra (Row 1-3), Time Histories (Row 1-5 for each component).
    """
    freqs = results['freqs']; t = results['t']
    nn = results['nn']
    
    t_psa = results['target_psa']; t_psd = results['target_psd']; t_fas = results['target_fas']
    
    p_sc = results['psa_rotd_sc']; f_sc = results['fas_rotd_sc']; d_sc = results['psd_rotd_sc']
    p_fas = results['psa_rotd_fas']; f_fas = results['fas_rotd_fas']; d_fas = results['psd_rotd_fas']
    p_fin = results['psa_rotd_final']; f_fin = results['fas_rotd_final']; d_fin = results['psd_rotd_final']

    r_psa_sc = p_sc/t_psa; r_psa_fas = p_fas/t_psa; r_psa_fin = p_fin/t_psa
    r_fas_sc = f_sc/t_fas; r_fas_fas = f_fas/t_fas; r_fas_fin = f_fin/t_fas
    r_psd_sc = d_sc/t_psd; r_psd_fas = d_fas/t_psd; r_psd_fin = d_fin/t_psd

    mpl.rcParams['font.size'] = 9; mpl.rcParams['legend.frameon'] = False
    
    # Limits
    PSAylim = (-0.05, 1.03 * np.nanmax(np.hstack((targetPSAlimits[1]*t_psa, p_sc, p_fin))))
    PSDylim = (PSDreduction * np.nanmin(t_psd), 1.1 * np.nanmax(np.hstack((d_sc, d_fin))))
    FASylim = (FASreduction * np.nanmin(t_fas), 1.1 * np.nanmax(np.hstack((f_sc, f_fin))))
    
    # Shading Boxes
    auxxPSA = [F1PSA, F1PSA, F2PSA, F2PSA, F1PSA]
    auxyPSA = [PSAylim[0], PSAylim[1], PSAylim[1], PSAylim[0], PSAylim[0]]
    auxxCheck = [F1Check, F1Check, F2Check, F2Check, F1Check]
    auxyPSD = [PSDylim[0], PSDylim[1], PSDylim[1], PSDylim[0], PSDylim[0]]
    auxyFAS = [FASylim[0], FASylim[1], FASylim[1], FASylim[0], FASylim[0]]
    
    ratioPSAlim = (0.7, 1.4); ratioPSDlim = (0.0, 2.0); ratioFASlim = (0.0, 2.0)
    
    # --- FIG 1: Spectra & Ratios ---
    fig1 = plt.figure(constrained_layout=True, figsize=(8, 7.5))
    gs = fig1.add_gridspec(3, 2)
    
    # PSA
    ax = fig1.add_subplot(gs[0,0])
    ax.fill_between(auxxPSA, auxyPSA, color='silver', alpha=0.3)
    ax.semilogx(freqs, t_psa, 'k-', lw=2, label='Target')
    ax.semilogx(freqs, targetPSAlimits[1]*t_psa, 'k--', label='limits'); ax.semilogx(freqs, targetPSAlimits[0]*t_psa, 'k--')
    ax.semilogx(freqs, p_sc, color='cornflowerblue', label='PSA Matched')
    ax.semilogx(freqs, p_fas, color='mediumseagreen', label='PSA/FAS')
    ax.semilogx(freqs, p_fin, color='salmon', label='PSA/FAS/PSD')
    ax.set_ylabel(f'PSA RotD{nn} [{units}]'); ax.set_xlim(0.09, 110); ax.set_ylim(PSAylim)
    ax.set_xticklabels([])
    
    ax = fig1.add_subplot(gs[0,1])
    # No shading in ratio plots, full range lines
    ax.semilogx([0.1, 100], [1.0, 1.0], 'k-') # Target line across full range
    ax.semilogx([0.1, 100], [targetPSAlimits[1], targetPSAlimits[1]], 'k--') # Limits across full range
    ax.semilogx([0.1, 100], [targetPSAlimits[0], targetPSAlimits[0]], 'k--')
    
    idx_psa = np.where((freqs >= F1PSA) & (freqs <= F2PSA))
    ax.semilogx(freqs[idx_psa], r_psa_sc[idx_psa], 'cornflowerblue')
    ax.semilogx(freqs[idx_psa], r_psa_fas[idx_psa], 'mediumseagreen')
    ax.semilogx(freqs[idx_psa], r_psa_fin[idx_psa], 'salmon')
    ax.set_ylabel('Ratio'); ax.set_xlim(0.09, 110); ax.set_ylim(ratioPSAlim); ax.set_xticklabels([])
    
    # FAS
    ax = fig1.add_subplot(gs[1,0])
    ax.fill_between(auxxCheck, auxyFAS, color='silver', alpha=0.3)
    ax.loglog(freqs, t_fas, 'k-', lw=2)
    ax.loglog(freqs, FASreduction*t_fas, 'k--')
    ax.loglog(freqs, f_sc, 'cornflowerblue'); ax.loglog(freqs, f_fas, 'mediumseagreen'); ax.loglog(freqs, f_fin, 'salmon')
    ax.set_ylabel(f'FAS RotD{nn} [{units}-s]'); ax.set_xlim(0.09, 110); ax.set_ylim(FASylim)
    ax.set_xticklabels([])

    ax = fig1.add_subplot(gs[1,1])
    # No shading
    ax.semilogx([0.1, 100], [1.0, 1.0], 'k-')
    ax.semilogx([0.1, 100], [FASreduction, FASreduction], 'k--')
    
    idx_check = np.where((freqs >= F1Check) & (freqs <= F2Check))
    ax.semilogx(freqs[idx_check], r_fas_sc[idx_check], 'cornflowerblue')
    ax.semilogx(freqs[idx_check], r_fas_fas[idx_check], 'mediumseagreen')
    ax.semilogx(freqs[idx_check], r_fas_fin[idx_check], 'salmon')
    ax.set_ylabel('Ratio'); ax.set_xlim(0.1, 100); ax.set_ylim(ratioFASlim); ax.set_xticklabels([])

    # PSD
    ax = fig1.add_subplot(gs[2,0])
    ax.fill_between(auxxCheck, auxyPSD, color='silver', alpha=0.3)
    ax.loglog(freqs, t_psd, 'k-', lw=2)
    ax.loglog(freqs, PSDreduction*t_psd, 'k--')
    ax.loglog(freqs, d_sc, 'cornflowerblue'); ax.loglog(freqs, d_fas, 'mediumseagreen'); ax.loglog(freqs, d_fin, 'salmon')
    ax.set_ylabel(f'PSD RotD{nn} [{units}^2/Hz]'); ax.set_xlim(0.1, 100); ax.set_ylim(PSDylim)
    ax.set_xlabel('F [Hz]')

    ax = fig1.add_subplot(gs[2,1])
    # No shading
    ax.semilogx([0.1, 100], [1.0, 1.0], 'k-')
    ax.semilogx([0.1, 100], [PSDreduction, PSDreduction], 'k--')
    ax.semilogx(freqs[idx_check], r_psd_sc[idx_check], 'cornflowerblue')
    ax.semilogx(freqs[idx_check], r_psd_fas[idx_check], 'mediumseagreen')
    ax.semilogx(freqs[idx_check], r_psd_fin[idx_check], 'salmon')
    ax.set_ylabel('Ratio'); ax.set_xlabel('F [Hz]'); ax.set_xlim(0.09, 110); ax.set_ylim(ratioPSDlim)
    
    fig1.legend(loc='upper center', ncols=5, labelcolor='linecolor', bbox_to_anchor=(0.5, 1.05))
    
    # --- FIG 2: Time Histories (2 Cols for H1, H2) ---
    s1_sc = results['sc1']; s2_sc = results['sc2']
    s1_fas = results['sc1_fas']; s2_fas = results['sc2_fas']
    s1_fin = results['sca1']; s2_fin = results['sca2']
    s1_scaled = results['s1_scaled']; s2_scaled = results['s2_scaled']
    
    # Helpers
    def get_metrics(acc):
        v = integrate.cumulative_trapezoid(acc, t, initial=0)
        d = integrate.cumulative_trapezoid(v, t, initial=0)
        ai = integrate.cumulative_trapezoid(acc**2, t, initial=0)
        cav = integrate.cumulative_trapezoid(np.abs(v), t, initial=0)
        return v, d, ai/ai[-1], cav
        
    v1_s, d1_s, ai1_s, cav1_s = get_metrics(s1_scaled)
    v2_s, d2_s, ai2_s, cav2_s = get_metrics(s2_scaled)
    v1_sc, d1_sc, ai1_sc, cav1_sc = get_metrics(s1_sc)
    v2_sc, d2_sc, ai2_sc, cav2_sc = get_metrics(s2_sc)
    v1_fas, d1_fas, ai1_fas, cav1_fas = get_metrics(s1_fas)
    v2_fas, d2_fas, ai2_fas, cav2_fas = get_metrics(s2_fas)
    v1_fin, d1_fin, ai1_fin, cav1_fin = get_metrics(s1_fin)
    v2_fin, d2_fin, ai2_fin, cav2_fin = get_metrics(s2_fin)

    # Use width 7.8 (20% more than 6.5)
    fig2, axs = plt.subplots(5, 2, figsize=(7.8, 7), sharex=True) 
    
    # H1 Col 0, H2 Col 1
    comps = [
        (s1_scaled, s1_sc, s1_fas, s1_fin, 'accel'),
        (v1_s, v1_sc, v1_fas, v1_fin, 'vel'),
        (d1_s, d1_sc, d1_fas, d1_fin, 'disp'),
        (ai1_s, ai1_sc, ai1_fas, ai1_fin, 'AI'),
        (cav1_s, cav1_sc, cav1_fas, cav1_fin, 'CAV')
    ]
    
    units_map = {'accel': units, 'vel': f'{units}-s', 'disp': f'{units}-s^2', 'AI': 'norm', 'CAV': f'{units}-s'}
    
    # Calculate limits component-wise to ensure symmetry, handling NaNs
    def get_sym_limit(traces):
        data = np.concatenate(traces)
        if np.all(np.isnan(data)): return 1.0
        mx = np.nanmax(np.abs(data)) # <--- UPDATED: Use nanmax
        if mx == 0 or not np.isfinite(mx): return 1.0
        return 1.05 * mx
        
    def sanitize(data): # <--- NEW: Sanitize helper
        d_clean = np.copy(data)
        d_clean[~np.isfinite(d_clean)] = 0.0
        return d_clean

    for row, (d_s, d_sc, d_fas, d_fin, name) in enumerate(comps):
        # Need corresponding H2 data
        if row==0: d2_s_p, d2_sc_p, d2_fas_p, d2_fin_p = s2_scaled, s2_sc, s2_fas, s2_fin
        elif row==1: d2_s_p, d2_sc_p, d2_fas_p, d2_fin_p = v2_s, v2_sc, v2_fas, v2_fin
        elif row==2: d2_s_p, d2_sc_p, d2_fas_p, d2_fin_p = d2_s, d2_sc, d2_fas, d2_fin
        elif row==3: d2_s_p, d2_sc_p, d2_fas_p, d2_fin_p = ai2_s, ai2_sc, ai2_fas, ai2_fin
        elif row==4: d2_s_p, d2_sc_p, d2_fas_p, d2_fin_p = cav2_s, cav2_sc, cav2_fas, cav2_fin
        
        # Calculate common symmetric limit
        if name in ['accel', 'vel', 'disp']:
            all_data = [d_s, d_sc, d_fas, d_fin, d2_s_p, d2_sc_p, d2_fas_p, d2_fin_p]
            common_lim = get_sym_limit(all_data)
            ylims = (-common_lim, common_lim)
        elif name == 'AI':
            ylims = (-0.05, 1.05)
        elif name == 'CAV':
            all_data = [d_s, d_sc, d_fas, d_fin, d2_s_p, d2_sc_p, d2_fas_p, d2_fin_p]
            common_lim = get_sym_limit(all_data)
            ylims = (0, common_lim)

        # Plot with sanitization
        axs[row, 0].plot(t, sanitize(d_s), 'silver', lw=1)
        axs[row, 0].plot(t, sanitize(d_sc), 'cornflowerblue', lw=1)
        axs[row, 0].plot(t, sanitize(d_fas), 'mediumseagreen', lw=1)
        axs[row, 0].plot(t, sanitize(d_fin), 'salmon', lw=1)
        axs[row, 0].set_ylabel(f'{name} [{units_map[name]}]')
        axs[row, 0].set_ylim(ylims)
        
        axs[row, 1].plot(t, sanitize(d2_s_p), 'silver', lw=1)
        axs[row, 1].plot(t, sanitize(d2_sc_p), 'cornflowerblue', lw=1)
        axs[row, 1].plot(t, sanitize(d2_fas_p), 'mediumseagreen', lw=1)
        axs[row, 1].plot(t, sanitize(d2_fin_p), 'salmon', lw=1)
        axs[row, 1].set_ylim(ylims)
        axs[row, 1].set_yticklabels([]) 

    # Titles inside axes (Top Right Corner)
    axs[0,0].text(0.95, 0.95, 'Component 1', transform=axs[0,0].transAxes, ha='right', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    axs[0,1].text(0.95, 0.95, 'Component 2', transform=axs[0,1].transAxes, ha='right', va='top', fontweight='bold', bbox=dict(facecolor='white', alpha=0.5, edgecolor='none'))
    
    axs[4,0].set_xlabel('t [s]'); axs[4,1].set_xlabel('t [s]')

    lines = [
        mpl.lines.Line2D([0], [0], color='silver', lw=1, label='scaled'),
        mpl.lines.Line2D([0], [0], color='cornflowerblue', lw=1, label='PSA matched'),
        mpl.lines.Line2D([0], [0], color='mediumseagreen', lw=1, label='PSA/FAS'),
        mpl.lines.Line2D([0], [0], color='salmon', lw=1, label='PSA/FAS/PSD')
    ]
    # Use fig.legend (or plt.figlegend) and make space with subplots_adjust
    fig2.legend(handles=lines, loc='upper center', ncols=4, labelcolor='linecolor')
    
    fig2.tight_layout()
    fig2.subplots_adjust(top=0.94) # Make room for legend
    
    return fig1, fig2

def save_generation_spectral_outputs(results: Dict[str, Any], output_prefix: str) -> None:
    """
    Saves the spectral results (PSA, FAS, PSD) for the scaled, matched, and final records
    to text files.
    """
    # 1. Save PSA Data
    periods = results['periods']
    psa_target = results['target_psa']
    psa_sc = results['psa_sc']
    psa_sca = results['psa_sca']
    
    header_psa = "Period(s)  Target_PSA  PSA_Matched  PSA_Final"
    data_psa = np.column_stack((periods, psa_target, psa_sc, psa_sca))
    np.savetxt(f"{output_prefix}_PSA.txt", data_psa, header=header_psa, fmt='%.6e')
    log.info(f"Saved PSA spectra to {output_prefix}_PSA.txt")
    
    # 2. Save Frequency Domain Data (FAS, PSD)
    freqs = results['freqs']
    target_psd = results['target_psd']
    psd_sc = results['psd_sc']
    psd_sca = results['psd_sca']

    header_psd = "Freq(Hz)  Target_PSD  PSD_Matched  PSD_Final"
    data_psd = np.column_stack((freqs, target_psd, psd_sc, psd_sca))
    np.savetxt(f"{output_prefix}_PSD.txt", data_psd, header=header_psd, fmt='%.6e')
    log.info(f"Saved PSD spectra to {output_prefix}_PSD.txt")

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
    sr = np.trapz(D, scales, axis=0) # Integrate along scale axis

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

    Internal helper function. Generally faster for moderate/high damping (z>=3%).

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

def _basecorr(t: np.ndarray, xg: np.ndarray, CT: float, porder: int = -1, imax: int = 80, tol: float = 0.01) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Performs the core baseline correction algorithm iteration [4]_.

    Internal helper function called iteratively by `baselinecorrect`. It applies
    linear correction factors at the start and end of the record to minimize
    final velocity and displacement.

    Parameters
    ----------
    t : np.ndarray
        Time vector (s).
    xg : np.ndarray
        Input acceleration time series (g).
    CT : float
        Duration of the correction window at the start and end of the record (s).
    porder : int, optional
        Order of polynomial for initial detrending. Default -1 (none).
    imax : int, optional
        Maximum number of iterations. Default 80.
    tol : float, optional
        Convergence tolerance as percentage of max response. Default 0.01 (1%).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        - vel (np.ndarray): Velocity of the initial (potentially detrended) record.
        - despl (np.ndarray): Displacement of the initial record.
        - cxg (np.ndarray): Corrected acceleration time series (g).
        - cvel (np.ndarray): Velocity corresponding to `cxg`.
        - cdespl (np.ndarray): Displacement corresponding to `cxg`.

    References
    ----------
    .. [4] Suarez, L. E., & Montejo, L. A. (2007). Applications...
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
    
    r"""Smooths a spectrum using a constant percentage (boxcar) window.

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
    
    r"""Applies Konno-Ohmachi smoothing to a spectrum.

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
    
    r"""Builds a sparse matrix representation of the Konno-Ohmachi 1998 smoother.

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

