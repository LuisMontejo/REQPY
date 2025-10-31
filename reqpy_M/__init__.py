"""
REQPY: A Python module for spectral matching of earthquake records.

This module implements the Continuous Wavelet Transform (CWT) based methodologies
described in the referenced papers to modify (match) earthquake acceleration
time histories to a target response spectrum.

Its primary capabilities include:
1.  Matching a single ground motion component to a target spectrum.
2.  Matching a pair of horizontal components to an orientation-independent
    target spectrum RotDnn (e.g. RotD100).
3.  Analysis functions for generating standard and rotated (RotDnn) spectra.
4.  Baseline correction routines for processed time histories.

---
Quick Start
---

**Example 1: Single Component Matching**

.. code-block:: python

    from reqpy_M import REQPY_single, load_PEERNGA_record, plot_single_results
    import numpy as np
    import matplotlib.pyplot as plt

    # Load seed record and target spectrum
    s, dt, _, _ = load_PEERNGA_record('RSN175_IMPVALL.H_H-E12140.AT2')
    target_spec = np.loadtxt('ASCE7.txt')
    To, dso = target_spec[:, 0], target_spec[:, 1]
    fs = 1/dt

    # Perform matching
    results = REQPY_single(
        s, fs=fs, dso=dso, To=To,
        T1=0.05, T2=6.0, zi=0.05, nit=15
    )

    # Plot results
    fig_hist, fig_spec = plot_single_results(
        results, s, target_spec=(To, dso), T1=0.05, T2=6.0
    )
    # fig_hist.savefig('single_time_history.png')
    # fig_spec.savefig('single_spectrum.png')
    plt.show()

    # Save matched record
    # np.savetxt('matched_single.txt', results['ccs'], header=f'accel [g], dt={dt}')


**Example 2: Two-Component RotDnn Matching (e.g., RotD100)**

.. code-block:: python

    from reqpy_M import REQPYrotdnn, load_PEERNGA_record, plot_rotdnn_results
    import numpy as np
    import matplotlib.pyplot as plt

    # Load seed record components
    s1, dt, _, _ = load_PEERNGA_record('RSN175_IMPVALL.H_H-E12140.AT2')
    s2, _, _, _ = load_PEERNGA_record('RSN175_IMPVALL.H_H-E12230.AT2')
    fs = 1/dt

    # Load target spectrum
    target_spec = np.loadtxt('ASCE7.txt')
    To, dso = target_spec[:, 0], target_spec[:, 1]

    # Perform RotD100 matching
    results = REQPYrotdnn(
        s1, s2, fs=fs, dso=dso, To=To, nn=100,
        T1=0.05, T2=6.0, zi=0.05, nit=15
    )

    # Plot results
    fig_hist, fig_spec = plot_rotdnn_results(
        results, s1, s2, target_spec=(To, dso), T1=0.05, T2=6.0
    )
    # fig_hist.savefig('rotdnn_time_history.png')
    # fig_spec.savefig('rotdnn_spectrum.png')
    plt.show()

    # Save matched records
    # header = f'accel [g], dt={dt}'
    # np.savetxt('matched_rotdnn_comp1.txt', results['scc1'], header=header)
    # np.savetxt('matched_rotdnn_comp2.txt', results['scc2'], header=header)

"""

__author__ = "Luis A. Montejo"
__copyright__ = "Copyright 2021-2025, Luis A. Montejo"
__license__ = "MIT"
__version__ = "0.2.0"
__email__ = "luis.montejo@upr.edu"

# =============================================================================
# IMPORTS
# =============================================================================
import logging
import numpy as np
from numba import jit
from scipy import integrate, signal
import matplotlib.pyplot as plt
import matplotlib as mpl
from typing import Tuple, List, Optional, Dict, Any

# Set up a logger for the module
log = logging.getLogger(__name__)
# Example basic configuration (user can configure this externally)
# logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


# =============================================================================
# PUBLIC API 
# =============================================================================

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
        log.warning("Input records have different lengths (%d vs %d). Truncating to %d points.", n1, n2, n)
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
    PSA180or, _, _ = _ResponseSpectrumTheta(T, s1, s2, zi, dt, theta)
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
    PSA180_iter0, _, _ = _ResponseSpectrumTheta(T, sc1, sc2, zi, dt, theta)
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
        PSA180_iter, _, _ = _ResponseSpectrumTheta(T, ns1[:, m], ns2[:, m], zi, dt, theta)
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
        PSA180_final, _, _ = _ResponseSpectrumTheta(T, scc1, scc2, zi, dt, theta)
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
    PSAs, _, _ = _ResponseSpectrum(T, s, zi, dt)
    PSAsr, _, _ = _ResponseSpectrum(T, sr, zi, dt)

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
        hPSAbc[:, m], _, _ = _ResponseSpectrum(T, ns[:, m], zi, dt)

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
        PSAccs, _, _ = _ResponseSpectrum(T, ccs, zi, dt)
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
        log.warning("Input records for rotdnn have different lengths (%d vs %d). Truncating.", n1, n2)
    s1 = s1[:n]; s2 = s2[:n]
    theta = np.arange(0, 180, 1) # Angles from 0 to 179 degrees
    PSA180, _, _ = _ResponseSpectrumTheta(T, s1, s2, zi, dt, theta)
    PSArotnn = np.percentile(PSA180, nn, axis=0)
    return PSArotnn, PSA180

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
            log.error("Baseline correction failed repeatedly. Returning uncorrected record.")
            # Return original integrated values if correction fails completely
            vel_fail = integrate.cumulative_trapezoid(sc, x=t, initial=0)
            despl_fail = integrate.cumulative_trapezoid(vel_fail, x=t, initial=0)
            return sc, vel_fail, despl_fail

        # Try again with the larger window
        _, _, ccs, cvel, cdespl = _basecorr(t, sc, CTn, porder=porder, imax=imax, tol=tol)

    log.info("Baseline correction successful.")
    return ccs, cvel, cdespl

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
                log.warning(f"Warning: Number of data points read ({len(acc)}) "
                            f"does not match NPTS specified in header ({npts}). Using read data.")
                npts = len(acc) # Update npts to actual data length

    except FileNotFoundError:
        log.error(f"File not found: {filepath}")
        raise
    except Exception as e:
        log.error(f"Error parsing file {filepath}: {e}")
        raise ValueError(f"Error parsing file {filepath}: {e}")

    return acc, dt, npts, eqname

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
        log.error("Results dictionary is missing essential data ('ccs' or 'T'). Cannot plot.")
        # Return empty figures or raise error
        return plt.figure(), plt.figure()
    # --- End data extraction ---

    n = len(ccs)
    t = np.linspace(0, (n - 1) * dt, n)

    # Calculate scaled original histories for comparison
    s_scaled = s_orig[:n] * sf # Ensure s_orig matches length if needed
    vel_scaled = integrate.cumulative_trapezoid(s_scaled, x=t, initial=0)
    despl_scaled = integrate.cumulative_trapezoid(vel_scaled, x=t, initial=0)
    # Ensure PSAs_orig corresponds to T from results
    if len(PSAs_orig) != len(T):
        log.warning("Length mismatch between original spectrum PSAs and periods T. Recalculating original spec.")
        PSAs_orig, _, _ = _ResponseSpectrum(T, s_orig[:n], results.get('zi', 0.05), dt) # Use damping from results if avail.

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
        log.warning("Invalid period range for highlighting (T1 >= T2). Skipping fill.")


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
        log.warning(f"Invalid xlim provided (min={x_min} >= max={x_max}). Using default limits.")
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
        log.error("Results dictionary is missing essential data ('scc1', 'scc2', or 'T'). Cannot plot.")
        return plt.figure(), plt.figure() # Return empty figures
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
        log.warning(f"Length mismatch between original RotDnn (len {len(PSArotnnor)}) and periods T (len {len(T)}). Recalculating.")
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
         log.warning(f"Invalid xlim provided (min={x_min} >= max={x_max}). Using defaults [{x_min_default:.2f}, {x_max_default:.2f}].")
         ax_spec.set_xlim(x_min_default, x_max_default)
    # --- END X-LIMITS BLOCK ---

    # Adjust legend position
    ax_spec.legend(ncol=3, bbox_to_anchor=(0.5, 1.02), loc='lower center')
    fig_spec.tight_layout(rect=(0, 0, 1, 0.95))

    return fig_hist, fig_spec

def save_results_as_at2(
    results: Dict[str, Any],
    filepath: str,
    comp_key: str = 'ccs',
    header_details: Optional[Dict[str, str]] = None
) -> None:
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
        log.error(f"Cannot save .AT2 file: '{comp_key}' or 'dt' not found in results dictionary.")
        return

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
    header_str: Optional[str] = None
) -> None:
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
        log.error(f"Cannot save 2-col file: '{comp_key}' or 'dt' not found in results.")
        return

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
    header_str: Optional[str] = None
) -> None:
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
        log.error(f"Cannot save 1-col file: '{comp_key}' or 'dt' not found in results.")
        return

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
        log.warning("K_psi is near zero. Reconstruction might be unstable. Setting K_psi=1.")
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
        log.warning(f"Specified T1 ({T1:.3f}s) < target spectrum minimum ({T_min_target:.3f}s). Clamping T1.")
        updated_T1 = T_min_target
    if updated_T2 > T_max_target:
        log.warning(f"Specified T2 ({T2:.3f}s) > target spectrum maximum ({T_max_target:.3f}s). Clamping T2.")
        updated_T2 = T_max_target

    # Check against record frequency limits
    if updated_T1 < T_min_record:
        log.warning(f"Specified/Adjusted T1 ({updated_T1:.3f}s) < record Nyquist limit ({T_min_record:.3f}s). Clamping T1.")
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

def _ResponseSpectrum(T: np.ndarray, s: np.ndarray, z: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return _RSFD_S(T, s, z, dt) 
    else:
        log.debug("Using Piecewise (PW) method for spectrum calculation (z<3%).")
        return _RSPW_S(T, s, z, dt)
    
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

@jit(nopython=True, cache=True)
def _RSPW_S(T: np.ndarray, s: np.ndarray, zeta: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        #log.error(f"Invalid damping ratio zeta={zeta:.4f}. _RSPW requires 0 <= zeta < 1. Returning NaNs.")
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
        u_state = np.zeros((2, n)) # Stores [disp, vel] history

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
        for q in range(n - 1):
            # U_{q+1} = A * U_q + B * P_q
            u_state[:, q + 1] = A @ u_state[:, q] + B @ np.array([s_input[q], s_input[q + 1]])

        # Find maximum absolute displacement
        SD[k] = np.max(np.abs(u_state[0, :]))

    # --- Calculate Pseudo Spectra from SD ---
    mask_Tvalid = (T > small_tolerance)
    omega_n = 2 * pi / T[mask_Tvalid] 
    PSV[mask_Tvalid] = omega_n * SD[mask_Tvalid]
    PSA[mask_Tvalid] = omega_n**2 * SD[mask_Tvalid]
 
    return PSA, PSV, SD

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

def _RSFD_S(T: np.ndarray, s: np.ndarray, z: float, dt: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
            log.warning(f"Polynomial detrending failed: {e}. Proceeding without detrending.")
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
        log.warning(f"Correction time CT={CT:.2f}s is too small or large for record "
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
        log.warning(f"_basecorr did not converge within {imax} iterations.")

    # Final integration results
    cvel = integrate.cumulative_trapezoid(cxg, x=t, initial=0)
    cdespl = integrate.cumulative_trapezoid(cvel, x=t, initial=0)
    # Original integrated results (from potentially detrended input)
    vel_orig = integrate.cumulative_trapezoid(xg_detrended, x=t, initial=0)
    despl_orig = integrate.cumulative_trapezoid(vel_orig, x=t, initial=0)

    return vel_orig, despl_orig, cxg, cvel, cdespl

def _ResponseSpectrumTheta(T: np.ndarray, s1: np.ndarray, s2: np.ndarray, z: float, dt: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return _RSFDtheta(T, s1, s2, z, dt, theta)
    else:
        log.debug("Using PW method for rotated spectra (z<3%).")
        return _RSPWtheta(T, s1, s2, z, dt, theta)

def _RSFDtheta(T: np.ndarray, s1: np.ndarray, s2: np.ndarray, zeta: float, dt: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
def _RSPWtheta(T: np.ndarray, s1: np.ndarray, s2: np.ndarray, zeta: float, dt: float, theta: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
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
    #if n1 != n2:
        #log.warning("Input records for _RSPWtheta different lengths. Truncating.")
    s1 = s1[:n]; s2 = s2[:n]
    s_input1 = -s1
    s_input2 = -s2
    
    # --- Validate damping ratio - Return NaNs immediately if invalid ---
    if not 0 <= zeta < 1:
        #log.error(f"Invalid damping ratio zeta={zeta:.4f}. _RSPWtheta requires 0 <= zeta < 1. Returning NaNs.") # <-- Corrected log message
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

        u1 = np.zeros((2, n)) # [disp, vel] history for s1
        u2 = np.zeros((2, n)) # [disp, vel] history for s2

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

        # Time stepping
        for q in range(n - 1):
            u1[:, q + 1] = A @ u1[:, q] + B @ np.array([s_input1[q], s_input1[q + 1]])
            u2[:, q + 1] = A @ u2[:, q] + B @ np.array([s_input2[q], s_input2[q + 1]])

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


# =============================================================================
# MODULE METADATA & REFERENCES
# =============================================================================

"""
REFERENCES:

[1] Montejo, L. A. (2021). Response spectral matching of horizontal ground motion
    components to an orientation-independent spectrum (RotDnn).
    Earthquake Spectra, 37(2), 1127-1144.

[2] Montejo, L. A. (2023). Spectrally matching pulse‐like records to a target
    RotD100 spectrum. Earthquake Engineering & Structural Dynamics, 52(9), 2796-2811.

[3] Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm for
    the generation of spectrum-compatible records.
    International Journal of Advanced Structural Engineering, 5(1), 26.

[4] Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform
    in the generation and analysis of spectrum-compatible records.
    Structural Engineering and Mechanics, 27(2), 173-197.

[5] Suarez, L. E., & Montejo, L. A. (2005). Generation of artificial
    earthquakes via the wavelet transform. Int. Journal of Solids and
    Structures, 42(21-22), 5905-5919.

CHANGELOG:

    v1.2.0 (Oct 2025):
    - Applied NumPy docstring standard and type hinting.
    - Separated plotting and file I/O from core computation functions.
    - Renamed internal helper functions with leading underscore (_).
    - Replaced print statements with logging.
    - Added public plotting functions: plot_single_results, plot_rotdnn_results.
    - Refactored core functions to return dictionaries.
    - Improved error handling and input validation (e.g., record length, T range).

    v1.1.0 (Jan 2025):
    - Fixed minor bugs on the legends of the plots generated
    - Updated numerical integration routine (previous was deprecated by scipy)
    - Optimized response spectra generation routine
    - Automatically saves plots and text files with the generated motions (Removed in v1.2.0)
    - Added optional detrending to the baseline correction routine
"""