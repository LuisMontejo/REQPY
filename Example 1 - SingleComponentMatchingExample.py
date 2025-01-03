'''

In this example we match a single horizontal component to a target spectrum

References:
    
    
    Montejo, L. A. (2021). Response spectral matching of horizontal ground motion 
    components to an orientation-independent spectrum (RotDnn). 
    Earthquake Spectra, 37(2), 1127-1144.
    
    Montejo, L. A. (2023). Spectrally matching pulse‐like records to a target 
    RotD100 spectrum. Earthquake Engineering & Structural Dynamics, 52(9), 2796-2811.
    
    Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm for 
    the generation of spectrum-compatible records.
    International Journal of Advanced Structural Engineering, 5(1), 26.
    
    Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform
    in the generation and analysis of spectrum-compatible records. 
    Structural Engineering and Mechanics, 27(2), 173-197.
    
'''

from REQPY_Module import REQPY_single, load_PEERNGA_record
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# input:

seed     = 'RSN175_IMPVALL.H_H-E12140.AT2'    # seeed record [g]
target   = 'ASCE7.txt'                        # target spectrum (T,PSA)
dampratio = 0.05                              # damping ratio for spectra
TL1 = 0.05; TL2 = 6                           # define period range for matching 
                                              # (T1=T2=0 matches the whole spectrum)

# load target spectrum and seed record:

s,dt,npts,eqname = load_PEERNGA_record(seed)    
# dt: time step, s: accelertion series, npts: number of points in record

fs   = 1/dt                         # sampling frequency (Hz)
tso = np.loadtxt(target)
To = tso[:,0]              # original target spectrum periods
dso = tso[:,1]             # original target spectrum psa

ccs,rms,misfit,cvel,cdespl,PSAccs,PSAs,T,sf = REQPY_single(s,fs,dso,To,
                                                    T1=TL1,T2=TL2,zi=dampratio,
                                                    nit=15,NS=100,
                                                    baseline=1,porder=-1,plots=1,nameOut='myoutout')



