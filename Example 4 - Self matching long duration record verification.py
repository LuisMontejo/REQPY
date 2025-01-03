
'''

In this example the two components of a long duration record with low frequency 
content are self-matched to their RotD100 response spectrum for verification 
purposes after an issue reported by https://github.com/longchen-geo. 

As a result, the code was modified to make the decomposition frequency range 
dependent on the record duration.

Luis A. Montejo (luis.montejo@upr.edu)

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


from REQPY_Module import REQPYrotdnn,  rotdnn
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')
pi = np.pi

seed1     = 'KNG007_NS_X.txt'   # seeed record comp1[g]
seed2     = 'KNG007_EW_Y.txt'   # seeed record comp2[g]
dt = 0.02                       # record time step
fs = 1/dt                       # sampling frequency

dampratio = 0.05                # damping ratio for spectra
TL1 = 0; TL2 = 0                # define period range for matching 
                                # (T1=T2=0 matches the whole spectrum)
NS = 200                        # number of CWT-scales / periods for response spectrum
                                # increased from 100 to 200 to improve resolution
nits = 1                        # just 1 iteration as we are just self-matching    
 
# load both components of the seed record:

gm1 = np.loadtxt(seed1)
s1 = gm1[:,1]

gm2 = np.loadtxt(seed2)
s2 = gm2[:,1]

n1 = np.size(s1); n2 = np.size(s2); n = np.min((n1,n2))
s1 = s1[:n]; s2 = s2[:n]

FF1 = min(4/(n*dt),0.1); FF2 = 1/(2*dt)    # defines frequency range for CWT decomposition
freqs = np.geomspace(FF2,FF1,NS)           # frequencies vector
T = 1/freqs                                # periods vector

# generate RotD100 response spectrum:
PSArot,_     = rotdnn(s1,s2,dt,dampratio,T,100)

# use the RotD100 response spectrum as target for the match:

(scc1,scc2,cvel1,cvel2,cdisp1,cdisp2,
  PSArotnn,PSArotnnor,T,misfit,rms) = REQPYrotdnn(s1,s2,fs,PSArot,T,100,
                                                  T1=TL1,T2=TL2,zi=dampratio,
                                                  nit=nits,NS=NS,
                                                  baseline=0,porder=-1,
                                                  plots=1,nameOut='KNG007')



        
