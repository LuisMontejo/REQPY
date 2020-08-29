
'''

In this example the two horizontal components from a historic record are 
modified so that the resulting RotD100 response spectrum match the specified 
RotD100 design/target spectrum.

Luis A. Montejo (luis.montejo@upr.edu)

References:
    
    Montejo, L. A. (2020). Response spectral matching of horizontal ground
    motion components to an orientation-independent spectrum (RotDnn). 
    Earthquake Spectra
    
    Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm
    for the generation of spectrum-compatible records. International Journal
    of Advanced Structural Engineering, 5(1), 26.
    
'''


from REQPY_Module import REQPYrotdnn, load_PEERNGA_record
import numpy as np
import matplotlib.pyplot as plt

plt.close('all')

# input:

seed1     = 'RSN730_SPITAK_GUK000.at2'   # seeed record comp1[g]
seed2     = 'RSN730_SPITAK_GUK090.at2'   # seeed record comp2[g]
target   = 'ASCE7.txt'                        # target spectrum (T,PSA)
dampratio = 0.05                              # damping ratio for spectra
TL1 = 0.05; TL2 = 6                           # define period range for matching 
                                              # (T1=T2=0 matches the whole spectrum)

# load target spectrum and seed record:

s1,dt,n,name1 = load_PEERNGA_record(seed1)    # dt: time step, s: accelertion series
s2,dt,n,name2 = load_PEERNGA_record(seed2)

fs   = 1/dt                # sampling frequency (Hz)
tso = np.loadtxt(target)
To = tso[:,0]              # original target spectrum periods
dso = tso[:,1]             # original target spectrum psa

nn = 100                   # percentile 100 for RotD100, 50 for RotD50, ...

(scc1,scc2,cvel1,cvel2,cdisp1,cdisp2,
 PSArotnn,PSArotnnor,T,misfit,rms) = REQPYrotdnn(s1,s2,fs,dso,To,nn,
                                                 T1=TL1,T2=TL2,zi=dampratio,
                                                 nit=15,NS=100,
                                                 baseline=1,plots=1)
                                                 
headerinfo = 'accelerations in g, dt = ' + str(dt)
np.savetxt('component1RotD100matched.txt',scc1,header=headerinfo)
np.savetxt('component2RotD100matched.txt',scc2,header=headerinfo)

        
