'''
REQPY_Module

===============================================================================

Luis A. Montejo (luis.montejo@upr.edu)

References:
    
    Montejo, L. A. (2020). Response spectral matching of horizontal ground motion 
    components to an orientation-independent spectrum (RotDnn). 
    Earthquake Spectra.
    
    Montejo, L. A., & Suarez, L. E. (2013). An improved CWT-based algorithm for 
    the generation of spectrum-compatible records.
    International Journal of Advanced Structural Engineering, 5(1), 26.
    
    Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform
    in the generation and analysis of spectrum-compatible records. 
    Structural Engineering and Mechanics, 27(2), 173-197.

    
===============================================================================

This module contains the python functions required to perform spectral matching 
of 2 horizontal components to a RotDnn target spectrum as described in 
Montejo (2020). The functions included can also be used to perform single 
component matching (Montejo and Suarez 2013), perform baseline correction 
(Suarez and Montejo 2007), and generate single component, rotated and RotDnn 
spectra.

The following is a list of the functions included in the module:   
    
*REQPYrotdnn: Response spectral matching of horizontal ground motion 
 components to an orientation-independent spectrum (RotDnn)

*REQPY_single: CWT based modification of a single component from
 a historic record to obtain spectrally equivalent acceleration series 
             
*ResponseSpectrum: decides what approach to use to estimate the response spectrum
based on the specified damping value (>=4% frequency domain, <4% piecewise)

*RSPW: Response spectra using a piecewise algorithm

*RSFD: Response spectra (operations performed in the frequency domain)

*ResponseSpectrumTheta: decides what approach to use to estimate the rotated 
response spectra based on damping value (>=4% frequency domain, <4% piecewise)

*RSFDtheta: Rotated response spectra, returns the spectra for each angle 
accommodated in a matrix (operations performed in the frequency domain)

*RSPWtheta: Rotated response spectra, returns the spectra for each angle 
accommodated in a matrix (piecewise approach)

*rotdnn - computes rotated and rotdnn spectra

*basecorr: Performs baseline correction

*baselinecorrect: Performs baseline correction (iteratively calling basecorr)

*cwtzm: Continuous Wavelet Transform using the Suarez-Montejo wavelet via 
convolution in the frequency domain

*zumontw: Generates the Suarez-Montejo Wavelet function

*getdetails: Generates the detail functions from the wavelet coefficients

*CheckPeriodRange: Verifies that the specified matching period range is doable

*load_PEERNGA_record: Load record in .at2 format (PEER NGA Databases)

'''

def REQPYrotdnn(s1,s2,fs,dso,To,nn,T1=0,T2=0,zi=0.05,nit=15,NS=100,
                baseline=1,plots=1):
    '''   
    REQPYrotdnn   - Response spectral matching of horizontal ground motion 
    components to an orientation-independent spectrum (RotDnn)
             
    Input:
        
        s1,s2: seed records (acceleration time series in g's)
        fs: seed records samplig frequnecy (Hz) 
        dso: design/target spectrum (g)
        To: vector with the periods at which DS is defined
        nn: percentile at which record is defined rotdnn (e.g. rotd100, rotd50)
        T1, T2: define period range for matching 
                (defautl T1=T2=0 matches the whole spectrum)
        zi: damping ratio for response spectrum (default 5%)
        nit: number of iterations (default 15)
        NS: number of scale values to perform the CWT (default 100)
        baseline: 1/0 (yes/no, whether baseline correction is performed, default 1)
        plots: 1/0 (yes/no, whether plots are generated, default 1)
        
        
    Returns:
        scc1,scc2: spectrally equivalent records (dirs 1 and 2, vectors, g)
        cvel1,cvel2: velocity time histories (dirs 1 and 2, vectors, vel/g)
        cdisp1,cdisp2: displacement time histories (dirs 1 and 2, vectors, displ./g)
        PSArotnn: PSArotnn spectrum for the spectrally matched components (vector, g)
        PSArotnnor: PSArotnnor spectrum for the original components components (vector, g)
        T: periods for PSA (vector, s)
        rmsefin: root mean squared error (float, %)
        meanefin: average misfit (float, %)
    
    '''
    import numpy as np
    from scipy import integrate
    
    pi = np.pi
    n = np.size(s1)
    theta = np.arange(0,180,1)
    
    n1 = np.size(s1); n2 = np.size(s2); n = np.min((n1,n2))
    s1 = s1[:n]; s2 = s2[:n]
    
    dt = 1/fs                       # time step
    t = np.linspace(0,(n-1)*dt,n)   # time vector
    FF1 = 0.1; FF2 = 1/(2*dt)       # defines frequency range for CWT decomposition
    
    Tsortindex=np.argsort(To)
    To = To[Tsortindex]
    dso = dso[Tsortindex]   # ensures ascending order in target spectrum
       
    T1,T2,FF1 = CheckPeriodRange(T1,T2,To,FF1,FF2) # verifies period range   
    
    # Perform Continuous Wavelet Decomposition:
    
    omega = pi; zeta  = 0.05            # wavelet function parameters
    freqs = np.geomspace(FF2,FF1,NS)    # frequencies vector
    T = 1/freqs                         # periods vector
    scales = omega/(2*pi*freqs)         # scales vector    
    C1 = cwtzm(s1,fs,scales,omega,zeta)   # performs CWT using Suarez-Montejo wavelet
    C2 = cwtzm(s2,fs,scales,omega,zeta)   # performs CWT using Suarez-Montejo wavelet
    
    print('='*40)
    print('Wavelet decomposition performed')
    print('='*40)
    
    D1,sr1 = getdetails(t,s1,C1,scales,omega,zeta)
    D2,sr2 = getdetails(t,s2,C2,scales,omega,zeta) # Detail functions and reconstructed signals
    
    print('='*40)
    print('Detail functions generated')
    print('='*40)
    
    ds = np.interp(T,To,dso,left=np.nan,right=np.nan)  # resample target spectrum
    Tlocs = np.nonzero((T>=T1)&(T<=T2))
    nTlocs = np.size(Tlocs)
    
    meane = np.zeros(nit)
    rmse  = np.zeros(nit)
    
    PSA180or,_,_ = ResponseSpectrumTheta(T,s1,s2,zi,dt,theta)
    PSArotnnor = np.percentile(PSA180or,nn,axis=0)
    
    nTlocs = np.size(Tlocs)
    sf = np.sum(ds[Tlocs])/np.sum(PSArotnnor[Tlocs]) # initial scaling factor
    
    sc1  = sf * sr1; D1  = sf * D1
    sc2  = sf * sr2; D2  = sf * D2
    
    # Iterative Process:
    
    meane  = np.zeros((nit+1))
    rmse   = np.zeros((nit+1))
    hPSArotnn = np.zeros((NS,nit+1))   
    hPSArotnn[:,0] = sf*PSArotnnor
    
    ns1 = np.zeros((n, nit+1))
    ns1[:,0] = sc1
    
    ns2 = np.zeros((n, nit+1))
    ns2[:,0] = sc2  
    
    dif = np.abs( hPSArotnn[Tlocs,0] - ds[Tlocs] ) / ds[Tlocs]
    meane[0] = np.mean(dif) * 100
    rmse[0]  = np.linalg.norm(dif) / np.sqrt(nTlocs) * 100
    factor = np.ones((NS,1))
        
    for m in range(1,nit+1):
        print('Now performing iteration %i of %i'%(m,nit))
        factor[Tlocs,0] = ds[Tlocs]/hPSArotnn[Tlocs,m-1]
        
        D1 = factor*D1
        ns1[:,m] = np.trapz(D1.T,scales)
        
        D2 = factor*D2
        ns2[:,m] = np.trapz(D2.T,scales)
        
        PSA180,_,_ = ResponseSpectrumTheta(T,ns1[:,m],ns2[:,m],zi,dt,theta)
        hPSArotnn[:,m] = np.percentile(PSA180,nn,axis=0)
        
        dif = np.abs( hPSArotnn[Tlocs,m] - ds[Tlocs] ) / ds[Tlocs]
        meane[m] = np.mean(dif) * 100
        rmse[m]  = np.linalg.norm(dif) / np.sqrt(nTlocs) * 100
    
    brloc = np.argmin(rmse)   # locates min error
    
    sc1 = ns1[:,brloc]        # compatible record
    sc2 = ns2[:,brloc]        # compatible record
    
    if baseline:
        scc1,cvel1,cdisp1 = baselinecorrect(sc1,t)
        scc2,cvel2,cdisp2 = baselinecorrect(sc2,t)
    else:
        print('='*40)
        print('**baseline correction was not performed**')
        print('='*40)
        scc1 = sc1
        scc2 = sc2
        cvel1 = integrate.cumtrapz(scc1, t, initial=0)
        cdisp1 = integrate.cumtrapz(cvel1, t, initial=0)
        cvel2 = integrate.cumtrapz(scc2, t, initial=0)
        cdisp2 = integrate.cumtrapz(cvel2, t, initial=0)

    PSA180,_,_ = ResponseSpectrumTheta(T,scc1,scc2,zi,dt,theta)
    PSArotnn = np.percentile(PSA180,nn,axis=0)
    
    dif = np.abs(PSArotnn[Tlocs] - ds[Tlocs] ) / ds[Tlocs]
    meanefin = np.mean(dif) * 100
    rmsefin  = np.linalg.norm(dif) / np.sqrt(nTlocs) * 100
    
    print('='*40)
    print('RMSE : %.2f %%'%rmsefin)
    print('AVG. MISFIT : %.2f %%'%meanefin)
    print('='*40)
    
    if plots:
        import matplotlib.pyplot as plt
        
        v1 = integrate.cumtrapz(s1, t, initial=0)
        d1 = integrate.cumtrapz(v1, t, initial=0)
    
        v2 = integrate.cumtrapz(s2, t, initial=0)
        d2 = integrate.cumtrapz(v2, t, initial=0)
        
        sf = np.sum(ds[Tlocs])/np.sum(PSArotnnor[Tlocs])
        sf1 = np.linalg.norm(cvel1)/np.linalg.norm(v1)
        sf2 = np.linalg.norm(cvel2)/np.linalg.norm(v2)
            
        alim = np.max(np.abs(np.array([sf1*s1,scc1,sf2*s2,scc2])))
        vlim = np.max(np.abs(np.array([sf1*v1,cvel1,sf2*v2,cvel2])))
        dlim = np.max(np.abs(np.array([sf1*d1,cdisp1,sf2*d2,cdisp2])))
        plt.figure(figsize=(6.5,5))
        
        plt.subplot(321)
        plt.plot(t,sf1*s1,linewidth=1,color='darkgray')
        plt.plot(t,scc1,linewidth=1,color='navy')
        plt.ylim(-alim,alim)
        plt.ylabel('acc. [g]')
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([])
        
        plt.subplot(323)
        plt.plot(t,sf1*v1,linewidth=1,color='darkgray')
        plt.plot(t,cvel1,linewidth=1,color='navy')
        plt.ylim(-vlim,vlim)
        plt.ylabel('vel./g')
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([])
        
        plt.subplot(325)
        plt.plot(t,sf1*d1,linewidth=1,color='darkgray')
        plt.plot(t,cdisp1,linewidth=1,color='navy')
        plt.ylim(-dlim,dlim)
        plt.ylabel('displ./g'); plt.xlabel('t [s]')
        
        plt.subplot(322)
        plt.plot(t,sf2*s2,linewidth=1,color='darkgray')
        plt.plot(t,scc2,linewidth=1,color='navy')
        plt.ylim(-alim,alim)
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([]);frame1.axes.yaxis.set_ticklabels([])
        
        plt.subplot(324)
        plt.plot(t,sf2*v2,linewidth=1,color='darkgray')
        plt.plot(t,cvel2,linewidth=1,color='navy')
        plt.ylim(-vlim,vlim)
        frame1 = plt.gca();frame1.axes.xaxis.set_ticklabels([]);frame1.axes.yaxis.set_ticklabels([])
        
        plt.subplot(326)
        plt.plot(t,sf2*d2,linewidth=1,color='darkgray',label='Scaled')
        plt.plot(t,cdisp2,linewidth=1,color='navy',label='Matched')
        plt.ylim(-dlim,dlim)
        frame1 = plt.gca();frame1.axes.yaxis.set_ticklabels([])
        plt.xlabel('t [s]')
        plt.figlegend(loc='lower center',ncol=2)
        plt.tight_layout(h_pad=0.3, w_pad=0.3, rect=(0,0.05,1,0.96))
        
        limy = 1.06*np.max([sf*PSArotnnor,PSArotnn])
        auxx = [T1,T1,T2,T2,T1]
        auxy = [0,limy,limy,0,0]
                
        plt.figure(figsize=(6.5,6.5))
        plt.fill_between( auxx, auxy, color='skyblue', alpha=0.2)
        plt.semilogx(T,ds,color='dimgray',linewidth=3)
        plt.semilogx(T,PSArotnnor,'-k')
        plt.semilogx(T,sf*PSArotnnor,'-c')
        plt.semilogx(T,PSArotnn,'-b')
        plt.plot(auxx, auxy, color='Slateblue', alpha=0.6)
        plt.legend((': target',': unscaled',': scaled',': matched'),
                   frameon=False,ncol=4, bbox_to_anchor=(0,1),
                   loc='lower left')
        plt.xlabel('T[s]'); plt.ylabel('PSA RotDnn [g]')
       
    return (scc1,scc2,cvel1,cvel2,cdisp1,cdisp2,PSArotnn,PSArotnnor,
            T,meanefin,rmsefin)
    
def REQPY_single(s,fs,dso,To,T1=0,T2=0,zi=0.05,nit=30,NS=100,baseline=1,plots=1):
    
    '''
    REQPY_single - CWT based modification of a single component from
    a historic records to obtain spectrally equivalent acceleration series 
             
    Input:
        
        Required:
            
        s: seed record (acceleration time series in g's)
        fs: seed record sampling frequency (Hz) 
        dso: design/target spectrum (g)
        To: vector with the periods at which the design spectrum is defined
        
        Optional:
            
        T1, T2: define period range for matching 
                (default: T1=T2=0 matches the whole spectrum)
        zi: damping ratio for response spectrum (default 5%)
        nit: max number of iterations (default 30)
        NS: number of scale values to perform the CWT (default 100)
        baseline: 1/0 (yes/no, whether baseline correction is performed, default 1)
        plots: 1/0 (yes/no, whether plots are generated, default 1)
        
    Returns:
        
        ccs: spectrally equivalent record (vector, g)
        rmsefin: root mean squared error (float, %)
        meanefin: average misfit (float, %)
        cvel: velocity time history (vector, vel/g)
        cdespl: displacement time history (vector, displ./g)
        PSAccs: PSA response spectrum for the compatible record (vector, g)
        PSAs: PSA response spectrum for the seed record (vector, g)
        T: Periods for PSA (vector, s)
        sf: Scaling factor for seed record (float)
    
    '''
        
    import numpy as np
    from scipy import integrate
    
    pi = np.pi
    n = np.size(s)                # number of data points in seed record
    dt = 1/fs                     # time step
    t = np.linspace(0,(n-1)*dt,n) # time vector
    FF1 = 0.1; FF2 = 1/(2*dt)     # frequency range for CWT decomposition
    
    Tsortindex=np.argsort(To)
    To = To[Tsortindex]
    dso = dso[Tsortindex]   # ensures ascending order in target spectrum
       
    T1,T2,FF1 = CheckPeriodRange(T1,T2,To,FF1,FF2) # verifies period range   
    
    # Perform Continuous Wavelet Decomposition:
    
    omega = pi; zeta  = 0.05            # wavelet function parameters
    freqs = np.geomspace(FF2,FF1,NS)    # frequencies vector
    T = 1/freqs                         # periods vector
    scales = omega/(2*pi*freqs)         # scales vector    
    C = cwtzm(s,fs,scales,omega,zeta)   # performs CWT 
    
    print('='*40)
    print('Wavelet decomposition performed')
    print('='*40)
    
    # Generate detail functions:
            
    D,sr = getdetails(t,s,C,scales,omega,zeta) # matrix with the detail
                                               # functions (D) and 
                                               # signal recondtructed (sr)
          
    print('='*40)
    print('Detail functions generated')
    print('='*40)
    
    # response spectra from the reconstructed and original signal:
       
    PSAs,_,_,_,_ =  ResponseSpectrum(T,s,zi,dt)
    PSAsr,_,_,_,_ = ResponseSpectrum(T,sr,zi,dt)
        
    # initial scaling of record:
    
    ds = np.interp(T,To,dso,left=np.nan,right=np.nan)  # resample target spectrum
    
    Tlocs = np.nonzero((T>=T1)&(T<=T2))
    nTlocs = np.size(Tlocs)
    sf = np.sum(ds[Tlocs])/np.sum(PSAs[Tlocs]) # initial scaling factor
    
    sr  = sf * sr; D  = sf * D
      
    # Iterative Process:
    
    meane  = np.zeros((nit+1))
    rmse   = np.zeros((nit+1))
    hPSAbc = np.zeros((NS,nit+1))
    ns = np.zeros((n, nit+1))
    hPSAbc[:,0] = sf*PSAsr
    ns[:,0] = s
    dif = np.abs( hPSAbc[Tlocs,0] - ds[Tlocs] ) / ds[Tlocs]
    meane[0] = np.mean(dif) * 100
    rmse[0]  = np.linalg.norm(dif) / np.sqrt(nTlocs) * 100
    factor = np.ones((NS,1))
    DN = D
    
    for m in range(1,nit+1):
        print('Now performing iteration %i of %i'%(m,nit))
        factor[Tlocs,0] = ds[Tlocs]/hPSAbc[Tlocs,m-1]
        DN = factor*DN
        ns[:,m] = np.trapz(DN.T,scales)
        hPSAbc[:,m],_,_,_,_ = ResponseSpectrum(T,ns[:,m],zi,dt)
        dif = np.abs( hPSAbc[Tlocs,m] - ds[Tlocs] ) / ds[Tlocs]
        meane[m] = np.mean(dif) * 100
        rmse[m]  = np.linalg.norm(dif) / np.sqrt(nTlocs) * 100
    
    brloc = np.argmin(rmse) # locates min error
    sc = ns[:,brloc]        # compatible record
    
    if baseline:
        # perform baseline correction:
        print('='*40)
        print('**now performing baseline correction**')
        print('='*40)
        
        CT = np.max(np.array([1,t[-1]/20])) # time to correct
        vel,despl,ccs,cvel,cdespl = basecorr(t,sc,CT)
        kka = 1; flbc = True
        
        while any(np.isnan(ccs)):
            kka = kka + 1
            CTn = kka*CT
            if CTn >= np.median(t):
                print('='*40)
                print('**baseline correction failed**')
                print('='*40)
                flbc = False; ccs = sc; cvel=vel; cdespl=despl
                break
            vel,despl,ccs,cvel,cdespl = basecorr(t,sc,CTn)
        if flbc:
            print('='*40)
            print('**baseline correction was succesful**')
            print('='*40)
        
        PSAccs,_,_,_,_ = ResponseSpectrum(T,ccs,zi,dt)
        
        difin = np.abs( PSAccs[Tlocs] - ds[Tlocs] ) / ds[Tlocs]
        meanefin = np.mean(difin) * 100
        rmsefin  = np.linalg.norm(difin) / np.sqrt(nTlocs) * 100
    else:
        print('='*40)
        print('**baseline correction was not performed**')
        print('='*40)
        ccs = sc
        cvel = integrate.cumtrapz(ccs, t, initial=0)
        cdespl = integrate.cumtrapz(cvel, t, initial=0)
        PSAccs = hPSAbc[:,brloc]
        meanefin = meane[brloc]
        rmsefin = rmse [brloc]
        
    print('='*40)
    print('RMSE : %.2f %%'%rmsefin)
    print('AVG. MISFIT : %.2f %%'%meanefin)
    print('='*40)
    
    if plots:
        import matplotlib.pyplot as plt
        sosc = sf*s
        velsc = integrate.cumtrapz(sosc, t, initial=0)
        desplsc = integrate.cumtrapz(velsc, t, initial=0)
        plt.figure(figsize=(6.5,6.5))
        plt.subplot(311);plt.plot(t,sosc,'c',t,ccs,'b'); plt.ylabel('acc. [g]')
        plt.subplot(312);plt.plot(t,velsc,'c',t,cvel,'b'); plt.ylabel('vel./g')
        plt.subplot(313);plt.plot(t,desplsc,'c',t,cdespl,'b'); plt.ylabel('displ./g')
        plt.xlabel('time [s]')
        plt.legend((': scaled',': matched'),frameon=False,loc='upper right')
        plt.tight_layout()
        
        limy = 1.06*np.max([sf*PSAs,PSAccs])
        auxx = [T1,T1,T2,T2,T1]
        auxy = [0,limy,limy,0,0]
        ds = np.interp(T,To,dso,left=np.nan,right=np.nan)
        
        plt.figure(figsize=(6.5,6.5))
        plt.fill_between( auxx, auxy, color='skyblue', alpha=0.2)
        plt.semilogx(T,ds,color=[0.5,0.5,0.5],linewidth=3)
        plt.semilogx(T,PSAs,'-k')
        plt.semilogx(T,sf*PSAs,'-c')
        plt.semilogx(T,PSAccs,'-b')
        plt.plot(auxx, auxy, color='Slateblue', alpha=0.6)
        plt.legend((': target',': unscaled',': scaled',': matched'),
                   frameon=False,ncol=4, bbox_to_anchor=(0,1),
                   loc='lower left')
        plt.xlabel('T[s]'); plt.ylabel('PSA [g]')
        
    
    return ccs,rmsefin,meanefin,cvel,cdespl,PSAccs,PSAs,T,sf

def zumontw(t,omega,zeta):
    '''
    zumontw - Generates the Suarez-Montejo Wavelet function
    
    Ref. Suarez, L.E. & Montejo, L.A. Generation of artificial earthquakes 
    via the wavelet transform, Int. Journal of Solids and Structures, 42, 2005
    
    input:
        t     : time vector
        omega : wavelet parameter
        zeta  : wavelet parameter
    
    output:
        wv : wavelet function
    '''
    
    import numpy as np
    wv = np.exp(-zeta*omega*np.abs(t))*np.sin(omega*t)
    return wv

def cwtzm(s,fs,scales,omega,zeta):
    '''
    cwtzm - Continuous Wavelet Transform using the Suarez-Montejo wavelet
    via convolution in the frequency domain

    input:
        s        : input signal (vector)
        fs       : sampling frequency
        scales   : scales at which cwt would be performed 
        omega    : wavelet parameter
        zeta     : wavelet parameter

    output:
        coefs    : wavelet coefficients

    References:
        
        Gaviria, C. A., & Montejo, L. A. (2018). Optimal Wavelet Parameters for 
        System Identification of Civil Engineering Structures. Earthquake Spectra, 
        34(1), 197-216.
        
        Montejo, L.A., Suarez, L.E., 2013. An improved CWT-based algorithm for 
        the generation of spectrum-compatible records. International Journal 
        of Advanced Structural Engineering 5, 1-7.

    '''
    import numpy as np
    from scipy import signal

    nf = np.size(scales)
    dt = 1/fs
    n  = np.size(s)
    t = np.linspace(0,(n-1)*dt,n)
    centertime = np.median(t)

    coefs  = np.zeros((nf,n))
    for k in range(nf):
        wv = zumontw((t-centertime)/scales[k],omega,zeta)/np.sqrt(scales[k])
        coefs[k,:] = signal.fftconvolve(s, wv, mode='same')
        
    return coefs

def getdetails(t,s,C,scales,omega,zeta):
    '''
    getdetails - Generates the detail functions
    
    input:
        t: time vector [s]
        s:  signal being analyzed
        C:  matrix with the coeff. from the CWT
        scales: vector with the scales at which the CWT was performed
        omega, zeta: wavelet parameters
        
    returns:
        D: 2D array with the detail functions
        sr: reconstructed signal        
    '''
    import numpy as np
    from scipy import signal
    
    NS = np.size(scales)
    n = np.size(s)
    D    = np.zeros((NS,n))
    
    centertime = np.median(t)
    
    for k in range(NS):
        wv = zumontw((t-centertime)/scales[k],omega,zeta)  
        D[k,:] = -signal.fftconvolve(C[k,:], wv, mode='same')/(scales[k]**(5/2))
    
    sr = np.trapz(D.T,scales)  # signal reconstructed from the details
    ff = np.max(np.abs(s))/np.max(np.abs(sr))
    sr  = ff * sr
    D   = ff * D
    
    return D,sr

def CheckPeriodRange(T1,T2,To,FF1,FF2):
    '''
    CheckPeriodRange - Verifies that the specified matching period 
    range  is doable 
    
    input:
        To: vector with the periods at which DS is defined
        T1, T2: define period range for matching 
                (defautl T1=T2=0 matches the whole spectrum)
        FF1, FF2: defines frequency range for CWT decomposition
        
    returns:
        updated values of T1,T2,FF1 if required
                
    '''
    if T1==0 and T2==0:
        T1 = To[0]; T2 = To[-1]
        
    if T1<To[0]: 
        T1 = To[0]
        print('='*40)
        print('warning: initial period for matching')
        print('fails outside the target spectrum')
        print('redefined to %.2f' %T1)
        print('='*40)
        
    if T2>To[-1]:
        T2 = To[-1]
        print('='*40)
        print('warning: final period for matching')
        print('fails outside the target spectrum')
        print('redefined to %.2f s' %T2)
        print('='*40)
    
    if T1<(1/FF2):
        T1 = 1/FF2
        print('='*40)
        print('warning: because of sampling frequency')
        print('limitations in the seed record')
        print('the target spectra can only be matched from %.2f s'%T1)
        print('='*40)
    
    if T2>(1/FF1):
        FF1 = 1/T2   # redefine FF1 to match the whole spectrum
    
    return T1,T2,FF1

   
def ResponseSpectrum(T,s,z,dt):
    '''
    ResponseSpectrum - decides what approach to use to estimate the 
    response spectrum based on damping value 
    (>=4% frequency domain, <4% piecewise)
    
    Input:
        T: vector with periods (s)
        s: acceleration time series
        zi: damping ratio
        dt: time steps for s
    
    Returns:
        PSA, PSV, SA, SV, SD
        
    '''
    
    if z>=0.04:
        PSA, PSV, SA, SV, SD = RSFD(T,s,z,dt)
    else:
        PSA, PSV, SA, SV, SD = RSPW(T,s,z,dt)
        
    return PSA, PSV, SA, SV, SD

def RSPW(T,s,zi,dt):
    '''      
    Response spectra using piecewise
    
    Input:
        T: vector with periods (s)
        s: acceleration time series
        zi: damping ratio
        dt: time steps for s
    
    Returns:
        PSA, PSV, SA, SV, SD
    
    '''
    import numpy as np
    
    pi = np.pi
    
    nper = np.size(T)						      # number of natural periods
    n    = np.size(s)                             # length of record
    
    SD   = np.zeros(nper)				          # rel. displac. spectrum
    SV   = np.zeros(nper)				          # rel. vel. spectrum
    SA   = np.zeros(nper)				          # total acc. spectrum	
     
    
    for k in range(nper):
       wn = 2*pi/T[k]
       wd = wn*(1-zi**2)**(1/2)
       
       u = np.zeros((2,n))          # matrix with velocities and displacements
       
       ex = np.exp(-zi*wn*dt)
       cwd = np.cos(wd*dt)
       swd = np.sin(wd*dt)
       zisq = 1/(np.sqrt(1-(zi**2)))
    
       a11 = ex*(cwd+zi*zisq*swd)
       a12 = (ex/wd)*swd
       a21 = -wn*zisq*ex*swd
       a22 = ex*(cwd-zi*zisq*swd)
    
       b11 = ex*(((2*zi**2-1)/((wn**2)*dt)+zi/wn)*(1/wd)*np.sin(wd*dt)+
           (2*zi/((wn**3)*dt)+1/(wn**2))*np.cos(wd*dt))-2*zi/((wn**3)*dt)
       b12 = -ex*(((2*zi**2-1)/((wn**2)*dt))*(1/wd)*np.sin(wd*dt)+
           (2*zi/((wn**3)*dt))*np.cos(wd*dt))-(1/(wn**2))+2*zi/((wn**3)*dt)
       b21 = -((a11-1)/((wn**2)*dt))-a12
       b22 = -b21-a12
       
       A = np.array([[a11,a12],[a21,a22]])
       B = np.array([[b11,b12],[b21,b22]])
    
       for q in range(n-1):
          u[:,q+1] = np.dot(A,u[:,q]) + np.dot(B,np.array([s[q],s[q+1]]))
       
       at = -2*wn*zi*u[1,:]-(wn**2)*u[0,:]
       
       SD[k]   = np.max( np.abs(u[0,:]) )
       SV[k]   = np.max( np.abs(u[1,:]) )
       SA[k]   = np.max( np.abs(at) )
    
    PSV = (2*pi/T)*SD                    # pseudo-vel. spectrum
    PSA = (2*pi/T)**2 *SD  	             # pseudo-accel. spectrum
    
    return PSA, PSV, SA, SV, SD



def RSFD(T,s,z,dt):
    '''   
    luis.montejo@upr.edu 
    
    Response spectra (operations in the frequency domain)
    
    Input:
        T: vector with periods (s)
        s: acceleration time series
        z: damping ratio
        dt: time steps for s
    
    Returns:
        PSA, PSV, SA, SV, SD
    
    '''
    import numpy as np
    from numpy.fft import fft, ifft
    
    pi = np.pi

    npo = np.size(s)
    nT  = np.size(T)
    SD  = np.zeros(nT)
    SV  = np.zeros(nT)
    SA  = np.zeros(nT)
    
    n = int(2**np.ceil(np.log2(npo+10*np.max(T)/dt)))  # add zeros to provide enough quiet time
    fs=1/dt;
    s = np.append(s,np.zeros(n-npo))
    
    fres  = fs/n                            # frequency resolution
    nfrs  = int(np.ceil(n/2))               # number of frequencies
    freqs = fres*np.arange(0,nfrs+1,1)      # vector with frequencies
    ww    = 2*pi*freqs                      # vector with frequencies [rad/s]
    ffts = fft(s);         
    
    m = 1
    for kk in range(nT):
        w = 2*pi/T[kk] ; k=m*w**2; c = 2*z*m*w
        
        H1 = 1       / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Receptance
        H2 = 1j*ww   / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Mobility
        H3 = -ww**2  / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Accelerance
        
        H1 = np.append(H1,np.conj(H1[n//2-1:0:-1]))
        H1[n//2] = np.real(H1[n//2])     # Transfer function (complete) - Receptance
        
        H2 = np.append(H2,np.conj(H2[n//2-1:0:-1]))
        H2[n//2] = np.real(H2[n//2])     # Transfer function (complete) - Mobility
        
        H3 = np.append(H3,np.conj(H3[n//2-1:0:-1]))
        H3[n//2] = np.real(H3[n//2])     # Transfer function (complete) - Accelerance
        
        CoF1 = H1*ffts   # frequency domain convolution
        d = ifft(CoF1)   # go back to the time domain (displacement)
        SD[kk] = np.max(np.abs(d))
            
        CoF2 = H2*ffts   # frequency domain convolution
        v = ifft(CoF2)   # go back to the time domain (velocity)
        SV[kk] = np.max(np.abs(v))
        
        CoF3 = H3*ffts   # frequency domain convolution
        a = ifft(CoF3)   # go back to the time domain (acceleration)
        a = a - s
        SA[kk] = np.max(np.abs(a))
    
    PSV = (2*pi/T)* SD
    PSA = (2*pi/T)**2 * SD
    
    return PSA, PSV, SA, SV, SD

def basecorr(t,xg,CT,imax=80,tol=0.01):
    '''
    performs baseline correction
    
    references:
        
    Wilson, W.L. (2001), Three-Dimensional Static and Dynamic Analysis of Structures: 
    A Physical Approach with Emphasis on Earthquake Engineering, 
    Third Edition, Computers and Structures Inc., Berkeley, California, 2001.
    
    Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform
    in the generation and analysis of spectrum-compatible records. 
    Structural Engineering and Mechanics, 27(2), 173-197.
    
    input:
        t: time vector [s]
        xg: time history of accelerations
        CT: time for correction [s]
        omax: maximum number of iterations (fefault 80)
        tol: tolerance (percent of the max, default 0.01)
    
    return:
        vel: time history of velocities (original record)
        despl: time history of diplacements (original record)
        cxg: baseline-corrected time history of accelerarions
        cvel: baseline-corrected history of velocities
        cdespl: baseline-corrected history of displacements
    '''
    import numpy as np
    from scipy import integrate
    
    n = np.size(xg)
    cxg = np.copy(xg)  
    
    vel = integrate.cumtrapz(xg, t, initial=0)
    despl = integrate.cumtrapz(vel, t, initial=0)
    dt = t[1]-t[0]
    L  = int(np.ceil(CT/(dt))-1)   
    M  = n-L
        
    for q in range(imax):
        
      dU, ap, an = 0, 0, 0
      dV, vp, vn = 0, 0, 0
      
      for i in range(n-1):
          dU = dU + (t[-1]-t[i+1]) * cxg[i+1] * dt
    
      for i in range(L+1):
          aux = ((L-i)/L)*(t[-1]-t[i]) * cxg[i] * dt
          if aux >= 0:
              ap = ap + aux
          else:
              an = an + aux
    
      alfap = -dU/(2*ap)
      alfan = -dU/(2*an)
    
      for i in range(1,L+1):

          if cxg[i]>0:
              cxg[i] = (1 + alfap*(L-i)/L) * cxg[i]
          else:
              cxg[i] = (1 + alfan*(L-i)/L) * cxg[i]
              
      for i in range(n-1):
          dV = dV + cxg[i+1] * dt
          
      for i in range(M-1,n):
          auxv = ((i + 1 - M)/(n-M))*cxg[i]*dt
          if auxv >= 0:
              vp = vp + auxv
          else:
              vn = vn + auxv

      valfap = -dV/(2*vp)
      valfan = -dV/(2*vn)
    
      for i in range(M-1,n):
         
          if cxg[i]>0:
              cxg[i] = (1 + valfap*((i + 1 - M)/(n-M))) * cxg[i]
          else:
              cxg[i] = (1 + valfan*((i + 1 - M)/(n-M))) * cxg[i]
 
      cvel = integrate.cumtrapz(cxg, t, initial=0)
      cdespl = integrate.cumtrapz(cvel, t, initial=0)

      errv = np.abs(cvel[-1]/np.max(np.abs(cvel)))
      errd = np.abs(cdespl[-1]/np.max(np.abs(cdespl)))
    
      if errv <= tol and errd <= tol:
          break
      
    return vel,despl,cxg,cvel,cdespl

def baselinecorrect(sc,t):
    '''
    baselinecorrect - performs baseline correction iteratively 
    calling basecorr
    
    references:
        
    Wilson, W.L. (2001), Three-Dimensional Static and Dynamic Analysis of Structures: 
    A Physical Approach with Emphasis on Earthquake Engineering, 
    Third Edition, Computers and Structures Inc., Berkeley, California, 2001.
    
    Suarez, L. E., & Montejo, L. A. (2007). Applications of the wavelet transform
    in the generation and analysis of spectrum-compatible records. 
    Structural Engineering and Mechanics, 27(2), 173-197.
    
    input:
        sc: uncorrected acceleration time series
        t: time vector
    returns:
        ccs,cvel,cdespl: corrected acc., vel. and disp.
        
    '''
    import numpy as np
    
    CT = np.max(np.array([1,t[-1]/20])) # time to correct
    vel,despl,ccs,cvel,cdespl = basecorr(t,sc,CT)
    kka = 1; flbc = True
    
    while any(np.isnan(ccs)):
        kka = kka + 1
        CTn = kka*CT
        if CTn >= np.median(t):
            print('='*40)
            print('**baseline correction failed**')
            print('='*40)
            flbc = False; ccs = sc; cvel=vel; cdespl=despl
            break
        vel,despl,ccs,cvel,cdespl = basecorr(t,sc,CTn)
    if flbc:
        print('='*40)
        print('**baseline correction was succesful**')
        print('='*40)
        
    return ccs,cvel,cdespl

def ResponseSpectrumTheta(T,s1,s2,z,dt,theta):
    '''
    ResponseSpectrumTheta - decides what approach to use to estimate 
    the response spectrum based on damping value 
    (>=4% frequency domain, <4% piecewise)
    
    Input:
        T: vector with periods (s)
        s1,s2: accelerations time series
        z: damping ratio
        dt: time steps for s
        theta: vector with the angles to calculate the spectra (deg)
    
    Returns:
        PSA,PSV,SD
    '''
    
    if z>=0.04:
        PSA, PSV, SD = RSFDtheta(T,s1,s2,z,dt,theta)
    else:
        PSA, PSV, SD = RSPWtheta(T,s1,s2,z,dt,theta)
        
    return PSA, PSV, SD


def RSFDtheta(T,s1,s2,z,dt,theta):
    '''   
   
    RSFDtheta - Rotated response spectra in the frequency domain, 
    returns the spectra for each theta accomodated in 2D arrays
    
    Input:
        T: vector with periods (s)
        s1,s2: accelerations time series
        z: damping ratio
        dt: time steps for s
        theta: vector with the angles to calculate the spectra (deg)
    
    Returns:
        2D arrays of PSA,PSV,SD
    
    '''
    import numpy as np
    from numpy.fft import fft, ifft
    
    pi = np.pi
    theta = theta*pi/180
    
    ntheta = np.size(theta)
    npo = np.max([np.size(s1),np.size(s2)])
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
    
    m = 1
    for kk in range(nT):
        w = 2*pi/T[kk] ; k=m*w**2; c = 2*z*m*w
        
        H1 = 1       / ( -m*ww**2 + k + 1j*c*ww )  # Transfer function (half) - Receptance
        
        H1 = np.append(H1,np.conj(H1[n//2-1:0:-1]))
        H1[n//2] = np.real(H1[n//2])     # Transfer function (complete) - Receptance
        
        CoFd1 = H1*ffts1   # frequency domain convolution
        d1 = ifft(CoFd1)   # go back to the time domain (displacement)
        d1 = d1[:nor]
        
        CoFd2 = H1*ffts2   # frequency domain convolution
        d2 = ifft(CoFd2)   # go back to the time domain (displacement)
        d2 = d2[:nor]
        
        Md1,Mtheta = np.meshgrid(d1,theta,sparse=True, copy=False)
        Md2,_      = np.meshgrid(d2,theta,sparse=True, copy=False)
                
        drot = Md1*np.cos(Mtheta)+Md2*np.sin(Mtheta)
        
        SD[:,kk] = np.max(np.abs(drot),axis=1)
            
    PSV = (2*pi/T)* SD
    PSA = (2*pi/T)**2 * SD
    
    return PSA,PSV,SD

def RSPWtheta(T,s1,s2,z,dt,theta):
    '''  
    
    RSPWtheta - Rotated response spectra using piecewise, 
    returns the spectra for each theta accomodated in 2D arrays
    
    Input:
        T: vector with periods (s)
        s1,s2: accelerations time series
        z: damping ratio
        dt: time steps for s
        theta: vector with the angles to calculate the spectra (deg)
    
    Returns:
        2D arrays of PSA,PSV,SD
        
    '''
    import numpy as np  
    
    pi = np.pi
    theta = theta*pi/180
    ntheta = np.size(theta)
    
    nT    = np.size(T)						      # number of natural periods
    SD  = np.zeros((ntheta,nT))
    n1 = np.size(s1); n2 = np.size(s2)
    
    if n1>n2:
        n = n2
        s1 = s1[:n]
    else:
        n = n1
        s2 = s2[:n]
    
    for k in range(nT):
       wn = 2*pi/T[k]
       wd = wn*(1-z**2)**(1/2)
       
       u1 = np.zeros((2,n))          # matrix with velocities and displacements
       u2 = np.zeros((2,n))          # matrix with velocities and displacements
       
       ex = np.exp(-z*wn*dt)
       cwd = np.cos(wd*dt)
       swd = np.sin(wd*dt)
       zisq = 1/(np.sqrt(1-(z**2)))
    
       a11 = ex*(cwd+z*zisq*swd)
       a12 = (ex/wd)*swd
       a21 = -wn*zisq*ex*swd
       a22 = ex*(cwd-z*zisq*swd)
    
       b11 = ex*(((2*z**2-1)/((wn**2)*dt)+z/wn)*(1/wd)*np.sin(wd*dt)+
           (2*z/((wn**3)*dt)+1/(wn**2))*np.cos(wd*dt))-2*z/((wn**3)*dt)
       b12 = -ex*(((2*z**2-1)/((wn**2)*dt))*(1/wd)*np.sin(wd*dt)+
           (2*z/((wn**3)*dt))*np.cos(wd*dt))-(1/(wn**2))+2*z/((wn**3)*dt)
       b21 = -((a11-1)/((wn**2)*dt))-a12
       b22 = -b21-a12
       
       A = np.array([[a11,a12],[a21,a22]])
       B = np.array([[b11,b12],[b21,b22]])
    
       for q in range(n-1):
          u1[:,q+1] = np.dot(A,u1[:,q]) + np.dot(B,np.array([s1[q],s1[q+1]]))
          u2[:,q+1] = np.dot(A,u2[:,q]) + np.dot(B,np.array([s2[q],s2[q+1]]))
       
       d1 = u1[0,:]; d2 = u2[0,:]
       
       Md1,Mtheta = np.meshgrid(d1,theta,sparse=True, copy=False)
       Md2,_      = np.meshgrid(d2,theta,sparse=True, copy=False)
                
       drot = Md1*np.cos(Mtheta)+Md2*np.sin(Mtheta) 
       SD[:,k] = np.max(np.abs(drot),axis=1)   
       

    
    PSV = (2*pi/T)*SD                    # pseudo-vel. spectrum
    PSA = (2*pi/T)**2 *SD  	             # pseudo-accel. spectrum
    
    return PSA, PSV, SD

def rotdnn(s1,s2,dt,zi,T,nn):
    '''
    rotdnn - computes rotated and rotdnn spectra
    
    input:
        a1, a2: acceleration series in two orthogonal horizontal directions
        dt: time step [s]
        zi: damping ratio for spectra
        T: periods defining the spectra
        nn: percentile for rotdnn
        
    returns:
        PSArotnn: vector containing the PSA RotDnn response spectrum
        PSA180: matrix containing the PSA response spectrum at different angles
                (from 0 to 179 degrees)
    '''
    import numpy as np
    n1 = np.size(s1); n2 = np.size(s2); n = np.min((n1,n2))
    s1 = s1[:n]; s2 = s2[:n]
    theta = np.arange(0,180,1)
    PSA180,_,_,=ResponseSpectrumTheta(T,s1,s2,zi,dt,theta)
    PSArotnn = np.percentile(PSA180,nn,axis=0)
    return PSArotnn,PSA180

def load_PEERNGA_record(filepath):
    '''
    Load record in .at2 format (PEER NGA Databases)

    Input:
        filepath : file path for the file to be load
        
    Returns:
    
        acc : vector wit the acceleration time series
        dt : time step
        npts : number of points in record
        eqname : string with year_name_station_component info

    '''

    import numpy as np

    with open(filepath) as fp:
        line = next(fp)
        line = next(fp).split(',')
        year = (line[1].split('/'))[2]
        eqname = (year + '_' + line[0].strip() + '_' + 
                  line[2].strip() + '_comp_' + line[3].strip())
        line = next(fp)
        line = next(fp).split(',')
        npts = int(line[0].split('=')[1])
        dt = float(line[1].split('=')[1].split()[0])
        acc = np.array([p for l in fp for p in l.split()]).astype(float)
    
    return acc,dt,npts,eqname
    
