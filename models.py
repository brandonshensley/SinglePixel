import numpy as np
import single_pixel as fg


#-------------------------------------------------------------------------------
# Dust models
#-------------------------------------------------------------------------------

class DustModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, 
                 dust_beta=1.6, dust_T=20., fcar=1., fsilfe=0., uval=0., 
                 hdmodel=False):
        """
        Generic dust component.
        """
        self.model = 'generic'
        
        # Conversion factor, 1uK_RJ at 353 GHz to uK_CMB
        nufac = 2.*(353e9)**2. * fg.k / (fg.c**2. * fg.G_nu(353e9, fg.Tcmb))
        
        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac
        
        # Set spectral parameters
        self.dust_beta = dust_beta
        self.dust_T = dust_T
        self.fcar = fcar
        self.fsilfe = fsilfe
        self.uval = uval
        
        # Interpolation parameters (if HD model is used)
        self.hdmodel = hdmodel
        if self.hdmodel:
            self.dust_interp = fg.initialize_hd_dust_model()
    
    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])
    
    def params(self):
        """
        Return list of parameters.
        """
        if self.hdmodel:
            return np.array([self.dust_interp, self.fcar, self.fsilfe, self.uval])
        else:
            return np.array([self.dust_beta, self.dust_T])


class DustMBB(DustModel):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        kwargs['hdmodel'] = False
        super(DustMBB, self).__init__(*args, **kwargs)
        self.model = 'mbb'


class DustHD(DustModel):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        kwargs['hdmodel'] = True
        super(DustHD, self).__init__(*args, **kwargs)
        self.model = 'hd'


#-------------------------------------------------------------------------------
# Synchrotron model
#-------------------------------------------------------------------------------

class SyncModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, sync_beta):
        """
        Generic synchrotron component.
        """
        self.model = 'generic'
        
        # Conversion factor, 1uK_RJ at 30 GHz to uK_CMB
        nufac = 2.*(30e9)**2. * fg.k / (fg.c**2. * fg.G_nu(30e9, fg.Tcmb))
        
        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac
        
        # Set spectral parameters
        self.sync_beta = sync_beta
    
    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])
    
    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.sync_beta,])


class SyncPow(SyncModel):
    def __init__(self, *args, **kwargs):
        """
        Powerlaw synchrotron component.
        """
        super(SyncPow, self).__init__(*args, **kwargs)
        self.model = 'pow'


#-------------------------------------------------------------------------------
# CMB model
#-------------------------------------------------------------------------------

class CMB(object):
    def __init__(self, amp_I, amp_Q, amp_U):
        """
        CMB component.
        """
        self.model = 'cmb'
        
        # Set amplitude parameters
        self.amp_I = amp_I
        self.amp_Q = amp_Q
        self.amp_U = amp_U
    
    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])
    
    def params(self):
        """
        Return list of parameters.
        """
        return np.array([])
