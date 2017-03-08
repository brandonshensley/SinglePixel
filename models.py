
import numpy as np
import single_pixel as fg
from scipy.interpolate import RectBivariateSpline

#-------------------------------------------------------------------------------
# Dust models
#-------------------------------------------------------------------------------

class DustModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, 
                 dust_beta=1.6, dust_T=20., fcar=1., fsilfe=0., uval=0., 
                 name=None):
        """
        Generic dust component.
        """
        self.model = 'generic'
        self.name = name
        
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
        
        # List of parameter names
        self.param_names = ['dust_beta', 'dust_T', 'fcar', 'fsilfe', 'uval']
    
    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])
    
    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_beta, self.dust_T, self.fcar, 
                         self.fsilfe, self.uval])
    
    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list 
        returned by self.params().
        """
        self.dust_beta, self.dust_T, self.fcar, self.fsilfe, self.uval = params
    
    def scaling(self, nu):
        return NotImplementedError("The generic DustModel class does not "
                                   "provide a generic scaling() method.")


class DustMBB(DustModel):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        #kwargs['hdmodel'] = False
        super(DustMBB, self).__init__(*args, **kwargs)
        self.model = 'mbb'
        if self.name is None: self.name = "DustMBB"
        
        # Reference frequency
        self.nu_ref = 353. * 1e9
        
        # List of parameter names
        self.param_names = ['dust_beta', 'dust_T',]
    
    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_beta, self.dust_T])
    
    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list 
        returned by self.params().
        """
        self.dust_beta, self.dust_T = params
    
    def scaling(self, nu):
        """
        Return frequency scaling factor at a given frequency.
        """
        beta = self.dust_beta
        Td = self.dust_T
        nu_ref = self.nu_ref
        
        # Frequency-dependent scalings.
        dust_I = (nu / nu_ref)**beta * fg.B_nu(nu, Td) \
               * fg.G_nu(nu_ref, fg.Tcmb) \
               / ( fg.B_nu(353.*1e9, 20.) * fg.G_nu(nu, fg.Tcmb) )
        dust_Q = dust_I
        dust_U = dust_I
        
        return np.array([dust_I, dust_Q, dust_U])


class DustSimpleMBB(DustMBB):
    def __init__(self, *args, **kwargs):
        """
        Simplified modified blackbody dust model.
        """
        super(DustSimpleMBB, self).__init__(*args, **kwargs)
        self.model = 'simplembb'
        if self.name is None: self.name = "DustSimpleMBB"
        
        # List of parameter names
        self.param_names = ['dust_T',]
    
    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.dust_T,])
    
    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list 
        returned by self.params().
        """
        self.dust_T = params


class DustHD(DustModel):
    def __init__(self, *args, **kwargs):
        """
        Modified blackbody dust component.
        """
        # Initialize the base class
        super(DustHD, self).__init__(*args, **kwargs)
        
        # Set model name
        self.model = 'hd'
        if self.name is None: self.name = "DustHD"
        
        # Reference frequency
        self.nu_ref = 353. * 1e9
        
        # Initialize HD model interpolation functions
        self.initialize_hd_dust_model()
        
        # List of parameter names
        self.param_names = ['fcar', 'fsilfe', 'uval']
    
    def initialize_hd_dust_model(self):
        """
        Initialize HD dust model interpolation fns using precomputed data.
        """
        # Read in precomputed dust emission spectra as a fn. of lambda and U
        data_sil = np.genfromtxt("data/sil_DH16_Fe_00_2.0.dat")
        data_silfe = np.genfromtxt("data/sil_DH16_Fe_05_2.0.dat")
        data_cion = np.genfromtxt("data/cion_1.0.dat")
        data_cneu = np.genfromtxt("data/cneu_1.0.dat")
        
        wav = data_sil[:,0] # Wavelength
        uvec = np.linspace(-0.5, 0.5, num=11, endpoint=True)
        
        # Construct splines over wavelength and U
        sil_i = RectBivariateSpline(uvec, wav, 
                    ( (data_sil[:,4:15]/(9.744e-27)) \
                    * (wav[:,np.newaxis]*1.e-4/c)*1.e23).T ) # to Jy
        car_i = RectBivariateSpline(uvec, wav, 
                    ( ((data_cion[:,4:15] + data_cneu[:,4:15])/(2.303e-27)) \
                    * (wav[:,np.newaxis]*1.e-4/c)*1.e23).T ) # to Jy
        silfe_i = RectBivariateSpline(uvec, wav, 
                    ( (data_silfe[:,4:15]/(9.744e-27)) \
                    * (wav[:,np.newaxis]*1.e-4/c)*1.e23).T ) # to Jy
        sil_p = RectBivariateSpline(uvec, wav, 
                    ( (data_sil[:,15:26]/(9.744e-27)) \
                    * (wav[:,np.newaxis]*1.e-4/c)*1.e23).T ) # to Jy
        car_p = RectBivariateSpline(uvec, wav, 
                    ( ((data_cion[:,15:26] + data_cneu[:,4:15])/(2.303e-27)) \
                     * (wav[:,np.newaxis]*1.e-4/c)*1.e23).T ) # to Jy
        silfe_p = RectBivariateSpline(uvec, wav, 
                    ( (data_silfe[:,15:26]/(9.744e-27)) \
                     * (wav[:,np.newaxis]*1.e-4/c)*1.e23).T ) # to Jy
        
        # Store inside object
        self.dust_interp = (car_i, sil_i, silfe_i, car_p, sil_p, silfe_p)
    
    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.fcar, self.fsilfe, self.uval])
    
    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list 
        returned by self.params().
        """
        self.fcar, self.fsilfe, self.uval = params
    
    def scaling(self, nu):
        """
        Return frequency scaling factor at a given frequency.
        """
        # Get necessary model parameters and dust interpolation functions
        fcar = self.fcar
        fsilfe = self.fsilfe
        uval = self.uval
        nu_ref = self.nu_ref
        car_i, sil_i, silfe_i, car_p, sil_p, silfe_p = self.dust_interp
        
        # Calculate wavelength and reference wavelength in suitable units
        lam = 1.e4 * c / (nu) # in microns
        lam_ref = 1.e4 * c / nu_ref # in microns
        
        unit_fac = fg.G_nu(nu_ref, fg.Tcmb) / fg.G_nu(nu, fg.Tcmb)
        
        # Calculate frequency-dependent scaling factors
        dust_I = unit_fac \
               * (  sil_i.ev(uval, lam) 
                  + fcar * car_i.ev(uval, lam) 
                  + fsilfe * silfe_i.ev(uval, lam) ) \
               / (  sil_i.ev(uval, lam_ref) 
                  + fcar * car_i.ev(uval, lam_ref) 
                  + fsilfe * silfe_i.ev(uval, lam_ref) )
        dust_Q = unit_fac \
               * (  sil_p.ev(uval, lam) 
                  + fcar * car_p.ev(uval, lam) 
                  + fsilfe * silfe_p.ev(uval, lam) ) \
               / (  sil_p.ev(uval, lam_ref) 
                  + fcar * car_p.ev(uval, lam_ref) 
                  + fsilfe * silfe_p.ev(uval, lam_ref) )
        dust_U = dust_Q
        
        return np.array([dust_I, dust_Q, dust_U])


#-------------------------------------------------------------------------------
# Synchrotron model
#-------------------------------------------------------------------------------

class SyncModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, sync_beta, name=None):
        """
        Generic synchrotron component.
        """
        self.model = 'generic'
        self.name = "Sync" if name is None else name
        
        # Reference frequency
        self.nu_ref = 30.*1.e9 # Hz
        
        # Conversion factor, 1uK_RJ at 30 GHz to uK_CMB
        nufac = 2.*(self.nu_ref)**2. * fg.k \
              / (fg.c**2. * fg.G_nu(self.nu_ref, fg.Tcmb))
        
        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac
        
        # Set spectral parameters
        self.sync_beta = sync_beta
        
        # List of parameter names
        self.param_names = ['sync_beta',]
    
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
    
    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list 
        returned by self.params().
        """
        self.sync_beta = params
    
    def scaling(self, nu):
        """
        Return frequency scaling factor at a given frequency.
        """
        sync_I = (nu / self.nu_ref)**self.sync_beta \
               * fg.G_nu(self.nu_ref, fg.Tcmb) / fg.G_nu(nu, fg.Tcmb)
        sync_Q = sync_I
        sync_U = sync_I
        
        return np.array([sync_I, sync_Q, sync_U])


class SyncPow(SyncModel):
    def __init__(self, *args, **kwargs):
        """
        Powerlaw synchrotron component.
        """
        super(SyncPow, self).__init__(*args, **kwargs)
        self.model = 'pow'
        if self.name is None: self.name = "SyncPowerlaw"


#-------------------------------------------------------------------------------
# Free-free model
#-------------------------------------------------------------------------------

class FreeFreeModel(object):
    def __init__(self, amp_I, amp_Q, amp_U, ff_beta=-0.118, name=None):
        """
        Generic free-free component.
        """
        self.model = 'generic'
        self.name = name
        
        self.nu_ref = 30. * 1e9 # Reference frequency
        
        # Conversion factor, 1uK_RJ at 30 GHz to uK_CMB
        nufac = 2.*(self.nu_ref)**2. * fg.k \
              / (fg.c**2. * fg.G_nu(self.nu_ref, fg.Tcmb))
        
        # Set amplitude parameters
        self.amp_I = amp_I * nufac
        self.amp_Q = amp_Q * nufac
        self.amp_U = amp_U * nufac
        
        # Set spectral parameters
        self.ff_beta = ff_beta
        
        # List of parameter names
        self.param_names = ['ff_beta',]
    
    def amps(self):
        """
        Return array of amplitudes, [I, Q, U].
        """
        return np.array([self.amp_I, self.amp_Q, self.amp_U])
    
    def params(self):
        """
        Return list of parameters.
        """
        return np.array([self.ff_beta,])
    
    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list 
        returned by self.params().
        """
        self.ff_beta = params
    
    def scaling(self, nu):
        """
        Return frequency scaling factor at a given frequency.
        """
        ff_I = (nu / self.nu_ref)**self.ff_beta \
             * fg.G_nu(self.nu_ref, fg.Tcmb) / fg.G_nu(nu, fg.Tcmb)
        ff_Q = ff_I
        ff_U = ff_I
        return np.array([ff_I, ff_Q, ff_U])


class FreeFreePow(FreeFreeModel):
    def __init__(self, *args, **kwargs):
        """
        Powerlaw synchrotron component.
        """
        super(FreeFreePow, self).__init__(*args, **kwargs)
        self.model = 'pow'
        if self.name is None: self.name = "FFPowerlaw"


#-------------------------------------------------------------------------------
# CMB model
#-------------------------------------------------------------------------------

class CMB(object):
    def __init__(self, amp_I, amp_Q, amp_U, name=None):
        """
        CMB component.
        """
        self.model = 'cmb'
        self.name = "CMB" if name is None else name
        
        # Set amplitude parameters
        self.amp_I = amp_I
        self.amp_Q = amp_Q
        self.amp_U = amp_U
        
        # List of parameter names
        self.param_names = []
    
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
    
    def set_params(self, params):
        """
        Set parameters from an array, using the same ordering as the list 
        returned by self.params().
        """
        pass
    
    def scaling(self, nu):
        """
        Return frequency scaling factor at a given frequency.
        """
        cmb_I = np.ones(len(nu))
        cmb_Q = cmb_I
        cmb_U = cmb_I
        return np.array([cmb_I, cmb_Q, cmb_U])
        
