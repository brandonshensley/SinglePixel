
import numpy as np
from scipy.linalg import sqrtm
from scipy.interpolate import interp1d
import copy, time
import emcee

def ln_prior(pvals, models):
    """
    log-prior
    """
    # Define priors
    priors = {
        'dust_T':       (16., 24.),
        'dust_beta':    (1.4, 1.8),
        'sync_beta':    (-3.6, -2.8)
    }
    
    # Make ordered list of parameter names
    pnames = []
    for m in models:
        pnames += m.param_names
    
    # Go through priors and apply them
    ln_prior = 0. # Set default prior value
    for pn in priors.keys():
        if pn not in pnames: continue
        pmin, pmax = priors[pn] # Prior bounds
        val = pvals[pnames.index(pn)] # Current value of parameter
        if val < pmin or val > pmax:
            ln_prior = -np.inf
    return ln_prior

def lnprob(pvals, data_spec, models_fit, param_spec, Ninv_sqrt):
    """
    log-probability (likelihood times prior) for a set of parameter values.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec
    pnames, initial_vals, parent_model = param_spec
    
    # Apply prior
    logpr = ln_prior(pvals, models_fit)
    if not np.isfinite(logpr):
        return -np.inf
    
    F_fg, F_cmb, F = F_matrix(pvals, nu, models_fit, param_spec)
    H = F_fg.T * Ninv * F_fg

    x_mat = np.linalg.inv(F.T * beam_mat.T * Ninv * beam_mat * F) \
          * F.T * beam_mat.T * Ninv * D_vec # Equation A3
    
    chi_square = (D_vec - beam_mat * F * x_mat).T * Ninv \
               * (D_vec - beam_mat * F * x_mat) # Equation A4
    
    # Equation A14
    U, Lambda, VT = np.linalg.svd(Ninv_sqrt*F_fg, full_matrices=False)
    
    # Equation A16
    N_eff_inv_cmb = F_cmb.T * Ninv_sqrt \
                  * (np.matrix(np.identity(U.shape[0])) - U*U.T) \
                  * Ninv_sqrt * F_cmb
        
    lnprob = logpr - chi_square - 0.5*np.log(np.linalg.det(H)) \
                              - 0.5*np.log(np.linalg.det(N_eff_inv_cmb))
    return lnprob


def F_matrix(pvals, nu, models_fit, param_spec):
    """
    Foreground spectral dependence operator.
    """
    pnames, initial_vals, parent_model = param_spec
    
    # Check that the CMB component is the last component in the model list
    if models_fit[-1].model != 'cmb':
        raise ValueError("The last model in the models_fit list should be a "
                         "CMB() object.")
    
    Nband = len(nu) # No. of frequency bands
    Npol = 3 # No. of data components (I, Q, U)
    Ncomp = len(models_fit) # No. of sky components
    
    F_fg = np.zeros((Npol * Nband, Npol * (Ncomp - 1)))
    F_cmb = np.zeros((Npol * Nband, Npol))
    F = np.zeros((Npol * Nband, Npol * Ncomp))
    
    # Create new copies of model objects to work with
    models = [copy.deepcopy(m) for m in models_fit]
    
    # Set new parameter values for the copied model objects, and then get 
    # scalings as a function of freq./polarisation
    pstart = 0
    for i in range(len(models)):
        m = models[i]
        
        # Set new parameter values in the models
        n = m.params().size
        m.set_params( pvals[pstart:pstart+n] )
        pstart += n # Increment for next model
        
        # Calculate scaling with freq. given new parameter values
        scal = m.scaling(nu)
        
        for j in range(Npol):
            # Fill FG or CMB -matrix with scalings, as appropriate
            if m.model != 'cmb':
                F_fg[j*Nband:(j+1)*Nband, i*Npol + j] = scal[j,:]
            else:
                F_cmb[j*Nband:(j+1)*Nband, j] = scal[j,:]
    
    # Stack CMB and FG F-matrices together
    F = np.hstack((F_cmb, F_fg))
    
    return np.matrix(F_fg), np.matrix(F_cmb), np.matrix(F)


def mcmc(data_spec, models_fit, param_spec, nwalkers=50, 
         burn=500, steps=1000, sample_file=None):
    """
    Run MCMC to fit model to some simulated data.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec
    pnames, initial_vals, parent_model = param_spec
    
    # Invert noise covariance matrix
    Ninv_sqrt = np.matrix(sqrtm(Ninv))
    
    # Get a list of model parameter names (FIXME: Ignores input pnames for now)
    pnames = []
    for mod in models_fit:
        pnames += mod.param_names
    
    # Define starting points
    ndim = len(initial_vals)
    pos = [initial_vals*(1.+1e-3*np.random.randn(ndim)) for i in range(nwalkers)]
    
    # Run emcee sampler
    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob, 
                           args=(data_spec, models_fit, param_spec, Ninv_sqrt) )
    sampler.run_mcmc(pos, burn + steps)
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    
    # Save chains to file
    if sample_file is not None:
        np.savetxt(sample_file, samples, fmt="%.6e", header=" ".join(pnames))
    
    # Summary statistics for fitted parameters
    params_out = np.median(samples, axis=0)
    
    # Return summary statistics and samples
    return params_out, pnames, samples


def noise_model(fname="CMBpol_extended_noise.dat", scale=1.):
    """
    Load noise model from file and create interpolation function as a fn of 
    frequency. This is the noise per pixel, for some arbitrary pixel size.
    """
    # Load from file
    nu, sigma = np.genfromtxt(fname).T
    
    # Extrapolate at the ends of the frequency range
    if nu[0] > 1.:
        sigma0 = sigma[0] \
               + (sigma[1] - sigma[0]) / (nu[1] - nu[0]) * (1. - nu[0])
        sigman = sigma[-1] \
               + (sigma[-1] - sigma[-2]) / (nu[-1] - nu[-2]) * (1e3 - nu[-1])
        if sigma0 < 0.: sigma0 = sigma[0]
        if sigman < 0.: sigman = sigma[-1]
        
        # Add to end of range
        nu = np.concatenate(([1.,], nu, [1e3,]))
        sigma = np.concatenate(([sigma0,], sigma, [sigman,]))
    
    # Rescale by constant overall factor
    sigma *= scale
    
    # Construct interpolation function
    return interp1d(nu, sigma, kind='linear', bounds_error=False)


def generate_data(nu, fsigma_T, fsigma_P, components, 
                  noise_file="core_plus_extended_noise.dat"):
    """
    Create a mock data vector from a given set of models, including adding a 
    noise realization.
    """
    # Loop over components that were included in the data model and calculate 
    # the signal at a given frequency (should be in uK_CMB)
    signal = 0
    cmb_signal = 0
    for comp in components:
        print comp.param_names
        
        # Add this component to total signal
        signal += np.atleast_2d(comp.amps()).T * comp.scaling(nu)
        
        # Store CMB signal separately
        if comp.model == 'cmb':
            cmb_signal = np.atleast_2d(comp.amps()).T * comp.scaling(nu)
    
    # Construct data vector
    D_vec = np.matrix(signal.flatten()).T
    
    # Noise rms as a function of frequency
    sigma_interp = noise_model(fname=noise_file, scale=1.)
    sigma_nu = sigma_interp(nu / 1e9)
    fsigma = np.zeros(3*len(nu))
    fsigma[0:len(nu)] = fsigma_T * sigma_nu # Stokes I
    fsigma[len(nu):2*len(nu)] = fsigma_P * sigma_nu # Stokes Q
    fsigma[2*len(nu):] = fsigma_P * sigma_nu # Stokes U
    
    #noise_mat = np.matrix( np.diagflat(cmb_signal.flatten() * fsigma) )
    noise_mat = np.matrix( np.diagflat(fsigma) )
    Ninv = np.linalg.inv(noise_mat)

    # Add noise to generated data
    D_vec += (np.matrix(np.random.randn(D_vec.size)) * noise_mat).T
    return D_vec, Ninv


def model_test(nu, D_vec, Ninv, models_fit, initial_vals=None, burn=500, 
               steps=1000, cmb_amp_in=None, sample_file=None):
    """
    Generate simulated data given an input model, and perform MCMC fit using 
    another model.
    """
    # Collect together data and noise/instrument model
    Ninv_sqrt = np.matrix(sqrtm(Ninv)) # Invert noise covariance matrix
    beam_mat = np.identity(3*len(nu)) # Beam model
    data_spec = (nu, D_vec, Ninv, beam_mat)
    
    # Loop over specified component models and set up MCMC parameters for them
    pnames = []; pvals = []; parent_model = []
    for mod in models_fit:
        # Get parameter names, initial parameter values, and component ID
        pn = mod.param_names
        pv = mod.params()
        
        # Loop through parameters from this component
        for i in range(len(pn)):
            pnames.append( "%s.%s" % (mod.name, pn[i]) )
            pvals.append( pv[i] )
            parent_model.append( mod )
    
    # Use 'guess' as the initial point for the MCMC if specified        
    if initial_vals is None: initial_vals = pvals
    
    # Collect names, initial values, and parent components for the parameters
    param_spec = (pnames, initial_vals, parent_model)
    
    # Run MCMC sampler on this model
    t0 = time.time()
    params_out, pnames, samples = mcmc(data_spec, models_fit, param_spec, 
                                       burn=burn, steps=steps,
                                       sample_file=sample_file)
    print "MCMC run in %d sec." % (time.time() - t0)
    
    # Estimate error on recovered CMB amplitudes
    F_fg, F_cmb, F = F_matrix(params_out, nu, models_fit, param_spec)
    
    H = F_fg.T * Ninv * F_fg
    
    # Equation A3
    x_mat = np.linalg.inv(F.T * beam_mat.T * Ninv * beam_mat * F) \
          * F.T * beam_mat.T * Ninv * D_vec
    
    # Equation A14
    U, Lambda, VT = np.linalg.svd(Ninv_sqrt*F_fg, full_matrices=False)
    
    # Equation A16
    N_eff_inv_cmb = F_cmb.T * Ninv_sqrt \
                  * (np.matrix(np.identity(U.shape[0])) - U*U.T) \
                  * Ninv_sqrt * F_cmb
    
    N_eff_cmb = np.linalg.inv(N_eff_inv_cmb)
    cmb_noise = np.array([N_eff_cmb[0,0], N_eff_cmb[1,1], N_eff_cmb[2,2]])

    gls_cmb = x_mat[0:3,0]
    cmb_chisq = (np.matrix(cmb_amp_in).T - gls_cmb).T * N_eff_inv_cmb \
              * (np.matrix(cmb_amp_in).T - gls_cmb)
    
    return gls_cmb, cmb_chisq, cmb_noise

