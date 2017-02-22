
import numpy as np
from scipy.linalg import sqrtm
import copy, time
import emcee

def ln_prior(pvals, models):
    """
    log-prior
    """
    # Define priors
    priors = {
        'dust_T':       (5., 100.),
        'dust_beta':    (0., 3.),
        'sync_beta':    (-5., 0.)
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

def lnprob(pvals, data_spec, models_fit, param_spec):
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
    
    return logpr - chi_square - 0.5*np.log(np.linalg.det(H))


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
    
    # Get a list of model parameter names (FIXME: Ignores input pnames for now)
    pnames = []
    for mod in models_fit:
        pnames += mod.param_names
    
    # Define starting points
    ndim = len(initial_vals)
    pos = [initial_vals*(1.+1e-3*np.random.randn(ndim)) for i in range(nwalkers)]
    
    # Run emcee sampler
    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob, 
                                     args=(data_spec, models_fit, param_spec) )
    sampler.run_mcmc(pos, burn + steps)
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    
    # Save chains to file
    if sample_file is not None:
        np.savetxt(sample_file, samples, fmt="%.6e", header=" ".join(pnames))
    
    # Summary statistics for fitted parameters
    params_out = np.median(samples, axis=0)
    
    # Return summary statistics and samples
    return params_out, pnames, samples


def generate_data(nu, fsigma_T, fsigma_P, components):
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
        
    # Noise model
    fsigma = np.zeros(3*len(nu))
    fsigma[0:len(nu)] = fsigma_T * np.ones(len(nu))
    fsigma[len(nu):] = fsigma_P * np.ones(2*len(nu))
    noise_mat = np.matrix( np.diagflat(cmb_signal.flatten() * fsigma) )
    Ninv = np.linalg.inv(noise_mat)

    # Add noise to generated data
    D_vec += (np.matrix(np.random.randn(D_vec.size)) * noise_mat).T
    return D_vec, Ninv


#def model_test(nu, fsigma_T, fsigma_P, models_in, amps_in, params_in, models_fit, label):
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
    
    #guess = np.concatenate((dust_guess, sync_guess, cmb_guess))
    #ndim = len(dust_guess) + len(sync_guess) + len(cmb_guess)
    
    # Collect names, initial values, and parent components for the parameters
    param_spec = (pnames, initial_vals, parent_model)
    
    # Run MCMC sampler on this model
    t0 = time.time()
    params_out, pnames, samples = mcmc(data_spec, models_fit, param_spec, 
                                       burn=burn, steps=steps,
                                       sample_file=sample_file)
    print "MCMC run in %d sec." % (time.time() - t0)
    #import pylab as P
    #P.plot(samples)
    #P.show()
    
    # Estimate error on recovered CMB amplitudes
    #F_fg, F_cmb, F = F_matrix(nu, dust_params_out, sync_params_out, 
    #                          cmb_params_out, models_fit)
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

