
import numpy as np
from scipy.linalg import sqrtm
import copy, time
import emcee

def map_params_to_models():
    """
    Given an ordered list of parameter values, update their value in a set of 
    models.
    """
    

def ln_prior(beta):
    (dust_beta, dust_Td, sync_beta) = beta
    if (5. < dust_Td < 100. and 0. < dust_beta < 3. and -5. < sync_beta < 0.):
        return 0.0
    return -np.inf

def lnprob(pvals, data_spec, models_fit, param_spec):
    """
    log-probability (likelihood times prior) for a set of parameter values.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec
    pnames, initial_vals, parent_model = param_spec
    
    #dust_params = beta[0:2]
    #sync_params = beta[2:3]
    #cmb_params = np.array([])
    
    
    # FIXME: 
    logpr = ln_prior([1., 10., -3.]) # FIXME
    if not np.isfinite(logpr): return -np.inf
    
    F_fg, F_cmb, F = F_matrix(pvals, nu, models_fit, param_spec)
    #F_fg, F_cmb, F = F_matrix(nu, dust_params, sync_params, cmb_params, models_fit)
    H = F_fg.T*Ninv*F_fg

    x_mat = np.linalg.inv(F.T * beam_mat.T * Ninv * beam_mat * F) \
          * F.T * beam_mat.T * Ninv * D # Equation A3

    chi_square = (D - beam_mat * F * x_mat).T \
               * Ninv * (D - beam_mat * F * x_mat) # Equation A4
    
    return logpr - chi_square - 0.5*np.log(np.linalg.det(H))


def F_matrix (pvals, nu, models_fit, param_spec):
    """
    """
    pnames, initial_vals, parent_model = param_spec
    
    # Check that th CMB component is the last component in the model list
    if models_fit[-1].model != 'cmb':
        raise ValueError("The last model in the models_fit list should be a "
                         "CMB() object.")
    
    #F_fg = np.zeros((3*len(nu), 3*2))
    #F = np.zeros((3*len(nu), 3*3))
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
        
        # Set parameter values
        n = m.params().size
        m.set_params( pvals[pstart:pstart+n] ) # FIXME: Test this
        pstart += n # Increment for next model
        
        # Calculate scaling with freq. given new parameter values
        scal = m.scaling(nu)
        print scal.shape
        for j in range(Npol):
            # Fill FG or CMB -matrix with scalings, as appropriate
            if m.model != 'cmb':
                # FIXME: Is the ordering right?
                #print "F_fg[%d, %d]"
                F_fg[j*Nband:(j+1)*Nband, i*Npol + j] = scal[j,:]
            else:
                F_cmb[j*Nband:(j+1)*Nband, j] = scal[j,:]
    
    # FIXME
    np.savetxt("NEW_F.dat", F)
    np.savetxt("NEW_Ffg.dat", F_fg)
    np.savetxt("NEW_Fcmb.dat", F_cmb)
    import sys
    sys.exit()
    
    # Stack CMB and FG F-matrices together
    F = np.hstack((F_cmb, F_fg))
    
    """
    if (models[0] == 'mbb'):
        dust = dust_model(nu, dust_params).T
    elif (models[0] == 'hd'):
        dust = dust_model_hd(nu, dust_params).T
    else :
        print 'Error! Dust model ' + models[0] + ' not recognized!'
        exit()

    if (models[1] == 'pow'):
        sync = sync_model(nu, sync_params).T
    else :
        print 'Error! Synchrotron model ' + models[1] + ' not recognized!'
        exit()
        
    cmb = cmb_model(nu, cmb_params).T
    
    nnu = len(nu)
    for i in range(nnu):
        F_fg[i,0] = dust[i,0]
        F_fg[i+nnu,1] = dust[i,1]
        F_fg[i+2*nnu,2] = dust[i,2]
        F_fg[i,3] = sync[i,0]
        F_fg[i+nnu,4] = sync[i,1]
        F_fg[i+2*nnu,5] = sync[i,2]

    F_cmb = np.zeros((3*len(nu), 3))
    for i in range(nnu):
        F_cmb[i,0] = cmb[i,0]
        F_cmb[i+nnu,1] = cmb[i,1]
        F_cmb[i+2*nnu,2] = cmb[i,2]

    F = np.hstack((F_cmb, F_fg))
    """
    return np.matrix(F_fg), np.matrix(F_cmb), np.matrix(F)


def mcmc(data_spec, models_fit, param_spec, nwalkers=50, 
         burn=500, steps=1000, save=False):
    """
    Run MCMC to fit model to some simulated data.
    """
    # Retrieve instrument/data model and parameter info
    nu, D_vec, Ninv, beam_mat = data_spec
    pnames, initial_vals, parent_model = param_spec
    
    # Define starting points
    ndim = len(initial_vals)
    pos = [initial_vals*(1.+0.1*np.random.randn(ndim)) for i in range(nwalkers)]
    
    # Run emcee sampler
    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob, 
                                     args=(data_spec, models_fit, param_spec) )
    sampler.run_mcmc(pos, burn + steps)
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    
    # FIXME: got to here
    
    # Save chains to file
    #if save:
    #    np.savetxt('samples.dat', samples)
    
    """
    # Summary statistics for fitted parameters
    # Dust model
    beta_out = np.median(samples, axis=0)
    ndust = 0
    if (models_fit[0] == 'mbb'):
        ndust = 2
        dust_params = beta_out[0:2]
    else:
        raise NotImplementedError("Code not configured to fit HD16 models.")
    
    # Synchrotron power law
    if (models_fit[1] == 'pow'):
        sync_params = beta_out[ndust:ndust+1]
    else:
        raise ValueError("Synchrotron model '%s' not recognized." % models_fit[1])
    
    # CMB
    cmb_params = np.array([])
    """
    
    # Return summary statistics and samples
    return dust_params, sync_params, cmb_params, samples


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
        
        # Add this component to total signal
        signal += comp.amps() * comp.scaling(nu)
        
        # Store CMB signal separately
        if comp.model == 'cmb':
            cmb_signal = comp.amps() * comp.scaling(nu)
    
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
def model_test(nu, D_vec, Ninv, models_fit, initial_vals=None):
    """
    Generate simulated data given an input model, and perform MCMC fit using 
    another model.
    """
    # Collect together data and noise/instrument model
    Ninv_sqrt = np.matrix(sqrtm(Ninv)) # Invert noise covariance matrix
    beam_mat = np.identity(3*len(nu)) # Beam model
    data_spec = (nu, D_vec, Ninv, beam_mat)
    
    # Loop over specified models and set up MCMC parameters for them
    pnames = []; pvals = []; parent_model = []
    for mod in models_fit:
        # Get parameter names, initial parameter values, and module ID
        pn = mod.param_names
        pv = mod.params()
        
        # Loop through parameter from this module
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
    params_out, samples = mcmc(data_spec, models_fit, param_spec)
    print "MCMC run in %d sec." % (time.time() - t0)
    
    exit() # FIXME
    
    # Estimate error on recovered CMB amplitudes
    #FIXME
    F_fg, F_cmb, F = F_matrix(nu, dust_params_out, sync_params_out, 
                              cmb_params_out, models_fit)
    H = F_fg.T * Ninv * F_fg
    
    # Equation A3
    x_mat = np.linalg.inv(F.T*beam_mat.T*Ninv*beam_mat*F)*F.T*beam_mat.T*Ninv*D_vec
    
    # Equation A14
    U, Lambda, VT = np.linalg.svd(Ninv_sqrt*F_fg, full_matrices=False)
    
    # Equation A16
    N_eff_inv_cmb = F_cmb.T * Ninv_sqrt * Ninv_sqrt * F_cmb \
                  * (np.matrix(np.identity(U.shape[0])) - U*U.T)
    N_eff_cmb = np.linalg.inv(N_eff_inv_cmb)
    cmb_noise = np.array([N_eff_cmb[0,0], N_eff_cmb[1,1], N_eff_cmb[2,2]])

    gls_cmb = x_mat[0:3,0]
    cmb_chisq = (np.matrix(cmb_amp).T - gls_cmb).T*N_eff_inv_cmb*(np.matrix(cmb_amp).T - gls_cmb)
    
    return gls_cmb, cmb_chisq, cmb_noise

