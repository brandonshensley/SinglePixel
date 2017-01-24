import numpy as np
from scipy import linalg
from scipy import interpolate
import pickle, os

# For emcee MCMC
import emcee
import corner
import time

# For Plotting
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.font_manager as font_manager

# Physical constants
c = 2.99792458e10       # Speed of light, cm/s
h = 6.62606957e-27      # Planck constant, erg s
k = 1.3806488e-16       # Boltzmann constant, erg/K
Tcmb = 2.7255           # CMB temperature, K

# Plotting parameters
res_dpi = 300
ext = 'png'
pdir = './'

def prep_figure(num_rows,num_cols,hgap,vgap,width,height):
    # Prepare figure
    horizontal_spacing = 0.4*hgap
    vertical_spacing = 0.4*vgap
    figure_width = width
    figure_height = height
    plt.figure(figsize=(figure_width,figure_height))
    plot_left_boundary = 0.1
    plot_right_boundary = 0.9
    # Prepare two "grids" (intermediate between figures and subfigures)
    plot_grid = gridspec.GridSpec(num_rows, num_cols)
    # Set grids' left and right boundaries to separate them
    plot_grid.update(left=plot_left_boundary, right=plot_right_boundary)
    # Divide the grid for plots into enough axes, and apply spacing
    plot_axes = [[plt.subplot(plot_grid[row,col]) for col in range(num_cols)] for row in range(num_rows)]
    plt.subplots_adjust(wspace=horizontal_spacing, hspace=vertical_spacing)
    return plot_axes

    res_dpi = 300
    ext = 'png'
    pdir = './'

# Planck function
def B_nu (nu, T):
    return 2.0*h*(nu**3)/(c*c*(np.expm1(h*nu/(k*T))))

# Conversion to K_CMB
def G_nu (nu, T):
    x = h*nu/(k*T)
    return B_nu(nu,T)*x*np.exp(x)/(np.expm1(x)*T)

# Simple MBB dust model
def dust_model (nu, dust_params):
    (beta, Td) = dust_params
    nu_ref = 353.*1.e9
    dust_I = (nu/nu_ref)**beta*B_nu(nu, Td)*G_nu(nu_ref, Tcmb)/(B_nu(353.*1.e9,20.)*G_nu(nu,Tcmb))
    dust_Q = dust_I
    dust_U = dust_I
    return np.array([dust_I, dust_Q, dust_U])

# Physical dust model
def initialize_hd_dust_model ():
    # Read in precomputed dust emission spectra as a function of lambda and U
    data_sil = np.genfromtxt("sil_DH16_Fe_00_2.0.dat")
    data_silfe = np.genfromtxt("sil_DH16_Fe_05_2.0.dat")
    data_cion = np.genfromtxt("cion_1.0.dat")
    data_cneu = np.genfromtxt("cneu_1.0.dat")

    wav = data_sil[:,0]
    uvec = np.linspace(-0.5,0.5,num=11,endpoint=True)
    sil_i = interpolate.RectBivariateSpline(uvec,wav,((data_sil[:,4:15]/(9.744e-27))*
                                     (wav[:,np.newaxis]*1.e-4/c)*1.e23).T) # to Jy
    car_i = interpolate.RectBivariateSpline(uvec,wav,(((data_cion[:,4:15] + data_cneu[:,4:15])/(2.303e-27))*
                                     (wav[:,np.newaxis]*1.e-4/c)*1.e23).T) # to Jy
    silfe_i = interpolate.RectBivariateSpline(uvec,wav,((data_silfe[:,4:15]/(9.744e-27))*
                                     (wav[:,np.newaxis]*1.e-4/c)*1.e23).T) # to Jy

    sil_p = interpolate.RectBivariateSpline(uvec,wav,((data_sil[:,15:26]/(9.744e-27))*
                                     (wav[:,np.newaxis]*1.e-4/c)*1.e23).T) # to Jy
    car_p = interpolate.RectBivariateSpline(uvec,wav,(((data_cion[:,15:26] + data_cneu[:,4:15])/(2.303e-27))*
                                     (wav[:,np.newaxis]*1.e-4/c)*1.e23).T) # to Jy
    silfe_p = interpolate.RectBivariateSpline(uvec,wav,((data_silfe[:,15:26]/(9.744e-27))*
                                     (wav[:,np.newaxis]*1.e-4/c)*1.e23).T) # to Jy

    return (car_i, sil_i, silfe_i, car_p, sil_p, silfe_p)
    

def dust_model_hd (nu, dust_params):
    (dust_interp, fcar, fsilfe, uval) = dust_params
    (car_i, sil_i, silfe_i, car_p, sil_p, silfe_p) = dust_interp
    nu_ref = 353.*1.e9
    lam = 1.e4*c/nu # in microns
    lam_ref = 1.e4*c/nu_ref # in microns
    unit_fac = G_nu(nu_ref, Tcmb)/G_nu(nu, Tcmb)
    dust_I = unit_fac*(sil_i.ev(uval, lam) + fcar*car_i.ev(uval, lam) + fsilfe*silfe_i.ev(uval, lam))/ \
      (sil_i.ev(uval, lam_ref) + fcar*car_i.ev(uval, lam_ref) + fsilfe*silfe_i.ev(uval, lam_ref))
    dust_Q = unit_fac*(sil_p.ev(uval, lam) + fcar*car_p.ev(uval, lam) + fsilfe*silfe_p.ev(uval, lam))/ \
      (sil_p.ev(uval, lam_ref) + fcar*car_p.ev(uval, lam_ref) + fsilfe*silfe_p.ev(uval, lam_ref))
    dust_U = dust_Q
    return np.array([dust_I, dust_Q, dust_U])
    

# Simple power law synchrotron model
def sync_model (nu, sync_params) :
    (beta) = sync_params
    nu_ref = 30.*1.e9
    sync_I = (nu/nu_ref)**beta*G_nu(nu_ref,Tcmb)/G_nu(nu,Tcmb)
    sync_Q = sync_I
    sync_U = sync_I
    return np.array([sync_I, sync_Q, sync_U])

# Simple power law free-free model
def ff_model (nu, ff_params) :
    nu_ref = 30.*1.e9
    ff_I = (nu/nu_ref)**-0.118*G_nu(nu_ref,Tcmb)/G_nu(nu,Tcmb)
    ff_Q = ff_I
    ff_U = ff_I
    return np.array([ff_I, ff_Q, ff_U])

# CMB model-- spectrally flat
def cmb_model (nu, cmb_params) :
    cmb_I = np.ones(len(nu))
    cmb_Q = cmb_I
    cmb_U = cmb_I
    return np.array([cmb_I, cmb_Q, cmb_U])

# Assemble the spectral parameter matrices
def F_matrix (nu, dust_params, sync_params, cmb_params, models):
    F_fg = np.zeros((3*len(nu), 3*2))
    F = np.zeros((3*len(nu), 3*3))

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
    
    return (np.matrix(F_fg),np.matrix(F_cmb),np.matrix(F))

# Priors
# emcee
def ln_prior(beta):
    (dust_beta, dust_Td, sync_beta) = beta
    if (5. < dust_Td < 100. and 0. < dust_beta < 3. and -5. < sync_beta < 0.):
        return 0.0
    return -np.inf

# Log likelihood
def lnprob(beta, nu, D, Ninv, beam_mat, models_fit):
    dust_params = beta[0:2]
    sync_params = beta[2:3]
    cmb_params = np.array([])
    
    lp = ln_prior(beta)
    if not np.isfinite(lp):
        return -np.inf
                         
    (F_fg, F_cmb, F) = F_matrix(nu, dust_params, sync_params, cmb_params, models_fit)
    H = F_fg.T*Ninv*F_fg

    x_mat = np.linalg.inv(F.T*beam_mat.T*Ninv*beam_mat*F)*F.T*beam_mat.T*Ninv*D # Equation A3

    chi_square = (D - beam_mat*F*x_mat).T*Ninv*(D - beam_mat*F*x_mat) # Equation A4
    
    return lp - chi_square - 0.5*np.log(np.linalg.det(H))


# Generate mock data
def gen_data(nu, fsigma_T, fsigma_P, models_in, amps_in, params_in):
    # Create the data vector
    (dust_params, sync_params, cmb_params) = params_in
    (dust_amp, sync_amp, cmb_amp) = amps_in

    if (models_in[0] == 'mbb'):
        dust_in = dust_model(nu, dust_params)*dust_amp[:,np.newaxis]
    elif (models_in[0] == 'hd'):
        dust_in = dust_model_hd(nu, dust_params)*dust_amp[:,np.newaxis]
    else :
        print 'Error! Dust model ' + models[0] + ' not recognized!'
        exit()

    if (models_in[1] == 'pow'):
        sync_in = sync_model(nu, sync_params)*sync_amp[:,np.newaxis]
    else :
        print 'Error! Synchrotron model ' + models[1] + ' not recognized!'
        exit()

    cmb_in = cmb_model(nu, cmb_params)*cmb_amp[:,np.newaxis]

    D_vec = np.matrix((dust_in + sync_in + cmb_in).flatten()).T
        
    # Noise Model
    fsigma = np.zeros(3*len(nu))
    fsigma[0:len(nu)] = fsigma_T*np.ones(len(nu))
    fsigma[len(nu):] = fsigma_P*np.ones(2*len(nu))
    noise_mat = np.matrix(np.diagflat((cmb_in).flatten()*fsigma))
    Ninv = np.linalg.inv(noise_mat)

    # Add noise to generated data
    D_vec += (np.matrix(np.random.randn(len(D_vec)))*noise_mat).T

    return (D_vec, Ninv)


def mcmc(guess, nu, D, Ninv, beam_mat, models_fit, label=None, nwalkers=50, burn=500, steps=1000, save=False):
    """
    Run MCMC to fit model to some simulated data.
    """
    # Define starting points
    ndim = guess.size
    pos = [guess*(1.+0.1*np.random.randn(ndim)) for i in range(nwalkers)]
    
    # Run emcee sampler
    sampler = emcee.EnsembleSampler( nwalkers, ndim, lnprob, 
                                     args=(nu, D, Ninv, beam_mat, models_fit) )
    sampler.run_mcmc(pos, burn+steps)
    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
    
    """
    # Plot the chains, including burn-in
    fig, axes = plt.subplots(figsize=(13, ndim*3.), nrows=ndim)
    labels = [r'$\beta$', r'$T_d$', r'$\alpha_s$']

    for i, (ax, lab) in enumerate(zip(axes, labels)):
        _ = ax.plot(sampler.chain[:,:,i].T, c='k', alpha=0.2)
        ax.axhline(y=guess[i], c='r')
        ax.axvline(x=burn, c='r')
        ax.set_title(lab)
    fig.savefig('chains_' + label + '.png')
    plt.close('all')
    """

    # Save chains to file
    if save:
        np.savetxt('samples.dat', samples)
    
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
    
    # Return summary statistics and samples
    return dust_params, sync_params, cmb_params, samples


# Multinest Call
def multinest (nu, D, Ninv, beam_mat, ndim, models_fit, label):
    
    import pymultinest
    import json
    
    if not os.path.exists("chains"): os.mkdir("chains")
    parameters=["dust_beta","dust_Td","sync_beta"]
    n_params = len(parameters)

    def prior_multi(cube, ndim, nparams):
        cube[0]=0+3*cube[0]
        cube[1]=5+95*cube[1]
        cube[2]=-5+5*cube[2]

    def loglike_multi(cube, ndim, nparams):
        dust_beta, dust_Td, sync_beta = cube[0],cube[1],cube[2]
        dust_params = np.array([dust_beta, dust_Td])
        sync_params = np.array([sync_beta])
        cmb_params = np.array([])
        (F_fg, F_cmb, F) = F_matrix(nu, dust_params, sync_params, cmb_params, models_fit)
        H = F_fg.T*Ninv*F_fg

        x_mat = np.linalg.inv(F.T*beam_mat.T*Ninv*beam_mat*F)*F.T*beam_mat.T*Ninv*D # Equation A3

        chi_square = (D - beam_mat*F*x_mat).T*Ninv*(D - beam_mat*F*x_mat) # Equation A4
    
        return -chi_square - 0.5*np.log(np.linalg.det(H))
    
    pymultinest.run(loglike_multi, prior_multi, n_params, outputfiles_basename='chains/single_pixel_',
                        resume=False, verbose=True,n_live_points=1000,
                        importance_nested_sampling=False)
    a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='chains/single_pixel_')
    s = a.get_stats()

    output=a.get_equal_weighted_posterior()
    outfile='test.out'
    pickle.dump(output,open(outfile,"wb"))

    # store name of parameters, always useful
    with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
        json.dump(parameters, f, indent=2)
    # store derived stats
    with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
        json.dump(s, f, indent=2)
    print()
    print("-" * 30, 'ANALYSIS', "-" * 30)
    print("Global Evidence:\n\t%.15e +- %.15e" %
              ( s['nested sampling global log-evidence'],
                    s['nested sampling global log-evidence error'] ))

    # Plots
    p = pymultinest.PlotMarginalModes(a)
    plt.figure(figsize=(5*n_params, 5*n_params))

    for i in range(n_params):
        plt.subplot(n_params, n_params, n_params * i + i + 1)
        p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
        plt.ylabel("Probability")
        plt.xlabel(parameters[i])
        
        for j in range(i):
            plt.subplot(n_params, n_params, n_params * j + i + 1)
            p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
            plt.xlabel(parameters[i])
            plt.ylabel(parameters[j])

    plt.savefig("chains/single_pixel_marginals_multinest.pdf") #, bbox_inches='tight')

    for i in range(n_params):
        outfile = '%s-mode-marginal-%d.pdf' % (a.outputfiles_basename,i)
        p.plot_modes_marginal(i, with_ellipses = True, with_points = False)
        plt.ylabel("Probability")
        plt.xlabel(parameters[i])
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()
        
        outfile = '%s-mode-marginal-cumulative-%d.pdf' % (a.outputfiles_basename,i)
        p.plot_modes_marginal(i, cumulative = True, with_ellipses = True, with_points = False)
        plt.ylabel("Cumulative probability")
        plt.xlabel(parameters[i])
        plt.savefig(outfile, format='pdf', bbox_inches='tight')
        plt.close()
    

def model_test(nu, fsigma_T, fsigma_P, models_in, amps_in, params_in, models_fit, label):
    """
    Generate simulated data given an input model, and perform MCMC fit using 
    another model.
    """
    # Generate fake data with some "true" parameters
    (D_vec, Ninv) = gen_data(nu, fsigma_T, fsigma_P, models_in, amps_in, params_in)
    Ninv_sqrt = np.matrix(linalg.sqrtm(Ninv))
    (dust_params, sync_params, cmb_params) = params_in
    (dust_amp, sync_amp, cmb_amp) = amps_in
    
    # Beam model
    beam_mat = np.identity(3*len(nu))

    # Set-up MCMC
    dust_guess = np.array([1.6, 20.])
    sync_guess = np.array([-3.])
    cmb_guess = np.array([])
    guess = np.concatenate((dust_guess, sync_guess, cmb_guess))
    #ndim = len(dust_guess) + len(sync_guess) + len(cmb_guess)
    
    # Run MCMC sampler on this model
    t0 = time.time()
    dust_params_out, sync_params_out, cmb_params_out, samples \
        = mcmc(guess, nu, D_vec, Ninv, beam_mat, models_fit, label)
    print "MCMC run in %d sec." % (time.time() - t0)
    
    # Estimate error on recovered CMB amplitudes
    (F_fg, F_cmb, F) = F_matrix(nu, dust_params_out, sync_params_out, cmb_params_out, models_fit)
    H = F_fg.T*Ninv*F_fg
    x_mat = np.linalg.inv(F.T*beam_mat.T*Ninv*beam_mat*F)*F.T*beam_mat.T*Ninv*D_vec # Equation A3
    
    U, Lambda, VT = np.linalg.svd(Ninv_sqrt*F_fg, full_matrices=False) # Equation A14
    N_eff_inv_cmb = F_cmb.T*Ninv_sqrt*(np.matrix(np.identity(U.shape[0])) - U*U.T)*Ninv_sqrt*F_cmb # Equation A16
    N_eff_cmb = np.linalg.inv(N_eff_inv_cmb)
    cmb_noise = np.array([N_eff_cmb[0,0], N_eff_cmb[1,1], N_eff_cmb[2,2]])

    gls_cmb = x_mat[0:3,0]
    cmb_chisq = (np.matrix(cmb_amp).T - gls_cmb).T*N_eff_inv_cmb*(np.matrix(cmb_amp).T - gls_cmb)
    
    # Output triangle plots for dust
    if label != None:
        if (models_fit[0] == 'mbb' and models_fit[1] == 'pow'):
            if (models_in[0] == 'mbb'):
                fig = corner.corner(samples, truths=[dust_params[0], dust_params[1], sync_params[0]],
                                        labels=[r"$\beta_d$", r"$T_d$",r"$\alpha_s$"])
            else :
                fig = corner.corner(samples, labels=[r"$\beta_d$", r"$T_d$",r"$\alpha_s$"])
        else :
            print 'Error! Not configured for this plot!'
            exit()
        fig.savefig('triangle_' + label + '.png')
        plt.close('all')
    
    # Run multinest sampler
    #multinest(nu, D_vec, Ninv, beam_mat, ndim, models_fit, label)
    
    return gls_cmb, cmb_chisq, cmb_noise


"""
def main():
    # Main parameters:

    # nfreq = Number of frequency channels
    # fsigma_T = Noise to signal in each band (Temperature)
    # fsigma_P = Noise to signal in each band (Polarization)
    
    # models_in = Models to generate the data
    # amps_in = Input component amplitudes
    # params_in = Input component spectral parameters
    # models_fit = Models to fit the data
    
    models_in = np.array(['mbb', 'pow'])
    models_fit = np.array(['mbb', 'pow'])
    nfreq = 10
    fsigma_T = 0.01
    fsigma_P = 0.01

    # Fiducial amplitudes
    dust_I_amp = 10.*2.*(353.*1.e9)**2*k/(c**2*G_nu(353.*1.e9,Tcmb)) # 10uK_RJ at 353 GHz to uK_CMB
    dust_Q_amp = 0.1*dust_I_amp
    dust_U_amp = 0.12*dust_I_amp
    dust_amp = np.array([dust_I_amp, dust_Q_amp, dust_U_amp])

    sync_I_amp = 10.*2.*(30.*1.e9)**2*k/(c**2*G_nu(30.*1.e9,Tcmb)) # 10uK_RJ at 30 GHz to uK_CMB
    sync_Q_amp = sync_I_amp*0.2
    sync_U_amp = sync_I_amp*0.15
    sync_amp = np.array([sync_I_amp, sync_Q_amp, sync_U_amp])

    cmb_I_amp = 100. # 100uK_CMB
    cmb_Q_amp = 10.
    cmb_U_amp = 20.
    cmb_amp = np.array([cmb_I_amp, cmb_Q_amp, cmb_U_amp])

    amps_in = np.array([dust_amp, sync_amp, cmb_amp])

    # Input spectral parameters
    dust_beta = 1.6
    dust_T = 20.
    fcar = 1.
    fsilfe = 0.
    uval = 0.
    dust_interp = initialize_hd_dust_model()
    dust_params_in = np.array([dust_beta, dust_T])
    #dust_params_in = np.array([dust_interp, fcar, fsilfe, uval])
    
    sync_beta = -3.
    sync_params_in = np.array([sync_beta])

    cmb_params_in = np.array([])

    params_in = np.array([dust_params_in, sync_params_in, cmb_params_in])
    

    filename = 'model_optimize.dat'
    outf = open(filename, 'w')
    
    freq_vec = np.arange(nfreq)
    #nu_min_array = np.array([20., 30., 40., 50., 60., 70.])
    nu_min_array = np.array([20.])
    #nu_max_array = np.array([100., 200, 300., 400, 500., 600, 700., 800, 900.])
    nu_max_array = np.array([500.])
    for nu_min in nu_min_array:
        for nu_max in nu_max_array:
            nu = nu_min*(nu_max/nu_min)**(freq_vec/(nfreq-1.))*1.e9
            label = str(nu_min) + '_' + str(nu_max)
            (gls_cmb, cmb_chisq, cmb_noise) = model_test(nu, fsigma_T, fsigma_P, models_in, amps_in, params_in, models_fit, label)
            outf.write(9*('%0.6e ') % (nu_min, nu_max, cmb_chisq, gls_cmb[0],
                                           gls_cmb[1], gls_cmb[2], cmb_noise[0]/cmb_I_amp,
                                           cmb_noise[1]/cmb_Q_amp, cmb_noise[2]/cmb_U_amp))
            outf.write('\n')
    outf.close()

    
                    
if __name__ == '__main__':
     main()
"""
