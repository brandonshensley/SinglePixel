from __future__ import absolute_import, unicode_literals, print_function
import pymultinest
import math, os
import threading, subprocess
#from fm import *
import pdb
import numpy as np
import pickle
#import triangle2
from matplotlib.pyplot import *
from scipy import linalg
#import emcee
import corner
import time

if not os.path.exists("chains"): os.mkdir("chains")

def show(filepath):
        """ open the output (pdf) file for the user """
        if os.name == 'mac': subprocess.call(('open', filepath))
        elif os.name == 'nt': os.startfile(filepath)
        elif os.name == 'posix': subprocess.call(('xdg-open', filepath))



# Physical constants
c = 2.99792458e10       # Speed of light, cm/s
h = 6.62606957e-27      # Planck constant, erg s
k = 1.3806488e-16       # Boltzmann constant, erg/K
Tcmb = 2.7255           # CMB temperature, K

## MCMC parameters
#nwalkers = 50
#burn = 500
#steps = 1000

# Planck function
def B_nu (nu, T):
    np.seterr(over='ignore')
    return 2.0*h*(nu**3)/(c*c*(np.expm1(h*nu/(k*T))))

# Conversion to K_CMB
def G_nu (nu, T):
    np.seterr(over='ignore')
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

# Simple power law synchrotron model
def sync_model (nu, sync_params) :
    (beta) = sync_params
    nu_ref = 30.*1.e9
    sync_I = (nu/nu_ref)**beta*G_nu(nu_ref,Tcmb)/G_nu(nu,Tcmb)
    sync_Q = sync_I
    sync_U = sync_I
    return np.array([sync_I, sync_Q, sync_U])

# CMB model-- spectrally flat
def cmb_model (nu, cmb_params) :
    cmb_I = np.ones(len(nu))
    cmb_Q = cmb_I
    cmb_U = cmb_I
    return np.array([cmb_I, cmb_Q, cmb_U])

# Assemble the spectral parameter matrices
def F_matrix (nu, dust_params, sync_params, cmb_params):
    F_fg = np.zeros((3*len(nu), 3*2))
    F = np.zeros((3*len(nu), 3*3))
    
    dust = dust_model(nu, dust_params).T
    sync = sync_model(nu, sync_params).T
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
#def ln_prior(beta):
#    (dust_beta, dust_Td, sync_beta) = beta
#    if (5. < dust_Td < 2000. and 0. < dust_beta < 3. and -5. < sync_beta < 0.):
#        return 0.0
#    return -np.inf


#prior--only use this to transform "unit (0-1)" coordinates to
#physical values...
#for instance to transform something to have a value between -6 and 6
#would require -6+12*cube[0]. When cube[0]=0, then the transformed
#cube[0] will equal -6+12.*0, or -6. When cube[0]=1, then the transformed cube[0]
#will be -6+12*1, or 6.
#define the prior cube only for the parameters to be retrieved
# here for: 
# dust_Td, dust_beta, sync_beta

def prior(cube, ndim, nparams):  # where are ndim and nparams defined? check this !!!!
    #print ('cube before', [cube[i] for i in range(ndim)])

    cube[0]=5+1995*cube[0]
    cube[1]=0+3*cube[1]
    cube[2]=-5+5*cube[2]

    #print ('python cube after', [cube[i] for i in range(ndim)])
    #print ('in prior: dust_Td, dust_beta, sync_beta=',cube[0],cube[1],cube[2])

    if (5. < cube[0] < 2000. and 0. < cube[1] < 3. and -5. < cube[2] < 0.):
        return 0.0
        #print('in prior,lp=',0.0)
    return -np.inf


# Log likelihood
#def lnprob(beta, nu, D, Ninv, beam_mat):
#    dust_params = beta[0:2]
#    sync_params = beta[2:3]
#    cmb_params = np.array([])
#
#    lp = ln_prior(beta)
#    if not np.isfinite(lp):
#        return -np.inf
#
#    (F_fg, F_cmb, F) = F_matrix(nu, dust_params, sync_params, cmb_params)
#    H = F_fg.T*Ninv*F_fg
#
#x_mat = np.linalg.inv(F.T*beam_mat.T*Ninv*beam_mat*F)*F.T*beam_mat.T*Ninv*D # Equation A3
#
#    chi_square = (D - beam_mat*F*x_mat).T*Ninv*(D - beam_mat*F*x_mat) # Equation A4
#
#    return lp - chi_square - 0.5*np.log(np.linalg.det(H))


# Log likelihood
#def lnprob(cube, nu, D, Ninv, beam_mat, ndim, nparams):

#def loglike(cube, nu, D, Ninv, beam_mat, ndim, nparams):
def loglike(cube, ndim, nparams):

    #lp=prior(cube, ndim, nparams)
    #print('in loglike, lp=',lp)
    #if not np.isfinite(lp):
    #    return -np.inf
   
    #dust_params[0], dust_params[1], sync_params = cube[0],cube[1],cube[2]
    dust_Td, dust_beta, sync_beta = cube[0],cube[1],cube[2]
    #print ('in loglike: dust_Td, dust_beta, sync_beta=',cube[0],cube[1],cube[2])
    beta=(dust_beta, dust_Td, sync_beta)
    #print('beta=',beta)
    #dust_Td=dust_param[0]
    #dust_beta=dust_params[1]
    #sync_params=sync_beta
    #(dust_beta, dust_Td, sync_beta) = beta
    dust_params = beta[0:2]
    sync_params = beta[2:3]
    cmb_params = np.array([])
    print('dust_params,sync_params',dust_params, sync_params)

    #sys.exit()
    
    #lp = ln_prior(beta)  #is this ln or prior?- check this!!!!
    #lp = prior(cube, ndim, nparams) #is this ln or prior?- check this!!!!
    #lp=0.0
    #print('in loglike, lp=',lp)
    #sys.exit()
    #if not np.isfinite(lp):
    #    return -np.inf

    #sys.exit()
                         
    (F_fg, F_cmb, F) = F_matrix(nu, dust_params, sync_params, cmb_params)
    H = F_fg.T*Ninv*F_fg

    x_mat = np.linalg.inv(F.T*beam_mat.T*Ninv*beam_mat*F)*F.T*beam_mat.T*Ninv*D_vec # Equation A3

    chi_square = (D_vec - beam_mat*F*x_mat).T*Ninv*(D_vec - beam_mat*F*x_mat) # Equation A4
    
    #sys.exit()
    #print ('lp=',lp)
    lp=0.0
    print ('chi_square=',chi_square)
    print ('0.5*np.log(np.linalg.det(H))=', 0.5*np.log(np.linalg.det(H)))
    print ('lp - chi_square - 0.5*np.log(np.linalg.det(H))=',lp - chi_square - 0.5*np.log(np.linalg.det(H)))

    loglikelihood = lp - chi_square - 0.5*np.log(np.linalg.det(H))
    print('loglikelihood=',loglikelihood)
    #return lp - chi_square - 0.5*np.log(np.linalg.det(H))
    #sys.exit()
    return loglikelihood

# MCMC call
#def mcmc (guess, nu, D, Ninv, beam_mat, ndim):
#    pos = [guess*(1.+0.1*np.random.randn(ndim)) for i in range(nwalkers)]
#    sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(nu,D,Ninv,beam_mat))
#    sampler.run_mcmc(pos, burn+steps)
#    samples = sampler.chain[:, burn:, :].reshape((-1, ndim))
#
#    #Uncomment to save chains
#    np.savetxt('chains/samples.dat',samples)
#    beta_out = np.median(samples,axis=0)
#    dust_params = beta_out[0:2]
#    sync_params = beta_out[2:3]
#    cmb_params = np.array([])
#    return (dust_params, sync_params, cmb_params, samples)


#def main():
nu = np.array([15., 30., 60., 80., 100., 120., 240., 480., 960.])*1.e9 # in Hz

# Generate fake data with some "true" parameters
dust_I_amp = 10.*2.*(353.*1.e9)**2*k/(c**2*G_nu(353.*1.e9,Tcmb)) # 10uK_RJ at 353 GHz to uK_CMB
dust_Q_amp = 0.1*dust_I_amp
dust_U_amp = 0.12*dust_I_amp
dust_pol_angle = 0.5 * np.arctan(dust_U_amp/dust_Q_amp)
dust_beta = 1.6
dust_T = 20.

sync_I_amp = 20.*2.*(30.*1.e9)**2*k/(c**2*G_nu(30.*1.e9,Tcmb)) # 10uK_RJ at 30 GHz to uK_CMB
sync_Q_amp = sync_I_amp*0.2
sync_U_amp = sync_I_amp*0.15
sync_beta = -3.
sync_pol_angle = 0.5 * np.arctan(sync_U_amp/sync_Q_amp)

cmb_I_amp = 100. # 100uK_CMB
cmb_Q_amp = 10.
cmb_U_amp = 20.
cmb_pol_angle = 0.5 * np.arctan(cmb_U_amp/cmb_Q_amp)

# Create the data vector
dust_params = (dust_beta, dust_T)
sync_params = (sync_beta)
cmb_params = ()
dust_fac = np.array([dust_I_amp, dust_Q_amp, dust_U_amp])
sync_fac = np.array([sync_I_amp, sync_Q_amp, sync_U_amp])
cmb_fac = np.array([cmb_I_amp, cmb_Q_amp, cmb_U_amp])
    
D_vec = np.matrix((dust_model(nu, dust_params)*dust_fac[:,np.newaxis] +
                      sync_model(nu, sync_params)*sync_fac[:,np.newaxis] +
                      cmb_model(nu, cmb_params)*cmb_fac[:,np.newaxis]).flatten()).T

# Noise Model
fsigma = 0.01
noise_mat = np.matrix(np.diagflat(D_vec*fsigma))
Ninv = np.linalg.inv(noise_mat)

# Add noise to generated data
D_vec += (np.matrix(np.random.randn(len(D_vec)))*noise_mat).T
        
# Beam model
beam_mat = np.identity(3*len(nu))

    # MCMC
#    dust_guess = np.array([dust_beta, dust_T])
#    sync_guess = np.array([sync_beta])
#    cmb_guess = np.array([])
#    guess = np.concatenate((dust_guess, sync_guess, cmb_guess))
#    ndim = len(dust_guess) + len(sync_guess) + len(cmb_guess)
#    start = time.clock()
#    (dust_params_out, sync_params_out, cmb_params_out, samples) = mcmc(guess, nu, D_vec, Ninv, beam_mat, ndim)
#    end = time.clock()
#    print ('Time in seconds:',(end - start))
#
#    fig = corner.corner(samples, truths=[dust_beta, dust_T, sync_beta],labels=[r"$\beta_d$", r"$T_d$",r"$\alpha_s$"])
#    fig.savefig('plots/triangle.png')
  


# Multinest 
parameters=["dust_Td", "dust_beta","sync_beta"]
n_params = len(parameters)
print ('starting to run multinest')
print ('n_params=',n_params)

    #sys.exit()
# we want to see some output while it is running                                                               \

#progress = pymultinest.ProgressPlotter(n_params = n_params, outputfiles_basename='chains/TESS/run_nlive1000/pl\anet_01/tess_planet_01_'); progress.start()
    #threading.Timer(2, show, ["chains/TESS/run_nlive1000/planet01/tess_planet_01_phys_live.points.pdf"]).start() #delayed opening

# run MultiNest
pymultinest.run(loglike, prior, n_params, outputfiles_basename='chains/single_pixel_',resume=False, verbose=True,n_live_points=1000,importance_nested_sampling=False)
#    print('loglike=',loglike)
#    print('prior=',prior)


#ok, done. Stop our progress watcher                                                                           \

#progress.stop()

a = pymultinest.Analyzer(n_params = n_params, outputfiles_basename='chains/single_pixel_')
s = a.get_stats()

output=a.get_equal_weighted_posterior()
outfile='test.out'
pickle.dump(output,open(outfile,"wb"))


import json
# store name of parameters, always useful
with open('%sparams.json' % a.outputfiles_basename, 'w') as f:
        json.dump(parameters, f, indent=2)
# store derived stats
with open('%sstats.json' % a.outputfiles_basename, mode='w') as f:
        json.dump(s, f, indent=2)
print()
print("-" * 30, 'ANALYSIS', "-" * 30)
print("Global Evidence:\n\t%.15e +- %.15e" % ( s['nested sampling global log-evidence'], s['nested sampling global log-evidence error'] ))

import matplotlib.pyplot as plt
plt.clf()

#Multinest plotting

# Here we will plot all the marginals and whatnot, just to show off
# You may configure the format of the output here, or in matplotlibrc
# All pymultinest does is filling in the data of the plot.

# Copy and edit this file, and play with it.

p = pymultinest.PlotMarginalModes(a)
plt.figure(figsize=(5*n_params, 5*n_params))
#plt.subplots_adjust(wspace=0, hspace=0) 
for i in range(n_params):
          plt.subplot(n_params, n_params, n_params * i + i + 1)
          p.plot_marginal(i, with_ellipses = True, with_points = False, grid_points=50)
          plt.ylabel("Probability")
          plt.xlabel(parameters[i])
        
          for j in range(i):
                 plt.subplot(n_params, n_params, n_params * j + i + 1)
#plt.subplots_adjust(left=0, bottom=0, right=0, top=0, wspace=0, hspace=0)
		 p.plot_conditional(i, j, with_ellipses = False, with_points = True, grid_points=30)
		 plt.xlabel(parameters[i])
		 plt.ylabel(parameters[j])

plt.savefig("chains/single_pixel_marginals_multinest.pdf") #, bbox_inches='tight')
#show("chains/planet_01/marginals_multinest.pdf")

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

print("Take a look at the pdf files in chains/")
 
                    
#if __name__ == '__main__':
#     main()
