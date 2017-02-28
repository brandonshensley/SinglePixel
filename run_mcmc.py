#!/usr/bin/python

import numpy as np
import matplotlib
#matplotlib.use('Agg') # Need this to avoid using X server
import pylab as plt
import single_pixel as fg
import models
import fitting
from multiprocessing import Pool

# Band parameter definitions
nbands = 7
NPROC = 4 #32
SEED = 10

np.random.seed(SEED)

#numin_vals = [5., 10., 20., 30., 40., 50., 60., 70.]
#numax_vals = [200., 300., 400., 500., 600., 700.]
numin_vals = [5., ] #10., 20., 30., 40., 50., 60., 70.]
numax_vals = [100.,] # 300., 400., 500., 600., 700.]

# Temperature/polarisation noise rms for all bands, as a fraction of T_cmb
fsigma_T = 0.01
fsigma_P = 0.01

# Define input models and their amplitudes/parameters
dust_model = models.DustMBB(amp_I=10., amp_Q=1., amp_U=1.2, dust_beta=1.6, dust_T=20.)
sync_model = models.SyncPow(amp_I=10., amp_Q=2., amp_U=1.5, sync_beta=-3.)
cmb_model = models.CMB(amp_I=100., amp_Q=10., amp_U=20.)

name_in = "MBBSync"
mods_in = [dust_model, sync_model, cmb_model]
amps_in = np.array([m.amps() for m in mods_in])
params_in = np.array([m.params() for m in mods_in])

# Define models to use for the fitting
name_fit = "MBBSync"
mods_fit = [dust_model, sync_model, cmb_model]



def bands_log(nu_min, nu_max, nbands):
    """
    Logarithmic set of bands.
    """
    freq_vec = np.arange(nbands)
    return nu_min * (nu_max/nu_min)**(freq_vec/(nbands-1.)) * 1e9 # in Hz

def bands_3groups(nu_low, nu_mid, nu_high, sep_low, sep_mid, sep_high, nbands):
    """
    Divide bands between three groups: low-, mid-, and high-frequency. Each 
    group is specified by a minimum frequency. Spacings are linear within each 
    group.
    """
    # Divide available bands into 3 groups
    nlow = nbands // 3
    nmid = nbands // 3
    nhigh = nbands // 3
    
    # Assign remainder bands into mid- and high-freq. groups respectively
    nleft = nbands - (nlow + nmid + nhigh)
    if nleft == 2:
        nmid += 1
        nhigh += 1
    elif nleft == 1:
        nmid += 1
    else:
        pass
    
    # Define bands
    nu = np.concatenate( [nu_low + sep_low * np.arange(nlow),
                          nu_mid + sep_mid * np.arange(nmid),
                          nu_high + sep_high * np.arange(nhigh),] )
    return nu * 1e9 # in Hz


# FIXME: Testing
#nu = np.array([10., 70., 140., 200.])
#D_vec, Ninv = fitting.generate_data(nu, fsigma_T, fsigma_P, components=mods)
#fitting.model_test(nu, D_vec, Ninv, mods)


# Expand into all combinations of nu_min,max
nu_min, nu_max = np.meshgrid(numin_vals, numax_vals)
nu_params = np.column_stack((nu_min.flatten(), nu_max.flatten()))

# Prepare output file for writing
filename = "output/summary_%s.%s_nb%d_seed%d.dat" % (name_in, name_fit, nbands, SEED)
f = open(filename, 'w')
f.write("# nu_min, nu_max, cmb_chisq, gls_cmb_I, gls_cmb_Q, gls_cmb_U, sig(I)/I, sig(Q)/Q, sig(U)/U\n")
f.close()

def run_model(nu_params):
    # Get band definition
    nu_min, nu_max = nu_params
    print "nu_min = %d GHz, nu_max = %d GHz" % (nu_min, nu_max)
    nu = bands_log(nu_min, nu_max, nbands)
    label = str(nu_min) + '_' + str(nu_max)
    
    # Name of sample file
    fname_samples = "output/samples_%s.%s_nb%d_seed%d_%s.dat" \
                  % (name_in, name_fit, nbands, SEED, label)
    
    # Simulate data and run MCMC fit
    D_vec, Ninv = fitting.generate_data(nu, fsigma_T, fsigma_P, 
                                        components=mods_in)
                                        
    gls_cmb, cmb_chisq, cmb_noise \
            = fitting.model_test(nu, D_vec, Ninv, mods_fit, 
                                 burn=200, steps=1000,
                                 cmb_amp_in=cmb_model.amps(),
                                 sample_file=fname_samples)
    
    print "chisq =", cmb_chisq, gls_cmb.flatten()
    
    # Append summary statistics to file
    f = open(filename, 'a')
    f.write(9*('%0.6e ') % (nu_min, nu_max, cmb_chisq, 
                            gls_cmb[0], gls_cmb[1], gls_cmb[2], 
                            cmb_noise[0]/cmb_model.amp_I,
                            cmb_noise[1]/cmb_model.amp_Q, 
                            cmb_noise[2]/cmb_model.amp_U) )
    f.write('\n')
    f.close()

# Run pool of processes
pool = Pool(NPROC)
pool.map(run_model, nu_params)
