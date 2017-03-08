#!/usr/bin/python
"""
Do MCMC runs to fit FG models to simulated data, over a grid of 
(nu_min, nu_max) values.
"""
import numpy as np
import models
import fitting
import sys
from multiprocessing import Pool

# Reference noise curve
NOISE_FILE = "data/core_plus_extended_noise.dat"

# Band parameter definitions
nbands = 7
NPROC = 4 #32

SEED = 10
if len(sys.argv) > 1:
    SEED = int(sys.argv[1])
print "SEED =", SEED

np.random.seed(SEED)

numin_vals = [15., 20., 25., 30., 35., 40.]
numax_vals = [300., 400., 500., 600., 700., 800.]
#numin_vals = [5., 10., 20., 30., 40., 50., 60., 70.]
#numax_vals = [200., 300., 400., 500., 600., 700.]
#numin_vals = [5., ] #10., 20., 30., 40., 50., 60., 70.]
#numax_vals = [100.,] # 300., 400., 500., 600., 700.]

# Temperature/polarisation noise rms for all bands, as a fraction of T_cmb
fsigma_T = 1. #0.01
fsigma_P = 2. #0.01

# Define input models and their amplitudes/parameters
#dust_model = models.DustMBB(amp_I=150., amp_Q=10., amp_U=10., dust_beta=1.6, dust_T=20.)
dust_model = models.DustSimpleMBB(amp_I=150., amp_Q=10., amp_U=10., dust_beta=1.6, dust_T=20.)
sync_model = models.SyncPow(amp_I=30., amp_Q=10., amp_U=10., sync_beta=-3.2)
cmb_model = models.CMB(amp_I=50., amp_Q=0.6, amp_U=0.6)

#name_in = "MBBSync"
name_in = "SimpleMBBSync"
mods_in = [dust_model, sync_model, cmb_model]
amps_in = np.array([m.amps() for m in mods_in])
params_in = np.array([m.params() for m in mods_in])

# Define models to use for the fitting
#name_fit = "MBBSync"
name_fit = "SimpleMBBSync"
mods_fit = [dust_model, sync_model, cmb_model]



def bands_log(nu_min, nu_max, nbands):
    """
    Logarithmic set of bands.
    """
    freq_vec = np.arange(nbands)
    return nu_min * (nu_max/nu_min)**(freq_vec/(nbands-1.)) * 1e9 # in Hz


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
                                        components=mods_in, 
                                        noise_file=NOISE_FILE)
                                        
    gls_cmb, cmb_chisq, cmb_noise \
            = fitting.model_test(nu, D_vec, Ninv, mods_fit, 
                                 burn=200, steps=800,
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
