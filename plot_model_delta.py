#!/usr/bin/python
"""
Plot how model spectrum changes as a function of frequency w.r.t changing 
parameters.
"""
import numpy as np
#matplotlib.use('Agg') # Need this to avoid using X server
import pylab as P
import models

dbeta = 0.05
dT = 0.5

# Define input models and their amplitudes/parameters
dust = models.DustMBB
dust0 = dust(amp_I=10., amp_Q=1., amp_U=1.2, dust_beta=1.6, dust_T=20.)
beta_p = dust(amp_I=10., amp_Q=1., amp_U=1.2, dust_beta=1.6+dbeta, dust_T=20.)
beta_m = dust(amp_I=10., amp_Q=1., amp_U=1.2, dust_beta=1.6-dbeta, dust_T=20.)
T_p = dust(amp_I=10., amp_Q=1., amp_U=1.2, dust_beta=1.6, dust_T=20.+dT)
T_m = dust(amp_I=10., amp_Q=1., amp_U=1.2, dust_beta=1.6, dust_T=20.-dT)

sync0 = models.SyncPow(amp_I=10., amp_Q=2., amp_U=1.5, sync_beta=-3.)
sbeta_p = models.SyncPow(amp_I=10., amp_Q=2., amp_U=1.5, sync_beta=-3.+dbeta)
sbeta_m = models.SyncPow(amp_I=10., amp_Q=2., amp_U=1.5, sync_beta=-3.-dbeta)

nu = np.logspace(np.log10(5.), 3., 500) * 1e9
dfdbeta = (beta_p.scaling(nu) - beta_m.scaling(nu)) / (2. * dbeta)
dfdT = (T_p.scaling(nu) - T_m.scaling(nu)) / (2. * dT)
dfdsbeta = (sbeta_p.scaling(nu) - sbeta_m.scaling(nu)) / (2. * dbeta)



P.subplot(111)
P.plot(nu/1e9, np.abs(dfdbeta[0]), 'b-', lw=1.8)
P.plot(nu/1e9, np.abs(dfdT[0]), 'c-', lw=1.8)
P.axvline(dust0.nu_ref/1e9, color='b', ls='dashed')

P.plot(nu/1e9, np.abs(dfdsbeta[0]), 'r-', lw=1.8)
P.axvline(sync0.nu_ref/1e9, color='r', ls='dashed')

"""
P.plot(nu, dust0.scaling(nu)[0], 'k-')
P.axvline(353., color='k', ls='dashed')

P.plot(nu, sync0.scaling(nu)[0], 'b-')
P.axvline(30., color='b', ls='dashed')

P.axhline(1., color='k', ls='dashed')
"""
P.yscale('log')
P.xscale('log')
P.show()

#sync_model = models.SyncPow(amp_I=10., amp_Q=2., amp_U=1.5, sync_beta=-3.)
#cmb_model = models.CMB(amp_I=100., amp_Q=10., amp_U=20.)
