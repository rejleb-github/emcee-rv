import rebound
import matplotlib.pyplot as plt
import observations
import state
import mcmc
import numpy as np

def run_emcee(Niter, true_state, obs, Nwalkers, scal):
    ens = mcmc.Ensemble(true_state,obs,scales=scal,nwalkers=Nwalkers)
    listchain = np.zeros((Nwalkers,ens.state.Nvars,0))
    listchainlogp = np.zeros((Nwalkers,0))
    tries=0
    for i in range(int(Niter/Nwalkers)):
        if(ens.step()):
            for q in range(len(ens.states)):
                if(np.any(ens.previous_states[q] != ens.states[q])):
                    tries += 1
        listchainlogp = np.append(listchainlogp, np.reshape(ens.lnprob, (Nwalkers, 1)), axis=1)
        listchain = np.append(listchain, np.reshape(ens.states, (Nwalkers,ens.state.Nvars,1)),axis=2)
    print("Acceptance rate: %.3f%%"%(tries/(float(Niter))*100))
    chain = np.zeros((ens.state.Nvars,0))
    chainlogp = np.zeros(0)
    for i in range(Nwalkers):
        chain = np.append(chain, listchain[i], axis=1)
        chainlogp = np.append(chainlogp, listchainlogp[i])
    return ens, np.transpose(chain), chainlogp

def create_obs(state, npoint, err, errVar, t):
    obs = observations.FakeObservation(state, Npoints=npoint, error=err, errorVar=errVar, tmax=(t))
    return obs
