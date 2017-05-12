import numpy as np
import emcee
from scipy import stats
import rebound
from datetime import datetime
import sys
import traceback
import copy
import state

'''
Parent MCMC class
'''
class Mcmc(object):
    def __init__(self, initial_state, obs):
        self.state = initial_state.deepcopy()
        self.obs = obs

    def step(self):
        return True 
    
    def step_force(self):
        tries = 1
        while self.step()==False:
            tries += 1
            pass
        return tries

#create a static lnprob function to pass to the emcee package
def lnprob(x, e):
    e.state.set_params(x)
    try:
        logp = e.state.get_logp(e.obs)
    except:
        print "Collision! {t}".format(t=datetime.utcnow())
        e.state.collisionGhostParams.append(e.state.deepcopy())
        #e.state.collisionGhostParams.append(e.state.get_params())
        #print e.state.collisionGhostParams
        return -np.inf
    return logp

'''
emcee MCMC coupled with rebound.
'''
class Ensemble(Mcmc):
    def __init__(self, initial_state, obs, scales, nwalkers=10):
        super(Ensemble,self).__init__(initial_state, obs)
        self.set_scales(scales)
        self.nwalkers = nwalkers
        self.states = [self.state.get_params() for i in range(nwalkers)]
        self.previous_states = [self.state.get_params() for i in range(nwalkers)]
        self.lnprob = None
        self.totalErrorCount = 0
        for i,s in enumerate(self.states):
            shift = 0.1e-2*self.scales*np.random.normal(size=self.state.Nvars)
            self.states[i] += shift
            self.previous_states[i] += shift
        self.sampler = emcee.EnsembleSampler(nwalkers,self.state.Nvars, lnprob, args=[self])

    '''
    Constitutes 1 emcee step.
    '''
    def step(self):
        self.previous_states = copy.deepcopy(self.states)
        self.state.collisionGhostParams = []
        self.states, self.lnprob, rstate = self.sampler.run_mcmc(self.states,1,lnprob0=self.lnprob)
        for i in range(len(self.states)):
            for j in range(len(self.states[0])):
                if(self.previous_states[i][j] != self.states[i][j]):
                    return True
        else:
            return False

    '''
    Sets the scales for the initial random distribution of walkers. Mileage may vary.
    '''
    def set_scales(self, scales):
        self.scales = np.ones(self.state.Nvars)
        keys = self.state.get_rawkeys()
        for i,k in enumerate(keys):
            if k in scales:
                self.scales[i] = scales[k]

