import torch 
import numpy as np
from scipy.stats.distributions import norm,truncnorm
from scipy.stats._discrete_distns import binom
import time
import CompDoobTransform as cdt

class Simulator:
    # optimization configuration (standard training)
    I = 2000
    optim_config = {'minibatch': 100, 
                    'num_obs_per_batch': 10, 
                    'num_iterations': I,
                    'learning_rate' : 0.01, 
                    'initial_required' : True}
    def __init__(self,
                drift,
                sigma,
                M,
                T,
                state_dim,
                var_obs,
                obs_log_density,
                X0,
                obs_cond_law):
        self.dim = state_dim
        self.drift = drift
        self.sigma=sigma
        self.M = M
        self.T = T
        # initialization des infos de l'espace d'etat
        self.state = {}
        self.state["dim"] = state_dim
        self.state["drift"] = drift
        self.state["sigma"] = sigma
        self.state["terminal_time"] = T

        # initialisation des infos de l'espace des observations
        self.obs = {}
        self.obs["dim"] = state_dim

        self.var_obs = var_obs
        self.obs["log_density"] = obs_log_density
        self.obs["obs_cond_law"] = obs_cond_law # loi conditionnelle de y par rapport à x

        ## initial state
        self.X0 = X0
        self.standardisation = {}

        ## saving model
        self.model = None

    def simulate_state_obs(self, nb_repeats, device):
        X = self.X0.clone()
        J = nb_repeats
        max_index = J*self.M+1
        store_states = torch.zeros(J*self.M+1, self.state["dim"], device = device)
        store_states[0,:] = X    
        store_obs = torch.zeros(J*self.M, self.state["dim"], device = device)
        stepsize = torch.tensor(self.state["terminal_time"] / self.M, device = device)
        for j in range(J):
            for m in range(self.M):
                euler = X + stepsize * self.state["drift"](X)
                W = torch.sqrt(stepsize) * torch.randn(X.shape, device = device)
                X = euler + self.state["sigma"](X) * W
                Y = X + torch.sqrt(self.var_obs) * self.obs["obs_cond_law"](X, 1, self.state["dim"], device = device)
                index = j*self.M + m + 1
                store_states[index,:] = X
                store_obs[index-1,:] = Y

        self.standardisation = {'x_mean': torch.mean(store_states), 
                   'x_std': torch.std(store_states), 
                   'y_mean': torch.mean(store_obs), 
                   'y_std': torch.std(store_obs)}

        # simulate initial states
        initial = lambda N: store_states[torch.randint(0, max_index, size = (N,)), :] # function to subsample states
        self.state['initial'] = initial

        # simulate observations
        observation = lambda N: initial(N) + torch.sqrt(self.var_obs) * self.obs["obs_cond_law"](X,N, self.state["dim"], device = device)
        self.obs['observation'] = observation
        return(store_states, store_obs)
    
    def train_static(self):
        # V0 and Z neural network configuration
        V0_net_config = {'layers': [16], 'standardization': self.standardisation}
        Z_net_config = {'layers': [self.state["dim"]+16], 'standardization': self.standardisation}
        net_config = {'V0': V0_net_config, 'Z': Z_net_config}
        # create model instance
        model_static = cdt.core.model(self.state, self.obs, self.M, net_config, device = 'cpu')

        # static training
        time_start = time.time() 
        model_static.train_standard(self.optim_config)
        time_end = time.time()
        time_elapsed = time_end - time_start
        print("Training time (secs): " + str(time_elapsed))
        self.model = model_static
    
    def train_iterative(self):
        # create model instance
        model = cdt.core.model(self.state, self.obs, self.M, self.net_config, device = 'cpu')

        # iterative training
        time_start = time.time() 
        model.train_iterative(self.optim_config)
        time_end = time.time()
        time_elapsed = time_end - time_start
        print("Training time (secs): " + str(time_elapsed))
        self.model = model

    
    def compare_apf_bpf(self,multiplier,num_obs,num_particles,nb_repeats):
        len_num_obs = len(num_obs)
        
        APFF = {'ess' : torch.zeros(len_num_obs, nb_repeats), 'log_estimate' : torch.zeros(len_num_obs, nb_repeats)}
        BPF = {'ess' : torch.zeros(len_num_obs, nb_repeats), 'log_estimate' : torch.zeros(len_num_obs, nb_repeats)}
        for i in range(len_num_obs):
            # number of observations
            K = num_obs[i]

            # number of particles
            N = num_particles[i]

            # simulate latent process and observations
            X0 = torch.ones(1,self.state["dim"])
            X = torch.zeros(K+1, self.state["dim"])
            X[0,:] = X0.clone()
            Y = torch.zeros(K, self.state["dim"])
            for k in range(K):
                X[k+1,:] = self.model.simulate_diffusion(X[k,:].reshape((1,self.state["dim"])))
                Y[k,:] = X[k+1,:] + multiplier * torch.sqrt(self.var_obs) * self.obs["obs_cond_law"](X,1,self.state["dim"])

            for r in range(nb_repeats):
                # run particle filters
                BPF_output = self.model.run_BPF(X0.repeat((N,1)), Y, N)
                APFF_output = self.model.run_APF(X0.repeat((N,1)), Y, N)

                # save average ESS%
                BPF_ESS = torch.mean(BPF_output['ess'] * 100 / N)
                APFF_ESS = torch.mean(APFF_output['ess'] * 100 / N)
                BPF['ess'][i,r] = BPF_ESS
                APFF['ess'][i,r] = APFF_ESS

                # save log-likelihood estimates
                BPF_log_estimate = BPF_output['log_norm_const'][-1]
                APFF_log_estimate = APFF_output['log_norm_const'][-1]
                BPF['log_estimate'][i,r] = BPF_log_estimate
                APFF['log_estimate'][i,r] = APFF_log_estimate

                # print output
                print('No. of observations: ' + str(K) + ' Repeat: ' + str(r)) 
                print('BPF ESS%: ' + str(BPF_ESS))
                print('APFF ESS%: ' + str(APFF_ESS)) 
                print('BPF log-estimate: ' + str(BPF_log_estimate))
                print('APFF log-estimate: ' + str(APFF_log_estimate))

        # save results
        results = {'BPF' : BPF, 'APFF' : APFF}
        return results
    
        




