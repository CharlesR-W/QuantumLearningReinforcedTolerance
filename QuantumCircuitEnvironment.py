# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:02:19 2021

@author: OwenHuisman, CharlesRW, RenzeSuters
"""



from tensorforce import Environment
import numpy as np
from scipy import linalg
from qiskit import QuantumCircuit
from qiskit.providers.aer import QasmSimulator
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt



class QuantumCircuitEnvironment(Environment):
    def __init__(self,num_qubits:int, num_gates:int, num_samples:int, gateNoiseParams:tuple, targetBigState, ket_rho_tradeoff:float) -> None:
        '''
        num_qubits = int
        num_samples = int
        gateNoiseParams is a list of (mu,sigma) pairs for each gate
            For now we will assume the 1-qubit gates, when written as R_n (theta), have n and theta Gaussian noise
            For 2 single-qubit gates, would be ((mu_th,sig_th),(mu_n,sig_n))'''

        #Number of qubits to simulate
        self.num_qubits = num_qubits

        #number of times to simulate a given QC to find rho
        self.num_samples = num_samples

        # reward = ket_reward + tradeoff * rho_reward
        self.ket_rho_tradeoff = ket_rho_tradeoff
        
        #NB this is NOT the number of distinct 1-qubit gates, but is the total possible number of actions
        # i.e. in the case of Hadamard, would have H1, H2...Hn
        self.num_gates = num_gates 
        self.simulator = QasmSimulator()
        #self.circuit = QuantumCircuit(num_qubits)
        self.gateNoiseParams = gateNoiseParams
        self.action_list = []
        #self.targetBigRho = targetBigRho
        
        all_idx = range(self.num_qubits)
        self.cart_prod = [(x,y) for x in all_idx for y in all_idx]
        for x in self.cart_prod:
            if x[0]==x[1]:
                self.cart_prod.remove(x)
        # make sure the length is right - number of unique ordered pairs
        n_pairs = len(self.cart_prod)
        assert n_pairs == self.num_qubits*(self.num_qubits-1)

        self.targetBigState = targetBigState
        

    def states(self):
        return dict(type='float', shape=(2**(self.num_qubits*3+1) + 2**(self.num_qubits)))


    def actions(self):
        # The +1 is to account for the NUL action
        return dict(type='int', num_values=self.num_gates+1)
    


    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    # Optional additional steps to close environment
    def close(self):
        super().close()


    def reset(self):
        '''reset states to the 'neutral' starting position
        seemingly essential function for the RL machine'''
        #state = np.zeros(self.targetBigRho.shape).flatten()
        #state = np.zeros(2**(self.num_qubits*3+1))
        #state[0] = 1
        
        # idealCircuit = self.constructIdealCircuit()
        # print(idealCircuit)
        
        self.action_list = []
        state, _reward , _terminal = self.getSampledBigState()
        
        return state


    def execute(self, actions):
        '''update ideal circuit based on action'''

        self.action_list.append(actions)
        next_state, terminal, reward= self.getSampledBigState()    
        if actions == 0:
            terminal = True
        else:
            terminal = False  # Always False if no "natural" terminal state
        
        return next_state, terminal, reward
    

    def constructIdealCircuit(self):
        '''construct a noiseless, ideal version of a circuit
        based off of the gates in the current self.action_list, implement everything without noise'''
        
        #initialise the circuit to zero state
        idealCircuit = QuantumCircuit(self.num_qubits)
        idealCircuit.initialize(0,idealCircuit.qubits)
        
        noiseless_list = self.getNoisyAngles(noise=False)
        self.constructCircuitRealisation(idealCircuit, noiseless_list)
        
        return idealCircuit


    def constructCircuitRealisation(self, noisyCircuit:QuantumCircuit, current_noise:list) -> None:
        '''add gates from the given action list to the noisyCircuit
        action is an int between 0 and self.num_gates
        need to specify the gate to be applied and the qubit line on which to apply it
        Assume gate order is NUL, Rz1,Rz2... S1,S2,.... CNOT(1,1) CNOT(1,2)...'''
        
        '''noise is an array of tuples (a,b,c) of the noise parameters to be passed to each circuit'''
        
        n = self.num_qubits
        
        for lv in range(len(self.action_list)):
            action = self.action_list[lv]
            gate_noise = current_noise[lv]
            if action == 0:
                self.applyNUL()
            elif action < n+1:
                qbit_idx = action - 1
                self.applyR(noisyCircuit, qbit_idx, gate_noise)
            elif action < 2*n+1:
                qbit_idx = action - n -1
                self.applyS(noisyCircuit, qbit_idx, gate_noise)
            else:
                tmp_idx = action - 2*n - 1
                self.applyCNOT(noisyCircuit, tmp_idx, gate_noise)
        #print(noisyCircuit)
    

    def applyNUL(self):
        '''Function that symbolizes doing nothing, gate=identity.'''
        pass
    
    
    def applyR(self, circuit, qbit_idx, gate_noise):
        '''add a noisy rotation over Z by pi/8 gate to the end of the circuit on qubit index qbit_idx
        the noise is given by a normal distribution with standard deviation given by gateNoiseParams
        forms a discrete universal single qubit gate set with the S gate'''

        # U(theta,0,0) = R(theta)

        theta = np.pi/8 + gate_noise[0]
        phi = 0 + gate_noise[1]
        lam = 0 + gate_noise[2]

        circuit.u(theta, phi, lam, qbit_idx)


    def applyS(self, circuit, qbit_idx, gate_noise):
        '''add a noisy S gate to the end of the circuit on qubit index qbit_idx
        the noise is given by a normal distribution with standard deviation given by gateNoiseParams
        forms a discrete universal single qubit gate set with the R gate above'''

        #U(0,pi/2,0) = S
        theta = 0 + gate_noise[0]
        phi = np.pi/2 + gate_noise[1]
        lam = 0 + gate_noise[2]

        circuit.u(theta, phi, lam, qbit_idx)


    def applyCNOT(self, circuit, tmp_idx, gate_noise):
        '''add a noisy CNOT gate to the end of the circuit on qubits tmp_idx 
        noisy in the sense that we add it with probability 1-p_error and we don't add it with prob. p_error'''

        #if noise:
        #    p_error = self.gateNoiseParams[2]
        #else:
        #    p_error = 0
            
        #implement = np.random.choice(2, 1, p = [p_error, 1-p_error])[0]
        
        #if implement:
        '''OLD CODE implement noise if we really want to?'''
        circuit.cnot(self.cart_prod[tmp_idx])


    def getSampledBigState(self):
        '''Loop over #samples and #basis_states to calculate the density matrix and kets for noisy circuit instances.
        Based on the rho's for all these samples, calculate sample averaged rho.
        Calculate reward based on sample averaged rho and for every single ket.
        Each basis state in one sample has the same noise parameters.

        Newest iteration of previous function getSampledBigRho.'''
        n = self.num_qubits
        num_basis_states = 2**n # number of classical basis states

        ket_samples = np.zeros([self.num_samples, num_basis_states,num_basis_states], dtype=complex)
        ket_rewards = np.zeros([self.num_samples, num_basis_states], dtype=complex)
        
        rho_samples = np.zeros([self.num_samples, num_basis_states, num_basis_states, num_basis_states], dtype=complex)

        # prepare to draw a single sample
        for sample_id in range(self.num_samples):
            current_noise = self.getNoisyAngles() #CHECK
            
            # loop over all basis input states, storing kets
            for basis_id in range(num_basis_states):
                # convert binary_state_id (an int) into a string of binary
                binary = np.binary_repr(basis_id,width=n)
    
                # create a circuit with the specified noise, init'ed with this basis as input
                rho, ket = self.runCircuit(binary, current_noise)

                # calculate ket part of reward:
                ket_samples[sample_id, basis_id] = ket
                ket_rewards[sample_id, basis_id] = self.calculateKetReward(ket, basis_id)
                
                # store the rho for later averaging:
                rho_samples[sample_id, basis_id,:,:] = rho
        
        # loop over basis states once again to calculate rewards and phase differences etc.
        
        rho_averaged = np.mean(rho_samples,axis=0)
        
        rewards = np.zeros((num_basis_states))

        for basis_id in range(num_basis_states):
            ket_reward = np.mean(ket_rewards[:,basis_id])
            rho_reward = self.calculateRhoReward(rho_averaged, basis_id)
        
            # calculate overall reward:
            rewards[basis_id] = np.real(ket_reward + self.ket_rho_tradeoff * rho_reward) / 2.0

        # add/average over rewards per basis state
        reward = np.mean(rewards)
        
        # calculate the phase difference of each output relative to the |00000...> output; used for observation
        avg_delta_phi = self.getAvgPhaseDifferences(ket_samples) # calculate phase differences of kets

        #observation = one f*cking large array with rho averaged + average phase differences for all basis states
        observation = self.observeState(rho_averaged, avg_delta_phi) #CHECK does it work this way?

        terminal = False

        return observation, terminal, reward


    '''OLD CODE
    def observeRho(self, rho_sample_averaged, n, num_basis_states):
        Takes as input the sample-averaged DMs, and flattens them.  Broken off mainly for modularity
        n and num_basis_states are passed strictly for convenience

        obs_size = num_basis_states**3 #total number of CPLX datapts to store
        
        # sample-averaged DMs, but flat + split into RE and IM parts
        observation = np.zeros(2*obs_size)

        #flatten sample-averaged DMs and store RE and IM parts separately
        observation[:obs_size] = np.ndarray.flatten(np.real(rho_sample_averaged))
        observation[obs_size:] = np.ndarray.flatten(np.imag(rho_sample_averaged))

        return observation


    def observeKet(self):
        #
        obs_size = num_basis_states**2 #total number of CPLX datapts to store
        
        # sample-averaged DMs, but flat + split into RE and IM parts
        observation = np.zeros(2*obs_size)

        #flatten sample-averaged DMs and store RE and IM parts separately
        observation[:obs_size] = np.ndarray.flatten(np.real(rho_sample_averaged))
        observation[obs_size:] = np.ndarray.flatten(np.imag(rho_sample_averaged))

        return observation'''


    def observeState(self, rho_sample_averaged, avg_delta_phi):
        rho_size = 2**(self.num_qubits*3) #total number of CPLX datapts to store
        avg_dph_size = 2**(self.num_qubits)
        
        # sample-averaged DMs, but flat + split into RE and IM parts
        observation = np.zeros(rho_size*2 + avg_dph_size)

        #flatten sample-averaged DMs and store RE and IM parts separately + phase differences
        observation[:rho_size] = np.ndarray.flatten(np.real(rho_sample_averaged))
        observation[rho_size:2*rho_size] = np.ndarray.flatten(np.imag(rho_sample_averaged))
        observation[2*rho_size:] = avg_delta_phi

        return observation


    def runCircuit(self, binary, currentNoise):
        '''create one instance of a noisy ciruit given currentNoise parameters and then calculate its density matrix
        First, initialize the circuit in one particular basis state given by binary.
        Finally, get the density matrix of the (noisy) circuit (one particular instance, so always pure density matrix).'''

        noisyCircuit = QuantumCircuit(self.num_qubits)

          ####-Old Code-####
    #     #loop over each qubit register to initialise
    #     for lv in range(self.num_qubits):
    #         b = int(binary[lv])
    #         assert b == 0 or b == 1
    #         if b == 0:
    #             vec = [1,0]
    #         elif b==1:
    #             vec = [0,1]
          #################

        noisyCircuit.initialize(binary,noisyCircuit.qubits)

        # Now set up noisyCircuit according to currentNoise:
        self.constructCircuitRealisation(noisyCircuit, currentNoise)

        rho = qi.DensityMatrix.from_instruction(noisyCircuit)
        rho = np.array(rho)

        statevector = qi.Statevector.from_instruction(noisyCircuit)
        statevector = np.array(statevector)

        return rho, statevector


    def calculateRhoReward(self, rho_averaged, basis_id):
        '''calculate the density matrix part of the RL reward given a mixed density matrix rho'''

        #reward = np.trace(np.square(rho_averaged[basis_id])) # trace of matrix**2 is a measure of purity, which is what we want to use as reward
        reward = np.trace(np.matmul((rho_averaged[basis_id]),np.transpose(np.conj(rho_averaged[basis_id]))))
        return np.real(reward)


    def calculateKetReward(self, statevector, basis_id):
        '''calculate the ket part of the RL reward given a statevector and the target statevector in a particular basis_id'''
        
        inner_product = np.inner(np.conj(self.targetBigState[basis_id]), statevector)
        reward = (np.abs(inner_product))*(1 + np.cos(np.angle(inner_product)))/2
        return np.real(reward) - 0.95


    def getNoisyAngles(self, noise=True):
        '''calculate noise parameters for all the gates in the action list and output them in one array'''

        n = self.num_qubits
        noisy_angles = np.zeros((len(self.action_list),3))

        for lv in range(len(self.action_list)):
            action = self.action_list[lv]
           # current_noise = current_noise[lv]
            
            #CHECK if this should just 'pass' - I don't think it makes a difference, but its weird as-is
            if action == 0:
                self.applyNUL()

            elif action < n+1:
                if noise:
                    sig = self.gateNoiseParams[0]
                else:
                    sig = [0,0,0]
                mu_theta = np.pi/4
                sig_theta = sig[0]
                mu_phi = 0
                sig_phi = sig[1]
                mu_lam = 0
                sig_lam = sig[2]

                theta = np.random.normal(loc=mu_theta, scale = sig_theta) - mu_theta
                phi = np.random.normal(loc=mu_phi, scale = sig_phi) - mu_phi
                lam = np.random.normal(loc=mu_lam, scale = sig_lam) -mu_lam

                noisy_angles[lv,0],noisy_angles[lv,1] ,noisy_angles[lv,2]  = theta, phi, lam
            
            elif action < 2*n+1:
                if noise:
                    sig = self.gateNoiseParams[1]
                else:
                    sig = [0,0,0]
                mu_theta = 0
                sig_theta = sig[0]
                mu_phi = np.pi/2
                sig_phi = sig[1]
                mu_lam = 0
                sig_lam = sig[2]


                theta = np.random.normal(loc=mu_theta, scale = sig_theta) - mu_theta
                phi = np.random.normal(loc=mu_phi, scale = sig_phi) - mu_phi
                lam = np.random.normal(loc=mu_lam, scale = sig_lam) - mu_lam

                noisy_angles[lv,0],noisy_angles[lv,1] ,noisy_angles[lv,2]  = theta, phi, lam
            
            else:
                if noise:
                    sig = self.gateNoiseParams[2]
                else:
                    sig = 0

                noisy_angles[lv,0], noisy_angles[lv,1], noisy_angles[lv,2]  = sig, sig, sig

        return noisy_angles


    def getAvgPhaseDifferences(self, ket_samples):
    # ket samples is num_samples * num_basis_states * num_basis_states
        num_basis_states = 2**self.num_qubits    
        delta_phi = np.zeros([self.num_samples, num_basis_states])
        
        for sample_id in range(self.num_samples):
            # the phase |00000...0> --> e^i phi_ref (a0bs(alpha) , beta, ...)
            phi_ref = np.angle(ket_samples[sample_id,0,0])
            
            for input_basis_id in range(num_basis_states):
                phi = np.angle(ket_samples[sample_id,input_basis_id,0])
                delta_phi[sample_id, input_basis_id] = phi - phi_ref
        
        avg_delta_phi = np.mean(delta_phi,axis=0)
        return avg_delta_phi

