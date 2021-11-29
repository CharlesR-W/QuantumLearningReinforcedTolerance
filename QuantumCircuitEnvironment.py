# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:02:19 2021

@author: OwenHuisman, CharlesRW, RenzeSuters
"""
from tensorforce import Environment
import numpy as np
from scipy import linalg
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
import qiskit.quantum_info as qi

from qiskit.providers.aer.noise import NoiseModel
from qiskit.providers.aer.noise.errors import pauli_error, depolarizing_error
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit, Aer, transpile, assemble
from qiskit.visualization import plot_histogram
aer_sim = Aer.get_backend('aer_simulator')


class QuantumCircuitEnvironment(Environment):
    def __init__(self,num_qubits, num_gates, num_samples, gateNoiseParams, targetBigRho):
        #num_qubits = int
        #num_samples = int
        #gateNoiseParams is a list of (mu,sigma) pairs for each gate
        #  For now we will assume the 1-qubit gates, when written as R_n (theta), have n and theta Gaussian noise
        #  For 2 single-qubit gates, would be ((mu_th,sig_th),(mu_n,sig_n))

        #Number of qubits to simulate
        self.num_qubits = num_qubits

        #number of times to simulate a given QC to find rho
        self.num_samples = num_samples

        #NB this is NOT the number of distinct 1-qubit gates, but is the total possible number of actions
        # i.e. in the case of Hadamard, would have H1, H2...Hn
        self.num_gates = num_gates 
        self.simulator = QasmSimulator()
        #self.circuit = QuantumCircuit(num_qubits)
        self.gateNoiseParams = gateNoiseParams
        self.action_list = []
        self.targetBigRho = targetBigRho
        
        all_idx = range(self.num_qubits)
        self.cart_prod = [(x,y) for x in all_idx for y in all_idx]
        for x in self.cart_prod:
            if x[0]==x[1]:
                self.cart_prod.remove(x)
        # make sure the length is right - number of unique ordered pairs
        n_pairs = len(self.cart_prod)
        assert n_pairs == self.num_qubits*(self.num_qubits-1)
        

    def states(self):
        return dict(type='float', shape=(2**(self.num_qubits*3+1),))


    def actions(self):
        return dict(type='int', num_values=self.num_gates+1)


    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    # Optional additional steps to close environment
    def close(self):
        super().close()


    def reset(self): #TODO I have no idea what reset does in tensorforce :P
        #state = np.zeros(self.targetBigRho.shape).flatten()
        state = np.zeros(2**(self.num_qubits*3+1))
        state[0] = 1
        
        idealCircuit = self.constructIdealCircuit()
        print(idealCircuit)
        
        self.action_list = []
        
        return state


    def execute(self, actions):
        #update ideal circuit based on action
        #assert len(actions) == 1
        self.action_list.append(actions)
        next_state, reward ,terminal = self.getSampledBigRho()

        terminal = False  # Always False if no "natural" terminal state
        
        return next_state, terminal, reward
    

    def constructIdealCircuit(self):
        '''TODO'''
        
        idealCircuit = QuantumCircuit(self.num_qubits)
        idealCircuit = self.constructCircuitRealisation(idealCircuit, noise=False)
        
        return idealCircuit

    def constructCircuitRealisation(self, noisyCircuit, noise=True):
        '''add gates from the given action list to the noisyCircuit
        action is an int between 0 and self.num_gates
        need to specify the gate to be applied and the qubit line on which to apply it
        Assume gate order is NUL, Rz1,Rz2... S1,S2,.... CNOT(1,1) CNOT(1,2)...'''
        
        n = self.num_qubits        
        
        for action in self.action_list:
            if action == 0:
                self.applyNUL()
            elif action < n+1:
                qbit_idx = action -1
                self.applyR(noisyCircuit, qbit_idx, noise)
            elif action < 2*n+1:
                qbit_idx = action - n -1
                self.applyS(noisyCircuit, qbit_idx, noise)
            else:
                tmp_idx = action - 2*n - 1
                assert tmp_idx < n_pairs #check we're not out of bounds
                assert self.cart_prod[tmp_idx] == (cart_prod[tmp_idx][0], cart_prod[tmp_idx][1]) # CHECK this automatically unpacks?
                self.applyCNOT(noisyCircuit, tmp_idx, noise)

        return noisyCircuit
    
    
    def applyNUL(self):
        pass
    
    
    def applyR(self, circuit, qbit_idx, noise):
        '''add a noisy rotation over Z by pi/8 gate to the end of the circuit on qubit index qbit_idx
        the noise is given by a normal distribution with standard deviation given by gateNoiseParams
        forms a discrete universal single qubit gate set with the S gate'''

        # U(theta,0,0) = R(theta)
        if noise:
            sig = self.gateNoiseParams[0]
        else:
            sig = [0,0,0]
        #sig = 0.01

        mu_theta = np.pi/4
        sig_theta = sig[0]
        mu_phi = 0
        sig_phi = sig[1]
        mu_lam = 0
        sig_lam = sig[2]

        theta = np.random.normal(loc=mu_theta, scale = sig_theta)
        phi = np.random.normal(loc=mu_phi, scale = sig_phi)
        lam = np.random.normal(loc=mu_lam, scale = sig_lam)

        circuit.u(theta,phi,lam, qbit_idx)


    def applyS(self, circuit, qbit_idx, noise):
        '''add a noisy S gate to the end of the circuit on qubit index qbit_idx
        the noise is given by a normal distribution with standard deviation given by gateNoiseParams
        forms a discrete universal single qubit gate set with the R gate above'''

        #U(0,pi/2,0) = S
        if noise:
            sig = self.gateNoiseParams[1]
        else:
            sig = [0,0,0]
        #sig = 0.01

        mu_theta = 0
        sig_theta = sig[0]
        mu_phi = np.pi/2
        sig_phi = sig[1]
        mu_lam = 0
        sig_lam = sig[2]
        

        theta = np.random.normal(loc=mu_theta, scale = sig_theta)
        phi = np.random.normal(loc=mu_phi, scale = sig_phi)
        lam = np.random.normal(loc=mu_lam, scale = sig_lam)

        circuit.u(theta,phi,lam, qbit_idx)


    def applyCNOT(circuit, tmp_idx, noise):
        '''add a noisy CNOT gate to the end of the circuit on qubits tmp_idx 
        noisy in the sense that we add it with probability 1-p_error and we don't add it with prob. p_error'''

        if noise:
            p_error = self.gateNoiseParams[2]
        else:
            p_error = 0
            
        implement = np.random.choice(2, 1, p = [p_error, 1-p_error])[0]
        
        if implement:
            circuit.cnot(self.cart_prod[tmp_idx])


    def getSampledBigRho(self):
        '''The main observation function: when run, 
        gets the density matrices for each of several
        sample realisations.  It then compares these to
        the ideal, and determines the reward.
        
        Reward = Fidelity of sample-averaged DMs compared to ideal
        Observed = average of sampled DMs
        Terminal = false (no natural end-state, ceiling on 
                          timesteps is implemented elsewhere)'''
        #targetrho should be 2^n*n*n rho' = U rho U* for input rho
        n = self.num_qubits
        num_basis_states = 2**n # number of classical basis states

        # to store all sample DMs: is 2**n , s , n, n
        rho_samples = np.zeros((num_basis_states, self.num_samples, num_basis_states, num_basis_states))
        
        # sample-averaged DM
        rho_sample_averaged = np.zeros((num_basis_states,num_basis_states, num_basis_states))
        
        # store reward for each basis state
        rewards = np.zeros((num_basis_states))

        # Loop over all basis states
        for basis_state_id in range(num_basis_states):
            #convert binary_state_id (an int) into a string of binary
            binary = np.binary_repr(basis_state_id, width=n)
            
            #sampling loop
            for sample_id in range(self.num_samples):
                #store each sample rho
                rho_samples[basis_state_id, sample_id,:,:] = self.runCircuit(binary)                
            
            # average the results of all samples
            rho_sample_averaged[basis_state_id] = np.average(rho_samples[basis_state_id], axis=0)

            #store the reward for this basis state
            rewards[basis_state_id] = self.calculateReward(rho_sample_averaged, basis_state_id)
        
        terminal = False #CHECK

        observation = self.observeRho(rho_sample_averaged,n,num_basis_states)

        reward = np.average(rewards, axis=0)
        #print("Reward, " + str(reward))
        return observation, reward, terminal

    def observeRho(self, rho_sample_averaged,n,num_basis_states):
        '''Takes as input the sample-averaged DMs, and flattens them.  Broken off mainly for modularity
        n and num_basis_states are passed strictly for convenience'''

        obs_size = num_basis_states**3 #total number of CPLX datapts to store
        
        # sample-averaged DMs, but flat + split into RE and IM parts
        observation = np.zeros(2*obs_size)

        #flatten sample-averaged DMs and store RE and IM parts separately
        observation[:obs_size] = np.ndarray.flatten(np.real(rho_sample_averaged))
        observation[obs_size:] = np.ndarray.flatten(np.imag(rho_sample_averaged))

        return observation

    def runCircuit(self, binary):
        '''create one instance of a noisy ciruit and then calculate its density matrix
        First, initialize the circuit in one particular basis state given by binary.
        Then, call consTtructCircuitRealisation to add (noisy) gates to the circuit.
        Finally, get the density matrix of the (noisy) circuit (one particular instance, so always pure density matrix).'''

        noisyCircuit = QuantumCircuit(self.num_qubits)
        #noisyCircuit.initialize(binary, noisyCircuit.qubits) #TODO fix bug of qbit init
        
        #loop over each qubit register to initialise
        for lv in range(self.num_qubits):
            b = int(binary[lv])
            assert b == 0 or b == 1
            if b == 0:
                vec = [1,0]
            elif b==1:
                vec = [0,1]

            noisyCircuit.initialize(vec,lv)

        noisyCircuit = self.constructCircuitRealisation(noisyCircuit)
        rho = qi.DensityMatrix.from_instruction(noisyCircuit)
        
        rho = np.array(rho)
 
        return rho


    def calculateReward(self, rhomixed, basis_state):
        '''calculate the RL reward given a mixed density matrix rhomixed and the pure target density matrix targetrho'''
        
        #error = np.linalg.norm((rhomixed[basis_state]-self.targetrho[basis_state]), ord='fro')
        #reward =  (error<0.2*bits**2) #Maybe there is a nicer way to assort rewards?

        fidelity = self.calculateFidelity(self.targetBigRho[basis_state], rhomixed[basis_state])
        reward = fidelity # fidelity always between 0 and 1, is 1 if perfect match

        return reward-0.5
    

    def calculateFidelity(self,target_rho, realized_rho):
        '''calculate the fidelity of the realized_rho density matrix and the target_rho
        Fidelity is always between 0 and 1. The fidelity is 1 iff the density matrices are exactly the same.'''

        # might be simplified because target rho is always pure, but nice for generality
        sqrt_target = linalg.sqrtm(target_rho)
        matrix1 = np.matmul(realized_rho, sqrt_target)
        matrix2 = np.matmul(sqrt_target, matrix1)
        sqrt_matrix = linalg.sqrtm(matrix2)
        fidelity = (np.trace(sqrt_matrix))**2

        return fidelity
