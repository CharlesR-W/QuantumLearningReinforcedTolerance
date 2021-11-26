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
        self.circuit = QuantumCircuit(num_qubits)
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
        return dict(type='float', shape=(2**(self.num_qubits*3),))


    def actions(self):
        return dict(type='int', num_values=self.num_gates)


    # Optional: should only be defined if environment has a natural fixed
    # maximum episode length; otherwise specify maximum number of training
    # timesteps via Environment.create(..., max_episode_timesteps=???)
    def max_episode_timesteps(self):
        return super().max_episode_timesteps()


    # Optional additional steps to close environment
    def close(self):
        super().close()


    def reset(self): #TODO
        state = np.zeros(self.targetBigRho.shape).flatten()
        return state


    def execute(self, actions):
        #update ideal circuit based on action
        #assert len(actions) == 1
        self.action_list.append(actions)
        next_state, reward ,terminal = self.getSampledBigRho()

        terminal = False  # Always False if no "natural" terminal state
        
        return next_state, terminal, reward
    

    # From the given ideal circuit, return a qiskit object which represents one 'noisy' version of the new gate
    def constructCircuitRealisation(self):
        #collect all the gates the agent thinks it's applying
        #ideal_instruction_set = self.IdealCircuit.InstructionSet()
       
        #action is an int between 0 and self.num_gates
        #need to specify the gate to be applied and the qubit line on which to apply it
        #Assume gate order is Rz1,Rz2... S1,S2,.... CNOT(1,1) CNOT(1,2)...
        
        n = self.num_qubits
        noisyCircuit = QuantumCircuit(n)
        
        
        
        for action in self.action_list:
            if action < n:
                qbit_idx = action
                self.applyR(noisyCircuit, qbit_idx)
            elif action < 2*n:
                qbit_idx = action - n
                self.applyS(noisyCircuit, qbit_idx)
            else:
                tmp_idx = action - 2*n
                assert tmp_idx < n_pairs #check we're not out of bounds
                assert self.cart_prod[tmp_idx] == (cart_prod[tmp_idx][0], cart_prod[tmp_idx][1]) # CHECK this automatically unpacks?
                self.applyCNOT(noisyCircuit, tmp_idx)

        return noisyCircuit
    
    
    def applyR(self, circuit, qbit_idx):
        # U(theta,0,0) = R(theta)

        sig = self.gateNoiseParams[0]
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


    def applyS(circuit, qbit_idx):
        #U(0,pi/2,0) = S

        sig = self.gateNoiseParams[1]
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


    def applyCNOT(circuit, tmp_idx):
        
        p_error = self.gateNoiseParams[2]
        implement = np.random.choice(2, 1, p = [p_error, 1-p_error])[0]
        
        if implement:
            circuit.cnot(self.cart_prod[tmp_idx])


    def getSampledBigRho(self):

        #targetrho should be 2^n*n*n rho' = U rho U* for input rho
        n = self.num_qubits
        bits = 2**n
        #circuit = self.constructNoisy
        rhosamples = np.zeros((bits, self.num_samples, n, n))
        rhomixed = np.zeros((bits,n, n))
        observation = np.zeros((2*bits*n**2))
        rewards = np.zeros((bits))
        for basis_state in range(bits):
            binary = np.binary_repr(basis_state, width=bits)
            
            for sample in range(self.num_samples):
                rhosamples[basis_state, sample] = self.runCircuit(binary)                
            
            rhomixed[basis_state] = np.average(rhosamples[basis_state], axis=0)
            rewards[basis_state] = self.calculateReward(rhomixed, basis_state)
        
        terminal = False #CHECK
        observation[:bits*n**2-1] = np.flatten(np.real(rhomixed))
        observation[bits*n**2:] = np.flatten(np.imag(rhomixed))
        reward = np.average(rewards, axis=0)
        return observation, reward, terminal


    def runCircuit(self, binary):

        circuit = self.constructCircuitRealisation()
        circuit.initialize(binary, circuit.qubits)


        rho = qi.DensityMatrix.from_instruction(circuit)
        
        return rho


    def calculateReward(self, rhomixed, basis_state):
        '''calculate the RL reward given a mixed density matrix rhomixed and the pure target density matrix targetrho'''
        
        #error = np.linalg.norm((rhomixed[basis_state]-self.targetrho[basis_state]), ord='fro')
        #reward =  (error<0.2*bits**2) #Maybe there is a nicer way to assort rewards?

        fidelity = self.calculateFidelity(self.targetBigRho[basis_state], rhomixed[basis_state])
        reward = fidelity # fidelity always between 0 and 1, is 1 if perfect match

        return reward
    

    def calculateFidelity(target_rho, realized_rho):
        '''description'''

        #TODO might be simplified because target rho is always pure
        sqrt_target = linalg.sqrtm(target_rho)
        matrix1 = np.matmul(realized_rho, sqrt_target)
        matrix2 = np.matmul(sqrt_target, matrix1)
        sqrt_matrix = linalg.sqrtm(matrix2)
        fidelity = (np.trace(sqrt_matrix))**2

        return fidelity
