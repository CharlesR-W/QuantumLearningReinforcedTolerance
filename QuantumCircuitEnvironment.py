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
    def __init__(self,num_qubits,num_gates, num_samples, gateNoiseParams,):
        #num_qubits = int
        #num_samples = int
        #gateNoiseParams is a list of (mu,sigma) pairs for each gate
        #  For now we will assume the 1-qubit gates, when written as R_n (theta), have n and theta Gaussian noise
        #  For 2 single-qubit gates, would be ((mu_th,sig_th),(mu_n,sig_n))

        super().__init__():
            #Number of qubits to simulate
            self.num_qubits = num_qubits

            #number of times to simulate a given QC to find rho
            self.num_samples = num_samples

            #NB this is NOT the number of distinct 1-qubit gates, but is the total possible number of actions
            # i.e. in the case of Hadamard, would have H1, H2...Hn
            self.num_gates = num_gates 
            self.simulator = QasmSimulator()
            self.Circuit = QuantumCircuit(num_qubits, 0)
            self.gateNoiseParams = gateNoiseParams
            self.action_list = []
            self.targetBigRho = self.runCircuit()
            
            all_idx = range(self.num_qubits)
            self.cart_prod = [(x,y) for x in all_idx for y in all_idx]
            for x in self.cartprod:
                if x[0]==x[1]:
                    self.cart_prod.remove[x]
            # make sure the length is right - number of unique ordered pairs
            n_pairs = self.cart_prod.length()
            assert n_pairs == self.num_qubits*(self.num_qubits-1)
        
    def states(self):
        return dict(type='float', shape=(2**(self.num_qubits*3+1),))

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


    # def reset(self):
    #     state = np.random.random(size=(8,))
    #     return state


    def execute(self, actions):
        #update ideal circuit based on action
        assert len(actions) == 1
        self.action_list.append(actions)
        next_state, reward ,terminal = self.getSampledBigRho(self)

        terminal = False  # Always False if no "natural" terminal state
        
        return next_state, terminal, reward
    

    # From the given ideal circuit, return a qiskit object which represents one 'noisy' version of the new gate
    def constructCircuitRealisation(self):
        #collect all the gates the agent thinks it's applying
        #ideal_instruction_set = self.IdealCircuit.InstructionSet()
       
        #action is an int between 0 and self.num_gates
        #need to specify the gate to be applied and the qubit line on which to apply it
        #Assume gate order is Rz1,Rz2... S1,S2,.... CNOT(1,1) CNOT(1,2)...
        
        noisyCircuit = QuantumCircuit(num_qubits, 0)
        for action in self.action_list:

            if action < n
                qbit_idx = action
                applyR(noisyCircuit, qbit_idx)
            elif action < 2*n
                qbit_idx = action - n
                applyS(noisyCircuit, qbit_idx)
            else
                tmp_idx = action - 2*n
                assert tmp_idx < n_pairs #check we're not out of bounds
                assert self.cart_prod[tmp_idx] == (cart_prod[tmp_idx][0], cart_prod[tmp_idx][1]) # CHECK this automatically unpacks?
                noisyCircuit.cnot(self.cart_prod[tmp_idx]) 
            
        return noisyCircuit
    
    
    def applyR(circuit, qbit_idx):
        sig = 0.01
        # U(theta,0,0) = R(theta)
        mu_theta = np.pi/4
        sig_theta = sig
        mu_phi = 0
        sig_phi = sig
        mu_lam = 0
        sig_lam = sig

        theta = np.random.normal(loc=mu_theta, scale = sig_theta)
        phi = np.random.normal(loc=mu_phi, scale = sig_phi)
        lam = np.random.normal(loc=mu_lam, scale = sig_lam)

        circuit.u(theta,phi,lam, qbit_idx)

    def applyS(circuit, qbit_idx):
        #U(0,pi/2,0) = S
        sig = 0.01
        mu_theta = 0
        sig_theta = sig
        mu_phi = np.pi/2
        sig_phi = sig
        mu_lam = 0
        sig_lam = sig
        

        theta = np.random.normal(loc=mu_theta, scale = sig_theta)
        phi = np.random.normal(loc=mu_phi, scale = sig_phi)
        lam = np.random.normal(loc=mu_lam, scale = sig_lam)

        circuit.u(theta,phi,lam, qbit_idx)


    def getSampledBigRho(self):

        #targetrho should be 2^n*n*n rho' = U rho U* for input rho
        n = self.num_qubits
        bits = 2**(n)
        #circuit = self.IdealCircuit
        rhosamples = np.zeros((bits, self.num_samples, n, n))
        rhomixed = np.zeros((bits,n, n))
        observation = np.zeros((2*bits*n**2))
        #rewards = np.zeros((bits))
        for basis_state in range(bits):
            binary = np.binary_repr(basis_state, width=bits)
            
            for sample in range(self.num_samples):
                rhosamples[basis_state, sample] = runCircuit(circuit, binary)                
            
            rhomixed[basis_state] = np.average(rhosamples[basis_state], axis=0)
            rewards[basis_state] = self.calculateReward(rhomixed, basis_state)
        
        #TODO terminal
        terminal = False #CHECK
        observation[:bits*n**2-1] = np.flatten(np.real(rhomixed))
        observation[bits*n**2:] = np.flatten(np.imag(rhomixed))
        return observation, reward, terminal


    def runCircuit(self, binary):

        circuit = self.constructCircuitRealisation(self)
        circuit.initialize(binary, circuit.qubits)

        #job = backend.run(circuit)
        #result = job.result()
        #rho = result.get_unitary(circuit, decimals =3)

        rho = qi.DensityMatrix.from_instruction(circuit)
        
        return rho


    def calculateReward(self, rhomixed, basis_state):
        '''calculate the RL reward given a mixed density matrix rhomixed and the pure target density matrix targetrho'''
        
        #error = np.linalg.norm((rhomixed[basis_state]-self.targetrho[basis_state]), ord='fro')
        #reward =  (error<0.2*bits**2) #Maybe there is a nicer way to assort rewards?

        fidelity = self.calculateFidelity(targetrho[basis_state], rhomixed[basis_state])
        reward = 1-fidelity # fidelity always between 0 and 1, is 1 if perfect match

        return reward
    

    def calculateFidelity(target_rho, realized_rho):
        '''description'''

        sqrt_target = linalg.sqrtm(target_rho)
        matrix1 = np.matmul(realized_rho, sqrt_target)
        matrix2 = np.matmul(sqrt_target, matrix1)
        sqrt_matrix = linalg.sqrtm(matrix2)
        fidelity = (np.trace(sqrt_matrix))**2

        return fidelity
