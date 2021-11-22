# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:02:19 2021

@author: CharlesRW
"""
from tensorforce import Environment
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.providers.aer import QasmSimulator
from qiskit.visualization import plot_histogram
import itertools as it

class QuantumCircuitEnvironment(Environment):
    def __init__(self,num_qubits,num_gates, num_samples):
        super().__init__():
            #Number of qubits to simulate
            self.num_qubits = num_qubits

            #number of times to simulate a given QC to find rho
            self.num_samples = num_samples

            #NB this is NOT the number of distinct 1-qubit gates, but is the total possible number of actions
            # i.e. in the case of Hadamard, would have H1, H2...Hn
            self.num_gates = num_gates 
            self.simulator = QasmSimulator()
            self.IdealCircuit = QuantumCircuit(num_qubits, 0)
        
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
        self.updateIdealCircuit(actions)
        #sample ensemble to get rhos after action
        self.evaluateDMs()
        #next-state = sampled rhos
        next_state = 
        #reward = reward(sampled rhos)
        reward = 
        terminal = False  # Always False if no "natural" terminal state
        
        return next_state, terminal, reward

    #For a given ideal circuit, evaluate the density matrix for each possible input state
    #Do multiple times to sample different gate noise, and return the average rhos
    def evaluateDMs(self.num_samples):

    def updateIdealCircuit(action):
        #action is an int between 0 and self.num_gates
        #need to specify the gate to be applied and the qubit line on which to apply it
        #Assume gate order is Rz1,Rz2... S1,S2,.... CNOT(1,1) CNOT(1,2)...
        #
        n = self.num_qubits

        if action < n
            qbit_idx = action
            self.IdealCircuit.Z(qbit_idx)
        elif action < 2*n
            qbit_idx = action - n
            self.IdealCircuit.S(qbit_idx)
        else
            tmp_idx = action - 2*n
            qbit_idx2 = tmp_idx % n
            qbit_idx1 = (tmp_idx-qbit_idx2) / n
            self.IdealCircuit.CNOT(qbit_idx1, qbit_idx2) 

        qubitindices = np.arange(0,n)
        gatecombos = it.product(qubitindices, qubitindices)
        f

        for combo in gatecombos:
            if combo[0] != combo[1]:

        if action < n
            qbit_idx = action
            self.IdealCircuit.Z(qbit_idx)
        elif action < 2*n
            qbit_idx = action - n
            self.IdealCircuit.S(qbit_idx)
        else
            tmp_idx = action - 2*n
            qbit_idx2 = tmp_idx % n
            qbit_idx1 = (tmp_idx-qbit_idx2) / n
            self.IdealCircuit.CNOT(qbit_idx1, qbit_idx2) 
        
        else:
            combo_idx = action-2*n
            combo = gatecombos[combo_idx]
            self.Idealcircuit.cnot(combo[0],combo[1])

        
            
        
        
