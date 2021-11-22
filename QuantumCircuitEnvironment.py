# -*- coding: utf-8 -*-
"""
Created on Mon Nov 22 11:02:19 2021

@author: CharlesRW
"""
from tensorforce import Environment

class QuantumCircuitEnvironment(Environment):
    def __init__(self):
        super().__init__()
        self.num_qubits = num_qubits
        self.num_gates = num_gates
        
    def states(self):
        return dict(type='float', shape=(2**(self.num_qubits*3+1),))

    def actions(self):
        return dict(type='int', num_values=self.num_gates) # needs to change for multi-qubits

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
        next_state = qc_append_gate()
        terminal = False  # Always False if no "natural" terminal state
        reward = qc_get_reward()
        return next_state, terminal, reward

