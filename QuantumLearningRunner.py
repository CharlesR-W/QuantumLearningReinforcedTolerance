from QuantumCircuitEnvironment import QuantumCircuitEnvironment
import numpy as np
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
import matplotlib.pyplot as plt
from itertools import product


num_qubits = 1
num_gates = 2 # not including NUL
num_samples = 1
sig = 0.0
gateNoiseParams = [[sig, sig, sig], [sig, sig, sig], sig]
ket_rho_tradeoff = 0


targetBigrho = np.zeros((6**(num_qubits),2**(num_qubits), 2**(num_qubits)), dtype=complex)
cart_prod_states = list(product(['0', '1', '+', '-', 'r', 'l',], repeat=num_qubits))
initstring = ''
for basis_id in range(6**(num_qubits)):
    initstring = ''
    for bit in range(num_qubits):
        initstring = initstring + cart_prod_states[basis_id][bit]
    circ = QuantumCircuit(num_qubits)
    circ.initialize(initstring,circ.qubits)
    ####Put gates here
    circ.ry(np.pi/4, 0)
    #circ.s(0)
    ####
    targetBigrho[basis_id] = qi.DensityMatrix.from_instruction(circ)

reward_offset=0.0
reward_scale = 1
is_reward_exp = False

#print(targetBigstate)
myQuantumCircuitEnvironment = QuantumCircuitEnvironment(num_qubits, num_gates, num_samples, gateNoiseParams, targetBigrho, reward_offset, reward_scale, is_reward_exp)

from tensorforce.environments import Environment
environment = Environment.create(
    environment=myQuantumCircuitEnvironment, max_episode_timesteps=101
)


batch_size = 1
from tensorforce.core.networks import AutoNetwork



from tensorforce.agents import Agent
agent = Agent.create(
    agent=ProximalPolicyOptimization(
        states=environment.states()
        , actions=environment.actions()
        , learning_rate=dict(type='exponential', unit='episodes', num_steps=100, initial_value=1.0, decay_rate=1e-3)
        , exploration = dict(type='exponential', unit='episodes', num_steps=1000, initial_value=1.0, decay_rate=1e-1)
        , batch_size=batch_size
        , max_episode_timesteps=environment.max_episode_timesteps()
        )
    #, environment=environment
    #, tracking = 'all'
    #, memory = 2000
)


# Train for num_episodes
num_episodes = 1000
rewards = []
#plt.figure()
for episode in range(num_episodes):

        # Episode using act and observe
        states = environment.reset()
        terminal = False
        sum_rewards = 0.0
        num_updates = 0
        while not terminal:
            actions = agent.act(states=states)
            states, terminal, reward = environment.execute(actions=actions)
            if not terminal:
                reward = 0 #only include the reward at the end of the episode
            num_updates += agent.observe(terminal=terminal, reward=reward)
            sum_rewards += reward
        print('Episode {}: return={} updates={}'.format(episode, sum_rewards, num_updates))
        rewards.append(sum_rewards)
        
agent.close()
environment.close()
#%%
#batch_size *= 10
batch_avg_rwds = []
for lv in range(int(num_episodes / batch_size)):
    tmp = np.mean(rewards[lv*batch_size:(lv+1)*batch_size])
    batch_avg_rwds.append(tmp)
plt.plot(batch_avg_rwds)
    
