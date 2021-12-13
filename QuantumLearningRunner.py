from QuantumCircuitEnvironment import QuantumCircuitEnvironment
from qiskit import QuantumCircuit
import qiskit.quantum_info as qi
import numpy as np
import matplotlib.pyplot as plt



num_qubits = 1
num_gates = 2 # not including NUL
num_samples = 1
sig = 0.
gateNoiseParams = [[sig, sig, sig], [sig, sig, sig], sig]
ket_rho_tradeoff = 0
#targetBigRho = np.zeros([2**num_qubits, 2**num_qubits, 2**num_qubits])
#targetBigRho[0,0,0] = 1
#targetBigRho[1,1,1] = 1

#OLD CODE
#targetBigRho = np.zeros([2**num_qubits, 2**num_qubits, 2**num_qubits])
#targetBigRho[0,0,0] = 1
#targetBigRho[1,1,1] = 1

# targetBigRho = np.zeros((2,2,2), dtype= np.complex64)

# circ = QuantumCircuit(1)
# circ.initialize("0",circ.qubits)
# #circ.ry(np.pi/4, 0)
# #circ.s(0)
# targetBigRho[0] = qi.DensityMatrix.from_instruction(circ)

# #circ = QuantumCircuit(1)
# circ.initialize("1",circ.qubits)
# #circ.ry(np.pi/4, 0)
# #circ.s(0)
# targetBigRho[1] = qi.DensityMatrix.from_instruction(circ)

targetBigState = np.zeros((2**(num_qubits),2**(num_qubits)),dtype='complex')
for basis_id in range(2**(num_qubits)):
    binary = np.binary_repr(basis_id,width=num_qubits)
    circ = QuantumCircuit(num_qubits)
    circ.initialize(binary,circ.qubits)
    ####Put gates here
    #circ.ry(np.pi/8, 0)
    circ.s(0)
    ####
    targetBigState[basis_id] = qi.Statevector.from_instruction(circ)


myQuantumCircuitEnvironment = QuantumCircuitEnvironment(num_qubits, num_gates, num_samples, gateNoiseParams, targetBigState, ket_rho_tradeoff)

from tensorforce.environments import Environment
environment = Environment.create(
    environment=myQuantumCircuitEnvironment, max_episode_timesteps=100
)

batch_size = 10
from tensorforce.agents import Agent
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=batch_size, learning_rate=0.0001, exploration = 0.001
)

# Train for num_episodes
num_episodes = 1000

#plt.figure()
rewards = list()
for _ in range(num_episodes):

    # Initialize episode
    states = environment.reset()
    terminal = False
    
    if _ % batch_size == 0:
        print(_)
        
    while not terminal:
        # Episode timestep
        actions = agent.act(states=states)
        states, terminal, reward = environment.execute(actions=actions)
        agent.observe(terminal=terminal, reward=reward)

        
    rewards.append(reward)
#plt.show()
agent.close()
environment.close()


tmp = []
for lv in range(int(np.floor(num_episodes/batch_size))):
    tmp.append(np.mean(rewards[lv*batch_size:(lv+1)*batch_size]))


plt.plot(tmp)