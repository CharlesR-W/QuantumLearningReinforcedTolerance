from QuantumCircuitEnvironment import QuantumCircuitEnvironment
import numpy as np

num_qubits = 1
num_gates = 2
num_samples = 15
sig = 0.3
gateNoiseParams = [[sig, sig, sig], [sig, sig, sig], sig]
targetBigRho = np.zeros([2**num_qubits, 2**num_qubits, 2**num_qubits])
targetBigRho[0,0,0] = 1
targetBigRho[1,1,1] = 1

myQuantumCircuitEnvironment = QuantumCircuitEnvironment(num_qubits, num_gates, num_samples, gateNoiseParams, targetBigRho)

from tensorforce.environments import Environment
environment = Environment.create(
    environment=myQuantumCircuitEnvironment, max_episode_timesteps=2
)

from tensorforce.agents import Agent
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)


from tensorforce.execution import Runner
runner = Runner(
    agent=agent,
    environment=environment
)

runner.run(num_episodes=1, evaluation=True)

runner.close()
