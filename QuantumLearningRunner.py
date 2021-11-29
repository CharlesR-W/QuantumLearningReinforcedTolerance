from QuantumCircuitEnvironment import QuantumCircuitEnvironment
import numpy as np
import matplotlib.pyplot as plt

num_qubits = 1
num_gates = 2
num_samples = 1
sig = 1
gateNoiseParams = [[sig, sig, sig], [sig, sig, sig], sig]
targetBigRho = np.zeros([2**num_qubits, 2**num_qubits, 2**num_qubits])
targetBigRho[0,0,0] = 1
targetBigRho[1,1,1] = 1

myQuantumCircuitEnvironment = QuantumCircuitEnvironment(num_qubits, num_gates, num_samples, gateNoiseParams, targetBigRho)

from tensorforce.environments import Environment
environment = Environment.create(
    environment=myQuantumCircuitEnvironment, max_episode_timesteps=40
)

from tensorforce.agents import Agent
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=5, learning_rate=1e-3
)

# Train for 100 episodes
all_rewards = list()
for _ in range(10):
    episode_states = list()
    episode_internals = list()
    episode_actions = list()
    episode_terminal = list()
    episode_reward = list()

    states = environment.reset()
    internals = agent.initial_internals()
    terminal = False
    while not terminal:
        episode_states.append(states)
        episode_internals.append(internals)
        actions, internals = agent.act(
            states=states, internals=internals, independent=True
        )
        episode_actions.append(actions)
        states, terminal, reward = environment.execute(actions=actions)
        episode_terminal.append(terminal)
        episode_reward.append(reward)

    agent.experience(
        states=episode_states, internals=episode_internals,
        actions=episode_actions, terminal=episode_terminal,
        reward=episode_reward
    )
    agent.update()
    all_rewards.append(episode_reward[-1])
plt.plot(all_rewards)
plt.show()