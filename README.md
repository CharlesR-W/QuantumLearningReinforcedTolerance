# QuantumLearningReinforcedTolerance
This represents a project completed for the TU Delft Class AP3421-PR, "Quantum Information - Project"

The aim of the project was to develop a reinforcement learning environment which would train an agent which could learn to implement arbitrary quantum circuits, starting from a small universal set of quantum gates (c.f.  https://en.wikipedia.org/wiki/Quantum_logic_gate#Universal_quantum_gates, and https://en.wikipedia.org/wiki/Solovay%E2%80%93Kitaev_theorem)

We use the TensorForce package for reinforcement learning.

The file QuantumCircuitEnvironment contains the code pertaining to the environment, incl. circuit definitions, state descriptions, reward functions, etc.
The file QuantumLearningRunner contains all information and functions pertaining to the agent, including the neural network specification and the learning hyperparameters.

The project engaged several questions, focused on elementary applications to toy problems;
- What observations are necessary to fully characterize the current state to the agent? (i.e., what form must the state take such that the unitary gate implemented is uniquely identifiable - the tomography problem)
- Can an RL agent learn to implement a specified circuit if some of its gates are miscalibrated or poorly-toleranced?
- Is the learner successfully able to learn representations of the Clifford Gates on the basis of the universal gate set?
- How does the learner navigate the "length-of-circuit vs. accuracy" tradeoff?  How does this compare to the known Solovay-Kitaev bound?

Project proposal is attached along with presentation and final report.
