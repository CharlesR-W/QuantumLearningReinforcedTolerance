# -*- coding: utf-8 -*-
"""
Created on Fri Nov 26 17:00:17 2021

@author: CharlesRW
"""

from tensorforce.environments import Environment
environment = Environment.create(
    environment=CustomEnvironment, max_episode_timesteps=100
)

from tensorforce.agents import Agent
agent = Agent.create(
    agent='ppo', environment=environment, batch_size=10, learning_rate=1e-3
)


from tensorforce.agents import Runner
runner = Runner(
    agent=agent
    environment=environment
)

runner.run(num_episodes=200)

runner.run(num_episodes=100, evaluation=True)

runner.close()