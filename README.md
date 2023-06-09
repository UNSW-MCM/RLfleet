# RLfleet
Reinforcement Learning to optimise fleet management

### Project description
This repository contains code to optimise the operation of a fleet of aircraft with RL.
An environment is designed where the fleet of aircraft is seen as a set of degrading systems that undergo variable loads as they fly missions and require maintenance throughout their lifetime. The objective is to find the fleet management strategy that maximises the fleet availability while minimising the maintenance costs.

Jupyter Notebooks:
- [0_test_environment.ipynb](./0_test_environment.ipynb)
- [1_fleet_optimisation_Qtable_2aircraft.ipynb](./1_fleet_optimisation_Qtable_2aircraft.ipynb)
- [2_fleet_optimisation_Qtable_3aircraft.ipynb](./2_fleet_optimisation_Qtable_3aircraft.ipynb)
- [3_fleet_optimisation_Qtable_4aircraft.ipynb](./3_fleet_optimisation_Qtable_4aircraft.ipynb)
- [4_fleet_optimisation_DQN_1aircraft.ipynb](./4_fleet_optimisation_DQN_1aircraft.ipynb)
- [5_fleet_optimisation_IQL.ipynb](./5_fleet_optimisation_IQL.ipynb)

### Environment description
Use the [0_test_environment.ipynb](./0_test_environment.ipynb) Jupyter Notebook to explore the environment class and its different attributes and methods. This is the environment that is used to test the RL algorithms. 

All the functions, classes, methods and RL algorithms are contained in the module [opfleet_env.py](./opfleet_env.py).

### Q-learning for different size fleets
Q-learning (with a look-up table) is used to demonstrate the potential of RL to optimise this complex decision-making problem.
This approach is tested on three environments with respectively 2, 3 and 4 aircraft ([1_fleet_optimisation_Qtable_2aircraft.ipynb](./1_fleet_optimisation_Qtable_2aircraft.ipynb),[2_fleet_optimisation_Qtable_3aircraft.ipynb](./2_fleet_optimisation_Qtable_3aircraft.ipynb),[3_fleet_optimisation_Qtable_4aircraft.ipynb](./3_fleet_optimisation_Qtable_4aircraft.ipynb)). The results indicate that the Q-learning policy is capable of outperforming baseline fleet management strategies like on-condition maintenance and force-life management.

The main limitation of this approach is that it cannot be applied to larger fleets as the state and action spaces increase exponentially and become computationally intractable. Alternative representation of the action space using multi-agent RL are currently explored, by having a centralized planning with decentralized execution we can decouple the size of the state and action spaces from the number of aircraft in the fleet and avoid the combinatorial explosion.

### Development of multi-agent RL (MARL) algorithms

So far, we have developed and tested a framework with Q-learning. The function that runs the Q-learning algorithm for the environment is `train_qtable(env,q_table,train_params)`. This method is used to optimise the environment for 3 different cases with fleets of 2, 3 and 4 aircraft respectively as shown in the Jupyter Notebooks above.

There is also a framework to run Deep Q-learning (DQN), where the look-up table (Q-table) is replaced with a neural network as a function approximator. This is implemented in `train_DQNagent(env,q_table,train_params)`. This framework is harder to run as there are more hyper-parameters and without the right values for those parameters the policy may not converge. This is covered in the Jupyter Notebook [4_fleet_optimisation_DQN_1aircraft.ipynb](./4_fleet_optimisation_DQN_1aircraft.ipynb).

The main limitation of the Q-learning and DQN methods however, is that they cannot scale to larger fleets. Running the notebook with 4-aircraft takes about 2 weeks on a powerful machine (on CPU as it is sequential). As the state and action space increase exponentially with the number of aircraft in the fleet, there is a need to formulate the problem differently to avoid the combinatorial explosion. One way of overcoming this issue is to use **multi-agent RL**, where there is centralized planning (central controlled optimising the team reward) and decentrilzed execution (agents deployed individually for each aircraft to avoid the combinatorial explosion).

Only initial progress has been done on the use of multi-agent RL, with the implementation of **Independent Q-learning (IQL)** in the `train_IQL(env,q_tables,train_params)` function.
In IQL, each agent is assigned an individual agent (Q-table) that learns independently, so there is not explicit cooperative behaviour. Generally it is used as a baseline to evaluate other MARL algorithms. In the Jupyter notebook [5_fleet_optimisation_IQL.ipynb](./5_fleet_optimisation_IQL.ipynb) IQL is run on the 2,3,4 aircaft environments (same as previously run with Q-learning) as well as on a larger 6-aircraft environments (with 2 mission types and 5 mission types). The results show that IQL can outperform the baseline policies (on-condition and force-life management) and in certain cases even the centralized Q-learning policy.

Next steps:
1. Add a cooperative behaviour to the IQL structure, for example a team reward using a Value Decomposition Network (VDN), as described in DeepMind's paper https://arxiv.org/abs/1706.05296.
2. If VDN does not work, implement the more complicated QMIX algorithm, which uses individual neural networks and a mixing network as described in https://arxiv.org/abs/1803.11485