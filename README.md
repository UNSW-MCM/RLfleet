# RLfleet
Reinforcement Learning to optimise fleet management

### Project description
This repository contains code to optimise the operation of a fleet of aircraft with RL.
An environment is designed where the fleet of aircraft is seen as a set of degrading systems that undergo variable loads as they fly missions and require maintenance throughout their lifetime. The objective is to find the fleet management strategy that maximises the fleet availability while minimising the maintenance costs.

### Environment description
Use the [0_test_environment.ipynb](./0_test_environment.ipynb) Jupyter Notebook to explore the environment class and its different attributes and methods. This is the environment that is used to test the RL algorithms. 

All the functions, classes, methods and RL algorithms are contained in the module [opfleet_env.py](./opfleet_env.py).

### Q-learning for different size fleets
Q-learning (with a look-up table) is used to demonstrate the potential of RL to optimise this complex decision-making problem.

Jupyter Notebooks:
- [1_fleet_optimisation_Qtable_2aircraft.ipynb](./1_fleet_optimisation_Qtable_2aircraft.ipynb)
- [2_fleet_optimisation_Qtable_3aircraft.ipynb](./2_fleet_optimisation_Qtable_3aircraft.ipynb)
- [3_fleet_optimisation_Qtable_4aircraft.ipynb](./3_fleet_optimisation_Qtable_4aircraft.ipynb)

The main limitation of this approach is that it cannot be applied to larger fleets as the state and action spaces increase exponentially and become intractable. Alternative representation of the action space using multi-agent RL are currently explored.

### Development of RL algorithms

So far, we have developed and tested a framework with Q-learning. The function that runs the Q-learning algorithm for the environment is `train_qtable(env,q_table,train_params)`. This method is used to optimise the environment for 3 different cases with fleets of 2, 3 and 4 aircraft respectively as shown in the Jupyter Notebooks above.

There is also a framework to run Deep Q-learning (DQN), where the look-up table (Q-table) is replaced with a neural network as a function approximator. This is implemented in `train_DQNagent(env,q_table,train_params)`. This framework is harder to run as there are more hyper-parameters and without the right values for those parameters the policy may not converge. This is covered in the Jupyter Notebook [4_fleet_optimisation_DQN_1aircraft.ipynb](./4_fleet_optimisation_DQN_1aircraft.ipynb]).

The main limitation of the Q-learning methods however, is that they cannot scale to larger fleets. Running the notebook with 4-aircraft takes about 2 weeks on a powerful machine (on CPU as it is sequential). As the state and action space increase exponentially with the number of aircraft in the fleet, there is a need to formulate the problem differently to avoid the combinatorial explosion. One way of overcoming this issue is to use **multi-agent RL**. Only initial progress has been done on the use of multi-agent RL:
1. Implement **Independent Q-learning (IQL)** where each agent is assigned an individual agent (Q-table) that learns independently. This can be used as a baseline and compared to the full Q-table examples in the Jupyter Notebooks. Started in [5_fleet_optimisation_IQL.ipynb](./5_fleet_optimisation_IQL.ipynb])
2. Add a cooperative behaviour to the IQL structure, for example a team reward using a Value Decomposition Network (VDN), as described in DeepMind's paper https://arxiv.org/abs/1706.05296.
3. If VDN does not work, implement the more complicated QMIX algorithm, which uses individual neural networks and a mixing network as described in https://arxiv.org/abs/1803.11485