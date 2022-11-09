# RLfleet
Reinforcement Learning to optimise fleet management

### Project description
This repository contains code to optimise the operation of a fleet of aircraft with RL.
An environment is designed where the fleet of aircraft is seen as a set of degrading systems that undergo variable loads as they fly missions and require maintenance throughout their lifetime. The objective is to find the fleet management strategy that maximises the fleet availability while minimising the maintenance costs.

### Q-learning for small state-aciton spaces
Q-learning is used to demonstrate the potential of RL to optimise this complex decision-making problem.

Example Jupyter Notebooks:
- [fleet_optimisation_Qtable_2aircraft.ipynb](./fleet_optimisation_Qtable_2aircraft.ipynb)
- [fleet_optimisation_Qtable_3aircraft.ipynb](./fleet_optimisation_Qtable_3aircraft.ipynb)

All the functions, classes and methods are contained in the module [opfleet_env.py](./opfleet_env.py).

The main limitation of this approach is that it cannot be applied to larger fleets as the state and action space increase exponentially and become intractable.
