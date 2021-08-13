# Code for Reward Misspecification Experiments

## Based off of code from
- flow: https://github.com/flow-project/flow
- FinRL: https://github.com/AI4Finance-LLC/FinRL
- PandemicSimulator: https://github.com/SonyAI/PandemicSimulator
- RL4BG: https://github.com/MLD3/RL4BG

The `flow`, `finrl`, `pansim`, `bgp` folders hold the core of the flow, FinRL, PandemicSimulator, RL4BG repos, respectively. Each folder has its own README.md, which contains the README from the original repo. A few changes have been made to original code, but the structure of the code is mostly the same. 

## Requirements
- python == 3.7
- SUMO (for flow): https://github.com/eclipse/sumo
- torch == 1.3.1 (for RL4BG) || torch >= 1.4 (for stable-baselines3, which FinRL and PandemicSimulator use)
- stable-baselines3: https://github.com/DLR-RM/stable-baselines3  

