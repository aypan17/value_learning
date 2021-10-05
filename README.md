# Code for Reward Misspecification Experiments

## Based off of code from
- flow: https://github.com/flow-project/flow
- PandemicSimulator: https://github.com/SonyAI/PandemicSimulator
- RL4BG: https://github.com/MLD3/RL4BG
- torchbeast: https://github.com/facebookresearch/torchbeast

The `flow`, `pansim`, `bgp`, `torchbeast` folders hold the core of the flow, PandemicSimulator, RL4BG, and torchbeast repos, respectively. Each folder has its own README.md, which contains the README from the original repo. A few changes have been made to original code, but the structure of the code is mostly the same. 

## Requirements
- python == 3.7
- SUMO (for flow): https://github.com/eclipse/sumo
- torch == 1.3.1 (for RL4BG; this is important) || torch >= 1.4 (for stable-baselines3, which PandemicSimulator uses)
- stable-baselines3: https://github.com/DLR-RM/stable-baselines3  

