# Count-based Exploration with the Successor Representation

These are the commands we used to obtain the results reported in the [Count-based Exploration with the Successor Representation](https://arxiv.org/abs/1807.11622). For the function approximation case the rom name should be adapted for different games, of course. This assumes one has the [Arcade Learning Environment](https://github.com/mgbellemare/Arcade-Learning-Environment) properly installed in their computer, as well as the other dependencies.

## Tabular case:

`python3 tabular_mbrl.py --input mdps/riverswim.mdp --num_episodes 100 --beta=100 --gamma 0.95 --epsilon 0.0 --seed 1`

`python3 tabular_mbrl.py --input mdps/sixarms.mdp --num_episodes 100 --beta=100 --gamma 0.95 --epsilon 0.0 --seed 1`

`python3 tabular_sarsa_sr.py --input mdps/riverswim.mdp --num_episodes 100 --beta 10000 --gamma 0.95 --epsilon 0.01 --seed 1 --step_size 0.1 --step_size_sr 0.5 --gamma_sr 0.5`

`python3 tabular_sarsa_sr.py --input mdps/sixarms.mdp --num_episodes 100 --beta 10000 --gamma 0.95 --epsilon 0.01 --seed 1 --step_size 0.5 --step_size_sr 0.25 --gamma_sr 0.5`

`python3 tabular_sarsa.py --input mdps/riverswim.mdp --num_episodes 100 --gamma 0.95 --epsilon 0.12 --seed 1 --step_size 0.37`

`python3 tabular_sarsa.py --input mdps/sixarms.mdp --num_episodes 100 --gamma 0.95 --epsilon 0.01 --seed 1 --step_size 0.43`

## Function Approximation case:

`python3 -m exp_eig_sr.train --rom ../roms/montezuma_revenge.bin`
