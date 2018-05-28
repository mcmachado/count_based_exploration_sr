# Count-based Exploration with the Successor Representation

## Tabular case:

`python3 tabular.py --input mdps/riverswim.mdp --num_episodes 100 --beta=100 --gamma 0.95 --epsilon 0.0 --seed 1`

`python3 tabular.py --input mdps/sixarms.mdp --num_episodes 1000 --beta=100 --gamma 0.95 --epsilon 0.0 --seed 1`

## Function Approximation case:

`python3 -m exp_eig_sr.train --rom ../roms/montezuma_revenge.bin`
