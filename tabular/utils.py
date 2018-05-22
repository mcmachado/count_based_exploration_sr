# Author: Marlos C. Machado

import argparse
import numpy as np


def argmax(vector):
    # This argmax breaks ties randomly
    return np.random.choice(np.flatnonzero(vector == vector.max()))


class ArgsParser:
    """
    Read the user's input and parse the arguments properly. When returning args, each value is properly filled.
    Ideally one shouldn't have to read this function to access the proper arguments, but I postpone this.
    """

    @staticmethod
    def read_input_args():
        # Parse command line
        parser = argparse.ArgumentParser(
            description='Define algorithm\'s parameters.')

        parser.add_argument('-s', '--seed', type=int, default=1, help='Seed to be used in the code.')
        parser.add_argument('-i', '--input', type=str, default='mdps/toy.mdp',
                            help='File containing the MDP definition (default: mdps/toy.mdp).')
        parser.add_argument('-n', '--num_episodes', type=int, default=1000,
                            help='For how many episodes we are going to learn.')
        parser.add_argument('-a', '--step_size', type=float, default=0.1,
                            help="Algorithm's step size. Alpha parameter in algorithms such as Sarsa.")
        parser.add_argument('-b', '--beta', type=float, default=1.0,
                            help="Real reward = Real reward + beta * Intrinsic Reward.")
        parser.add_argument('-g', '--gamma', type=float, default=0.95,
                            help='Gamma. Discount factor to be used by the algorithm.')
        parser.add_argument('-e', '--epsilon', type=float, default=0.05,
                            help='Epsilon. This is the exploration parameter (trade-off).')

        args = parser.parse_args()

        return args
