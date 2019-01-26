# Author: Marlos C. Machado

import utils
import random
import numpy as np
import environment as env
from ssr_bellman_q import SSRBellmanQ

if __name__ == "__main__":
    # Read arguments:
    args = utils.ArgsParser.read_input_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Instantiate objects I'll need
    environment = env.MDP(args.input)

    # Actual learning algorithm
    for ep in range(args.num_episodes):
        agent = SSRBellmanQ(environment, args.gamma, args.epsilon, beta=args.beta)
        time_step = 1
        while not environment.is_terminal():
            # print(time_step)
            agent.step()
            time_step += 1
        environment.reset()
        print(ep, ",", agent.get_avg_undisc_return())
