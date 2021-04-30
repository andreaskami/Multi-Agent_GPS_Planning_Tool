from VRP.Christofides.christofides import ChristofidesApprox
import numpy as np


# TODO sometimes christofides produces infeasible cost results. Check why this is the case
def christofides_vrp(env, scenario, clustering='kmeans', **kwargs):
    """
    Calculates a Christofides-based VRP solution. The function needs pre-made clusters of target points, and tries to
    find an optimal path through them using Christofides' algorithm.

    :param env: an instance of the MultiAgentEnv class
    :param scenario: an instance of the Scenario class (see scenario.py)
    :param clustering: clustering type to make clusters. Currently only kmeans is supported (see scenario.py).
    :return: solution paths and used clustering type (in info)
    """
    scenario.set_distance_lut_diagonal(scenario.inf_dist)  # large diagonal value for christofides
    scenario.set_targets(env.world, target_init=clustering)  # divide paths between agents by using k-means clustering

    paths = []

    for i, agent in enumerate(env.world.agents):
        spawn = scenario.num_landmarks + agent.spawn

        if len(scenario.targets[i]) < 2:  # empty cluster found, or cluster with single item
            if len(scenario.targets[i]) == 0:
                paths.append([spawn])
            elif len(scenario.targets[i]) == 1:
                paths.append([spawn, scenario.targets[i][0], spawn])
            continue

        t_idx = np.array(scenario.targets[i], dtype=np.intp)
        dist_slice = scenario.distances[np.ix_(t_idx, t_idx)]
        solution = ChristofidesApprox(dist_slice)
        paths.append([t_idx[sol_idx] for sol_idx in solution['path']])

        # find fastest way to return home
        # TODO this might not be the best way - works good for circle-like far away clusters but not for other clusters?
        paths[i].pop(-1)  # remove final, repeated index
        dists_home = []
        for point in paths[i]:
            dists_home.append(scenario.distances[spawn][point])
        closest = dists_home.index(min(dists_home))
        new_path = [spawn] + paths[i][closest:] + paths[i][:closest] + [paths[i][closest]] + [spawn]
        paths[i] = new_path

    return {'solution': paths, 'info': {'cluster_type': clustering}}
