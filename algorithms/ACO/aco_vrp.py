from algorithms.ACO.ACO_train import train_ACO
from utility.vrp_utils import VRPDescriptor
import numpy as np


def aco_vrp(env, scenario, alpha=3, beta=12, rho=0.6, q=0.2, tol=5e-2,
            n_mean=10, max_iter=50, share_pheromone=True, mode=1, **kwargs):
    """
    Defaults parameters were: alpha=3, beta=5, rho=0.01, q=1, tol=5e-2, max_iter=50 from previous project
    The new params come from Gaertner & Clark, On Optimal Parameters for ACO algorithms
    Actually, parameters should vary for each instance - optimize this later

    :param env: an instance of the MultiAgentEnv class (currently not used by this function)
    :param scenario: an instance of the Scenario class (see scenario.py)
    :param alpha: ACO parameter
    :param beta: ACO parameter
    :param rho: evaporation coefficient, ACO parameter
    :param q: ACO parameter
    :param tol: tolerance
    :param n_mean: number of latest samples considered in the running mean
    :param max_iter: maximum number of iterations
    :param share_pheromone: share pheromone between ants
    :param mode: leave this at 1 for now
    """

    scenario.set_distance_lut_diagonal(0)  # needs 0 in diagonal for ACO
    vrp = VRPDescriptor(scenario)

    locations = np.array(scenario.get_locations(env.world))
    tours, length_opt, info = train_ACO(env, locations[scenario.num_landmarks:], locations[0:scenario.num_landmarks],
                                        scenario,
                                        alpha=alpha, beta=beta, evap_coeff=rho, q=q, tol=tol, max_iter=max_iter,
                                        n_mean=n_mean, share_pheromone=share_pheromone, mode=mode)
    info['length_opt'] = length_opt  # note that this does not include journey home of the ant

    vrp.wrap_routes(tours)  # add return home
    return {'solution': tours, 'info': info}
