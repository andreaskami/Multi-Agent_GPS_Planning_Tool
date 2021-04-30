from algorithms.Christofides.christofides_vrp import christofides_vrp
from algorithms.ACO.aco_vrp import aco_vrp
from algorithms.Clarke_Wright.clarke_wright import clarke_wright
from algorithms.GA.genetic_algorithm import genetic_algorithm
from algorithms.DRL_VRP.inference import drlvrp_inference
from algorithms.OR_Tools.or_tools import or_tools_sa, or_tools_ts, or_tools_gls
from algorithms.POPCORN.popcorn import popcorn_cpp

from utility.vrp_utils import render_path_planning_result, cost
from utility.misc import get_path_distances

import time

agent_info_override = {'life': 8*3600, 'speed': 1, 'vspeed': 0, 'range': float('inf')}


class VRPSolver:
    def __init__(self, env, scenario, override_agent_specs=False):
        """
        This class can be used to generate solutions to the VRP, based on several included algorithms.
        New algorithms can be added through the use of add_new_algorithm().
        apply_all_algorithms() runs all algorithms available, and apply_algorithm can be used to apply a single method.
        This is a useful function, since keyword arguments can be set for each algorithm separately.

        :param env: an instance of the MultiAgentEnv class
        :param scenario: an instance of the Scenario class (see scenario.py)
        """
        self.env = env
        self.scenario = scenario

        if override_agent_specs:  # old scenario instances have only air-based agent info. Override with ground specs
            self.scenario.agent_info = agent_info_override
            self.scenario.agent_info['distlimit'] = (agent_info_override['life'] * agent_info_override['speed']) / \
                                                     self.scenario.topology.scale_factor

        self.callbacks = {'christofides': christofides_vrp, 'aco': aco_vrp, 'ga': genetic_algorithm,
                          'cw': clarke_wright, 'sa': or_tools_sa, 'gls': or_tools_gls, 'ts': or_tools_ts,
                          'popcorn': popcorn_cpp, 'drlvrp': drlvrp_inference} 

        self.names = {'christofides': 'Christofides with k-means', 'aco': 'Ant Colony Optimization',
                      'ga': 'Genetic Algorithm', 'cw': 'Clarke-Wright', 'sa': 'Simulated Annealing',
                      'gls': 'Guided Local Search', 'ts': 'Tabu Search', 'popcorn': 'POPCORN',
                      'drlvrp': 'VRP Reinforcement Learning'} 

        self.options = list(self.callbacks.keys())
        self.results = {}

    def set_env_and_scenario(self, env, scenario):
        """Set new environment and scenario for solver"""
        self.env = env
        self.scenario = scenario

    def apply_all_algorithms(self, render=False):
        """Apply all stored algorithms sequentially. Fills the result dict and returns this as well."""
        for algorithm in self.options:
            self.results[algorithm] = self.apply_algorithm(algorithm, render)
        return self.results

    def apply_algorithm(self, algorithm, render=False, im_path='', **kwargs):
        """
        Apply the specified algorithm to the environment. Takes named keyword args.
        Can also render the result, and store the resulting image (if render=True and im_path is provided).
        """
        # print('----------------------------------------------------------')
        # print('Getting agent paths using ' + algorithm + '...')
        # print('----------------------------------------------------------')

        callback = self.callbacks.get(algorithm, None)
        if callback is None:
            raise ValueError('Unsupported algorithm for the VRP solver')
        if len(im_path) == 0 and render:
            raise ValueError('Empty image storage path provided (might be default)')

        self.env.reset()
        start_time = time.time()

        return_values = callback(self.env, self.scenario, **kwargs)

        comp_time = time.time() - start_time

        solution = return_values['solution']
        info = return_values.get('info', None)
        # get path lengths etc.
        absolute = not isinstance(solution[0][0], int)
        dists = get_path_distances(solution, self.scenario.distances, absolute=absolute)
        if render:
            render_path_planning_result(self.env, self.scenario, solution, dists, im_path, absolute=absolute)
        # return dict with results
        for key in dists.keys():
            dists[key] *= self.scenario.topology.scale_factor
        run_time = dists['max'] / self.scenario.agent_info['speed']

        return {'solution': solution, 'dists': dists, 'run_time': run_time,
                'cost': cost(dists), 'comp_time': comp_time, 'info': info}

    def add_new_algorithm(self, tag, callback, name=''):
        """
        Adds a new algorithm to the solver for easy access.

        :param tag: a string with the algorithm's name
        :param callback: a callback to the main function of the algorithm. Should take an environment and scenario as
        first arguments, and further arguments should be optional named arguments with default values.
        :param name: optionally, a full algorithm name can be added as well.
        See also apply_algorithm() for how the algorithm will be called.
        """
        self.options.append(tag)
        self.callbacks[tag] = callback
        if len(name) > 0:
            self.names[tag] = name
        else:
            self.names[tag] = tag
