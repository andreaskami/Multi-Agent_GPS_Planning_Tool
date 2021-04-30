from utility.file_io import load_class_objects, store_class_objects, save_csv_solution
from utility.export_kml import export_to_kml
from utility.vrp_solver import VRPSolver
from utility.vrp_utils import render_path_planning_result

from multiagent.environment import MultiAgentEnv
from topology import Topology

import os

# overrides for non-defined attributes in older class objects
view_range_default = 10
agent_info_ground = {'life': 8*3600, 'speed': 1, 'vspeed': 0, 'range': float('inf')}
agent_info_default = {'life': 25 * 60, 'speed': 15, 'vspeed': 4, 'range': 4 * 1e3}

flag_defaults = {'landmark_amount': 10, 'agent_amount': 3, 'grid_size_l': 0.05, 'grid_rotation': 0.}


class WorldHandler:
    """
    WorldHandler is an overarcing class, which can create, load or store environments.
    It can apply algorithms to solve the environment.

    Used by the GUI.
    """
    def __init__(self, vrp_solver=None):
        if vrp_solver is None:
            self.solver = VRPSolver(None, None)
        else:
            self.solver = vrp_solver

        self.topology = None
        self.scenario = None
        self.world = None
        self.env = None

        self.output_path = '.'
        self.default_flags = flag_defaults

    def create_environment(self, preloaded, filepath, render, **flags):
        """Create an environment object from the loaded/created class objects and set flags."""
        self.env, self.scenario, self.world, self.topology = None, None, None, None  # reset class objects
        if preloaded:
            self.load_objects(filepath)
        else:
            self.create_topology(filepath, **flags)

        if not flags['ground_flag'] and self.topology.flags['ground_flag']:
            self.scenario.switch_to_aerial(self.world)
        self.scenario.set_agents(self.world, flags['agent_amount'])
        self.scenario.reset_world(self.world)

        # compatibility check for older class objects
        if not hasattr(self.scenario, 'view_range'):
            self.scenario.view_range = view_range_default
        if self.scenario.agent_info['speed'] == agent_info_default['speed'] and flags['ground_flag']:
            self.scenario.agent_info = agent_info_ground
            self.scenario.agent_info['distlimit'] = (agent_info_ground['life'] * agent_info_ground['speed']) / \
                self.scenario.topology.scale_factor

        self.env = MultiAgentEnv(self.world, self.scenario.reset_world, self.scenario.reward, self.scenario.observation,
                                 done_callback=self.scenario.check_done, info_callback=None, shared_viewer=True)

        self.solver.set_env_and_scenario(self.env, self.scenario)

        if render:
            self.env.render()
            self.env.render(end_screen=True)
            self.env.store_images(os.path.join(self.output_path, 'environment'))

    def load_objects(self, filepath):
        """Load environment from filepath."""
        self.topology, self.scenario, self.world = load_class_objects(filepath)

    def store_objects(self, filepath):
        """Store the currently loaded environment at filepath."""
        store_class_objects(self.topology, self.scenario, self.world, filepath)

    def create_topology(self, filepath, **flags):
        """Create a new topology form a .kml file."""
        self.topology = Topology(top=filepath, **flags)
        self.scenario, self.world = self.topology.setup_topology()

    def apply_algorithm(self, algorithm, store_im=False, store_kml=False, store_gps=True, **kwargs):
        """Apply an algorithm to the currently loaded environment. If render is True, also renders the output"""
        im_path = os.path.join(self.output_path, algorithm)
        res = self.solver.apply_algorithm(algorithm, store_im, im_path, **kwargs)

        if store_kml:
            export_to_kml(self.scenario, self.world, res['solution'],
                          os.path.join(self.output_path, algorithm + '_out.kml'))
        if store_gps:
            gps_coordinates = []
            for route in res['solution']:
                gps_coordinates.append(self.scenario.topology.get_real_world_coordinates(route))
            save_csv_solution(gps_coordinates, os.path.join(self.output_path, algorithm + '_coords'))
        return res

    def render_result(self, result_dict, im_path, ab=False):
        """Render a path planning result and store a screenshot at image_path."""
        render_path_planning_result(self.env, self.scenario, result_dict['solution'], result_dict['dists'], im_path, ab)

    def get_flags(self):
        """Get a dictionary containing all scenario flags and their current values."""
        return self.scenario.topology.flags

    def close_env(self):
        """Close rendered environment."""
        if self.env is not None:
            try:
                self.env.close()
                return ''
            except Exception as e:
                return e
        else:
            return ''

    def check_loaded(self):
        """Returns True if an environment is currently loaded into the class."""
        return self.env is not None and self.scenario is not None and self.world is not None
