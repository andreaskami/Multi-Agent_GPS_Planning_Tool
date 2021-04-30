import numpy as np
import pickle
import os

from scenario import Scenario

from utility.import_kml import KMLTopology
from utility.misc import generate_random_polygon, normalize_coordinates


class Topology:
    """
    A Topology object stores the location data for the environment.
    Its setup_topology() function returns a Scenario and World object from this location data.
    The get_coordinates() function can be used to revert relative location coordinates to actual GPS (if applicable).
    The override_flags() function can be used to override user flags (see __init__() documentation).
    A Topology object can be stored and loaded from file using store_topology() and load_topology().

    Correct usage of .kml files is further explained in HowToMakeKMLTopology.md
    """
    def __init__(self, top='basic', size=None, landmark_amount=10, agent_amount=3, spawn_amount=None, grid_size=0.05,
                 grid_size_l=0.05, grid_rotation=0., collaborative=False, just_planning=False, ground_flag=False,
                 from_grid_l=False, cluster_l=False, row_structure=False, random_spawn_l=False, random_spawn_a=False,
                 **kwargs):
        """
        :param top: use 'basic' for a basic topology or a filepath to a valid .kml file (for loading Google Earth data)
        :param size: bounds for the output environment, default is -1 to 1 in both directions
        :param landmark_amount: amount of landmarks to add to environment - any more than predefined are added randomly
        :param agent_amount: amount of agents to add to environment, divided equally between the number of spawns
        :param spawn_amount: amount of places to spawn agents, should be at least 1
        :param grid_size: size of underlying navigational grid
        :param grid_size_l: grid size for grid to spawn landmarks on
        :param grid_rotation: the amount (in radians) the grid should be rotated
        :param collaborative: currently not used, intented for making the env collaborative, cooperative or mixed
        :param just_planning: whether the environment is just for path planning or a continuous, dynamic environment
        :param ground_flag: if True, agents are ground-based, if False, they are aerial and can fly over nonhard objects
        :param from_grid_l: if True, landmark_amount is ignored and a landmark is spawned on each grid points instead.
        :param cluster_l: if True, random landmarks are initialized in clusters.
        :param row_structure: if True, rows will be added based on the underlying grid. Works best if from_grid_l = True
        :param random_spawn_l: whether or not to spawn (landmark_amount) landmarks randomly initially and each reset
        :param random_spawn_a: whether or not to create (spawn_amount) agent spawns randomly at start (not each reset!)
        """
        if size is None:
            size = [(-1, -1), (1, 1)]  # default size
            
        self.flags = {
            'size': np.asarray(size),
            'landmark_amount': landmark_amount,  # not used when getting landmarks from grid
            'agent_amount': agent_amount,
            'spawn_amount': spawn_amount,  # amount of spawns to use. Default=1
            'grid_size': grid_size,
            'grid_size_l': grid_size_l,
            'grid_rotation': grid_rotation,
            'collaborative': collaborative,  # currently not used
            'just_planning': just_planning,
            'ground_flag': ground_flag,
            'from_grid_l': from_grid_l,
            'cluster_l': cluster_l,
            'row_structure': row_structure,
            'random_spawn_l': random_spawn_l,
            'random_spawn_a': random_spawn_a
        }

        self.landmarks = [] 
        self.agents = []
        self.spawns = []  # Useful for calculating distance table later on
        self.walls = []
        self.boundary = None

        self.names = {}
        self.boundary_file_loc = ''

        self.scale_factor = 1.0  # holds scale factor for conversion to meters
        self.reverse_bounds = []  # store world conversion variables here to revert to coordinates later on

        if top == 'basic':
            self.generate_basic_topology()
        else:  # topology from .kml file path
            self.generate_topology_kml(top)

    def setup_topology(self):
        """Creates a Scenario object, and from that, a World object for initial setup."""
        self.assert_flag_values()
        self.set_landmarks_and_agents()
        scenario = Scenario(self)
        world = scenario.make_world()
        return scenario, world

    def generate_basic_topology(self):
        """Very basic manual topology to demonstrate the creation of a custom, non-kml topology."""
        self.names['agents'] = []
        self.names['landmarks'] = []
        self.names['spawns'] = ['spawn_0']
        for i in range(self.flags['agent_amount']):
            self.agents.append([.0, .0+0.05*i])
            self.names['agents'].append('agent_%d' % i)
            self.landmarks.append([])  # one landmark per agent
            self.names['landmarks'].append('agent_%d' % i)
        self.spawns = [(.0, .0)]
        wall_1 = [[.1, .2], [.1, .4], [.3, .4], [.3, .2]]
        self.walls = [wall_1]
        self.names['walls'] = ['obstacle_0']
        self.boundary = generate_random_polygon(n=15, ctr_loc=(.0, .0), avg_radius=.9, min_radius=0.4,
                                                max_radius=1.0, irregularity=.2, spikeyness=.2)
        self.names['boundary'] = 'boundary'

    def generate_topology_kml(self, path):
        """Get location data from a .kml file (Google Earth)."""
        kml_topology = KMLTopology(path, self.flags['size'])
        self.boundary, self.walls, self.landmarks, self.spawns = kml_topology.get_world_locations()
        self.names = kml_topology.names
        self.boundary_file_loc = kml_topology.boundary_location
        self.scale_factor = 1 / kml_topology.scale_factor
        self.reverse_bounds = [kml_topology.bounds_meters, kml_topology.bounds]

    def assert_flag_values(self):
        """Makes sure flag values are up-to-date."""
        if self.flags['landmark_amount'] is None:
            self.flags['landmark_amount'] = len(self.landmarks)
        if self.flags['spawn_amount'] is None:
            self.flags['spawn_amount'] = len(self.spawns)
        elif self.flags['spawn_amount'] > len(self.spawns):
            if self.flags['random_spawn_a']:
                self.spawns += [[] for _ in range(self.flags['spawn_amount'] - len(self.spawns))]
            else:
                self.flags['spawn_amount'] = len(self.spawns)

    def set_landmarks_and_agents(self):
        """Sets the actual number of landmarks of agents based on the input flags and the kml data."""
        if self.flags['random_spawn_l']:
            self.landmarks = [[] for _ in range(self.flags['landmark_amount'])]
            self.names['landmarks'] = ['landmark_%d' % i for i in range(self.flags['landmark_amount'])]

        landmarks_to_add_randomly = 0
        if len(self.landmarks) + landmarks_to_add_randomly < self.flags['agent_amount']:
            landmarks_to_add_randomly += self.flags['agent_amount'] - len(self.landmarks)
        if len(self.landmarks) + landmarks_to_add_randomly < self.flags['landmark_amount']:
            landmarks_to_add_randomly += self.flags['landmark_amount'] - len(self.landmarks)
        self.landmarks += [[] for _ in range(landmarks_to_add_randomly)]
        self.names['landmarks'] += ['rand_landmark_%d' % i for i in range(landmarks_to_add_randomly)]

        # remove landmarks if landmark amount has changed to be smaller than the number of saved landmarks
        if self.flags['landmark_amount'] < len(self.landmarks):
            self.landmarks = list(self.landmarks[0:self.flags['landmark_amount']])

        assert len(self.landmarks) == self.flags['landmark_amount'], 'Amount of landmarks should match flag.'

        # self.agents = [self.spawns[i % self.flags['spawn_amount']] for i in range(self.flags['agent_amount'])]
        self.agents = [[] for _ in range(self.flags['agent_amount'])]
        self.names['agents'] = ['agent_%d' % i for i in range(self.flags['agent_amount'])]

        assert len(self.agents) > 0, 'Error: at least one agent position should be defined at all times.'
        assert len(self.landmarks) > 0, 'Error: at least one landmark position should exist at all times.'
        assert len(self.landmarks) >= len(self.agents), 'Error: num_landmarks should be >= num_agents.'

    def get_real_world_coordinates(self, locations):
        """Note: GPS coordinates given in (longitude, latitude), not (latitude, longitude)"""
        if len(self.reverse_bounds) == 0:
            print('Unable to revert to GPS coordinates. No conversion bounds specified beforehand.')
            print('Returning original location coordinates...')
            return locations
        coordinates, _ = normalize_coordinates(locations, self.flags['size'], self.reverse_bounds[0],
                                               keep_aspect_ratio=True, aspect_max=True)
        coordinates, _ = normalize_coordinates(coordinates, self.reverse_bounds[0], self.reverse_bounds[1])
        return coordinates

    def get_environment_coordinates(self, locations):
        """Note: GPS coordinates taken in (longitude, latitude), not (latitude, longitude)"""
        if len(self.reverse_bounds) == 0:
            print('Warning: unable to revert to env coordinates. No conversion bounds specified beforehand.')
            print('Returning original coordinates...')
            return locations
        coordinates, _ = normalize_coordinates(locations, self.reverse_bounds[1], self.reverse_bounds[0])
        coordinates, _ = normalize_coordinates(coordinates, self.reverse_bounds[0], self.flags['size'],
                                               keep_aspect_ratio=True)
        return coordinates

    def store_topology(self, file_path):
        """Save the current topology to a .dictionary file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path + '_topology.dictionary', 'wb') as top_dict:
            pickle.dump(self.__dict__, top_dict)

    def load_topology(self, file_path):
        """Load a topology from .dictionary file."""
        with open(file_path + '_topology.dictionary', 'rb') as top_dict:
            self.__dict__ = pickle.load(top_dict)

    def override_flags(self, **kwargs):
        """Overrides any number of variables in the flag dict. Currently not used."""
        for kw in kwargs.keys():
            if kw in self.flags.keys():
                if kw == 'size':
                    self.flags[kw] = np.asarray(kwargs[kw])
                else:
                    self.flags[kw] = kwargs[kw]
            else:
                print('Unknown flag not set: ' + kw)
