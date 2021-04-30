import numpy as np
import pickle
import os
from time import time

from sklearn.cluster import KMeans

from shapely.geometry.polygon import Polygon
from shapely.geometry.linestring import LineString

from multiagent.core import World, Agent, Landmark, Boundary, Entity
from multiagent.scenario import BaseScenario

from utility.misc import distance
from utility.grid import Grid


class Scenario(BaseScenario):
    """
    The Scenario object manages the world. It contains functions that define the agents' rewards and observations,
    as well as many useful functionalities such as the distance lookup table, targets, random spawning of landmarks etc.
    The Scenario objects needs a Topology object for its initial setup, which contains some user-defined flags
    but most importantly, location data (coordinates) for the world.

    Often used functions outside of this class:
    make_world() - create initial world setup
    reset_random_landmarks() - reset the POI locations randomly
    update_distance_table() - update distance lookup table
    set_targets() - get or set new targets for the agents
    reset_world() - necessary call after updating the LUT, changing flags, or resetting landmarks

    Often used attributes:
    scenario.distances - the distance lookup table
    scenario.targets - current targets of the agents
    scenario.num_agents, num_landmarks, num_walls, num_spawns - amount of each object in the world
    scenario.global_visited - the amount of visited POIs
    """
    def __init__(self, topology, agent_info=None, infeasible_distance_factor=1000):
        """
        :param topology: a complete Topology object (see topology.py).
        :param agent_info: a dict with a few variables for the agent: max battery life in s, speed in m/s
        :param infeasible_distance_factor: cost value that represents infeasible distances is max_dist times this factor
        """
        # print('Initializing Scenario...')
        # store topology - location data of each object
        self.topology = topology
        self.grid = None  # can be used to store grid later on
        self.grid_step_size = topology.flags['grid_size']
        # amount of each kind of object in the world
        self.num_agents = len(topology.agents)
        self.num_landmarks = len(topology.landmarks)
        self.num_walls = len(topology.walls)
        self.num_spawns = topology.flags['spawn_amount']  # consider only spawns in use
        # variables for path planning and target assignment
        self.targets = None
        self.global_visited = 0
        self.new_location_flags = None
        self.distances = None
        self.invalid_edges = None
        self.curved_paths = {}
        # Get maximum distance found in world (bottom left to top right)
        self.max_dist = distance(self.topology.flags['size'][0, :], topology.flags['size'][1, :])
        self.inf_dist = self.max_dist * infeasible_distance_factor  # cost value for representing infeasible distances
        self.env_bounds = self.topology.flags['size']  # is later updated to represet min and max of world boundary

        self.landmark_color = np.array([0.25, 0.25, 0.25])
        self.spawn_color = np.array([0.75, 0.0, 0.0])
        self.wall_color = np.array([0.0, 0.0, 0.75])

        self.cluster_info = {}  # only used with landmark clusters (self.topology.flags['cluster_l'])
        self.precalc_time = .0

        self.view_range = 10  # used for mean field q

        if agent_info is None:
            # life in seconds, speed in m/s
            if self.topology.flags['ground_flag']:
                self.agent_info = {'life': 8*3600, 'speed': 1, 'vspeed': 0, 'range': float('inf')}
            else:
                self.agent_info = {'life': 25*60, 'speed': 15, 'vspeed': 4, 'range': 4*1e3}
        else:
            self.agent_info = agent_info
        self.agent_info['distlimit'] = (self.agent_info['life'] * self.agent_info['speed']) / self.topology.scale_factor

    def make_world(self):
        """Make initial world setup, based on provided topology."""
        # create world object
        # print('Initializing World...')
        world = World()
        # add boundarys
        # print('Initializing boundary...')
        world.boundary = Boundary(Polygon(self.topology.boundary))
        world.boundary.name = self.topology.names['boundary']
        world.boundary.hard = True if world.boundary.name.find('hard') != -1 else False  # currently not used
        self.env_bounds = np.asarray(world.boundary.polygon.bounds).reshape((2, 2))

        # add walls - also transforms non-convex shapes into convex ones -  needs to be done BEFORE landmarks/agents
        # print('Initializing obstacles...')
        world.walls = [Boundary(Polygon(wall).convex_hull) for wall in self.topology.walls]
        for i, wall in enumerate(world.walls):
            wall.name = self.topology.names['walls'][i]
            wall.index = i
            wall.hard = True if wall.name.find('hard') != -1 else False
            wall.color = self.wall_color
            wall.width = 2  # just for rendering purposes, no coordinate-based meaning

        # print('Initializing underlying grid...')
        self.make_grid(world, self.grid_step_size)  # make underlying grid

        # add rows for row-like field structure if row_structure is True
        if self.topology.flags['row_structure']:
            n_current_walls = len(world.walls)
            for r, row in enumerate(self.grid.get_rows()):
                w_id = r + n_current_walls
                world.walls.append(Boundary(Polygon(row)))
                world.walls[w_id].index = w_id
                world.walls[w_id].name = 'field_row_%d' % world.walls[w_id].index
                self.topology.names['walls'].append(world.walls[w_id].name)
                self.topology.walls.append(row)
                world.walls[w_id].hard = False  # field rows are now always non-hard
                world.walls[w_id].color = self.wall_color
                world.walls[w_id].width = 2  # just for rendering purposes, no coordinate-based meaning
                self.make_grid(world, self.grid_step_size)  # update grid

        # add grid-based landmarks if from_grid_l is True
        # print('Initializing landmarks...')
        landmark_grid = None
        if self.topology.flags['from_grid_l']:
            landmark_grid = Grid(self, world, self.topology.flags['grid_size_l'], self.topology.flags['grid_rotation'])
            self.num_landmarks = landmark_grid.n_nodes
            self.topology.flags['landmark_amount'] = self.num_landmarks
            self.topology.landmarks = [[] for _ in range(self.num_landmarks)]
            self.topology.names['landmarks'] = [[] for _ in range(self.num_landmarks)]
        # set up clusters for cluster-based landmark generation
        if self.topology.flags['cluster_l']:
            self.cluster_info['num_clusters'] = np.random.randint(3, 15)  # TODO get a better way of defining clusters
            clusters = [Landmark() for _ in range(self.cluster_info['num_clusters'])]
            for cluster in clusters:
                self.spawn_at_random_position(cluster, world, self.env_bounds)
            self.cluster_info['locs'] = [cluster.state.p_pos for cluster in clusters]
            self.cluster_info['radius'] = 1 / self.cluster_info['num_clusters']  # proportional to n_clusters

        # add landmarks
        world.landmarks = [Landmark() for _ in range(self.num_landmarks + self.num_spawns)]
        for i, landmark in enumerate(world.landmarks):
            landmark.index = i % self.num_landmarks  # spawn landmarks get new indexing
            landmark.collide = False
            landmark.movable = False
            landmark.state.p_vel = np.zeros(world.dim_p)
            landmark.size = landmark.size * 0.3  # decrease size
            if i >= self.num_landmarks:  # currently adding agent spawns
                if self.topology.flags['random_spawn_a']:
                    self.spawn_at_random_position(landmark, world, self.env_bounds)
                    self.topology.spawns[landmark.index] = landmark.state.p_pos  # store for later respawn
                    landmark.name = 'random_spawn_%d' % landmark.index
                else:
                    landmark.state.p_pos = np.array(self.topology.spawns[landmark.index])
                    landmark.name = self.topology.names['spawns'][landmark.index]
                landmark.color = self.spawn_color
            else:  # non agent-spawns
                if self.topology.flags['from_grid_l']:
                    landmark.state.p_pos = list(landmark_grid.nodes.values())[landmark.index].real_pos
                    landmark.name = 'grid_landmark_%d' % landmark.index
                    self.topology.landmarks[i] = landmark.state.p_pos
                    self.topology.names['landmarks'][i] = landmark.name
                elif len(self.topology.landmarks[i]) == 0 or self.topology.flags['random_spawn_l']:
                    if self.topology.flags['cluster_l']:
                        self.spawn_entity_near_position(landmark,
                                                        self.cluster_info['locs'][i % self.cluster_info['num_clusters']],
                                                        world, self.cluster_info['radius'])
                    else:
                        self.spawn_at_random_position(landmark, world, self.env_bounds)
                    self.topology.landmarks[i] = landmark.state.p_pos  # store for later respawn - only used if moving?
                    landmark.name = self.topology.names['landmarks'][landmark.index]
                else:
                    landmark.state.p_pos = np.array(self.topology.landmarks[landmark.index])  # get pos from topology
                    landmark.name = self.topology.names['landmarks'][landmark.index]
                landmark.color = self.landmark_color

        # add agents
        # print('Initializing agents...')
        self.set_agents(world, self.num_agents)

        # make initial conditions
        # print('Initializing lookup tables...')
        init_complete = False
        while not init_complete:
            timer = time()
            self.update_distance_lut(world)
            self.update_nonstraight_edges(world)
            self.precalc_time = time() - timer
            init_complete = self.check_landmark_reachability(world)
        self.reset_world(world)
        return world

    def reset_world(self, world):
        """
        Resets world to initial conditions and stored starting locations.
        Also called by env.reset() and in constructor. Necessary after updating distance LUTs and random resets.
        """
        for i, landmark in enumerate(world.landmarks):
            landmark.visited = False
            if i >= self.num_landmarks:
                continue  # now on spawns, not landmarks - they do not need further resetting.
            landmark.color = self.landmark_color
            # landmark.state.p_pos = self.topology.landmarks[i]  # only necessary in case of moving landmarks

        self.set_targets(world, target_init='kmeans')
        self.global_visited = 0
        self.new_location_flags = [False for _ in range(self.num_agents)]

        for agent in world.agents:
            self.respawn_agent(agent, world)
        world.agent_paths = []

    def reward(self, agent, world):
        """
        Returns the reward at this timestep for this specific agent.
        Updates whether points have been visited, and the global amount of visited points.
        """
        reward = 0

        target = agent.target_history[-1]
        if len(agent.target_history) >= 2:
            agent_loc = agent.target_history[-2]  # agent has already moved when getting reward, get previous location
        else:
            agent_loc = agent.target_history[-1]

        if target == agent_loc:
            reward += -0.1  # penalty for moving to the same point
        else:
            reward += -0.01 * self.distances[target][agent_loc]  # distance penalty

        if self.new_location_flags[agent.index]:
            reward += 1
            self.new_location_flags[agent.index] = False

        # if self.global_visited >= self.num_landmarks:
        #     reward += 5  # big reward when finishing
        return reward

    def observation(self, agent, world):
        """Returns the observation for this agent. Used mainly for reinforcement learning."""
        closest_idxs, dists = self.get_closest_indices(world, agent.target_history[-1], self.view_range)

        goal_dists = [dist for dist in dists]
        # visited = [world.landmarks[idx].visited for idx in closest_idxs]  # not the most efficient; called for each agent but returns the same each time
        other_agents = [self.get_agent_location_indices(world)[idx] for idx in closest_idxs]  # not the most efficient; called for each agent but returns the same each time
        obs = np.concatenate((goal_dists, other_agents)).astype(np.float32)  # make nx3 observation matrix
        return obs.reshape(1, obs.shape[0])

    def check_done(self, agent, world):
        """Returns True if an episode has been completed for an agent. Currently checks if all landmarks are visited"""
        if self.global_visited >= self.num_landmarks + self.num_spawns:
            for agent in world.agents:
                if agent.target_history[-1] != self.num_landmarks:
                    return False  # not all agents returned home yet
            return True
        # if not self.inside_boundary(agent, world):
        #     return True  # exited testing area
        return False

    def set_agents(self, world, num_agents, just_planning_override=None):
        """"Adds num_agents agents to the world. Needs a call to scenario.reset_world() afterwards..."""
        self.num_agents = num_agents
        self.topology.flags['agent_amount'] = self.num_agents
        self.topology.names['agents'] = ['agent_%d' % i for i in range(self.num_agents)]

        if just_planning_override is not None:
            self.topology.flags['just_planning'] = just_planning_override

        world.agents = [Agent() for _ in range(self.num_agents)]
        for i, agent, in enumerate(world.agents):
            agent.name = self.topology.names['agents'][i]
            agent.index = i
            agent.collide = not self.topology.flags['just_planning']
            agent.silent = True
            agent.ghost = not self.topology.flags['ground_flag']
            agent.color = np.random.random(3)  # random color for each agent
            agent.size = agent.size * 0.6
            agent.spawn = i % self.num_spawns  # get correct spawn index

    def reset_random_landmarks(self, world):
        """Resets all landmark locations randomly. Agent spawns not included. Updates distance LUT as well."""
        for landmark in world.landmarks[0:self.num_landmarks]:
            self.spawn_at_random_position(landmark, world, self.env_bounds)
            self.topology.landmarks[landmark.index] = landmark.state.p_pos  # store new location
        self.update_distance_lut(world)

    def reset_random_agent_spawns(self, world):
        """Randomly reset the agent spawn(s)."""
        for landmark in world.landmarks[self.num_landmarks:]:
            self.spawn_at_random_position(landmark, world, self.env_bounds)
            self.topology.spawns[landmark.index] = landmark.state.p_pos
        self.update_distance_lut(world)

    def respawn_agent(self, agent, world):
        """Spawns the agent at or near its spawn location, depending on the settings."""
        if not self.topology.flags['just_planning'] and self.num_agents > self.num_spawns:
            self.spawn_entity_near_position(agent, self.topology.spawns[agent.spawn], world)
        else:
            agent.state.p_pos = self.topology.spawns[agent.spawn]  # collision should be turned off!
        agent.state.p_vel = np.zeros(world.dim_p)
        agent.state.c = np.zeros(world.dim_c)
        agent.target_iterator = 0
        agent.target_history = [agent.spawn + self.num_landmarks]  # Spawn location marked as target history
        agent.state.target = self.targets[agent.index][agent.target_iterator]

    def move_to_target(self, agent, target, world):
        """Move agent to target location (landmark index) and update target history. Does not update environment."""
        target_landmark = world.landmarks[target]
        agent.state.p_pos = target_landmark.state.p_pos
        agent.target_history.append(target)
        if not target_landmark.visited:
            self.new_location_flags[agent.index] = True
            target_landmark.visited = True
            target_landmark.color = agent.color
            self.global_visited += 1

    def update_distance_lut(self, world):
        """
        Updates the distance look-up table, with diagonal_value as the diagonal.
        Also updates the invalid_edges array, which contain invalid paths between points.
        Invalid edges are currently given the value of the world max_distance * 1000 (self.inf_dist)
        """
        # Does not take into account asymmetric distance table, although this could be a possibility!
        tab_size = self.num_landmarks + self.num_spawns
        self.distances = np.zeros((tab_size, tab_size))
        inv_edges_idx = [[], []]
        locations = world.landmarks
        for i in range(tab_size):
            for j in range(tab_size):
                if i < j:
                    if self.valid_edge(locations[i], locations[j], world):  # Check if edge is a valid one
                        self.distances[i, j] = distance(locations[i], locations[j])
                    else:  # invalid edge
                        inv_edges_idx[0].extend([i, j])  # add mirrored indices as well
                        inv_edges_idx[1].extend([j, i])
        self.distances += self.distances.T  # mirror across diagonal to avoid calculating everything twice
        self.set_distance_lut_diagonal(0)  # default diagonal value is zero, can be changed (set_distance_lut_diagonal)
        self.invalid_edges = (np.array(inv_edges_idx[0]), np.array(inv_edges_idx[1]))
        self.set_distance_lut_invalid_values(float('inf'))  # can be changed using set_distance_lut_invalid_values()

    def set_distance_lut_diagonal(self, diagonal_value):
        """Sets the diagonal value of the distances lookup-table."""
        diagonal = np.diag_indices_from(self.distances)
        self.distances[diagonal] = diagonal_value

    def set_distance_lut_invalid_values(self, invalid_value):
        """Sets the value for invalid edges in the distances lookup-table. Requires self.invalid_edges to be filled."""
        if len(self.invalid_edges[0]) == len(self.invalid_edges[1]) and len(self.invalid_edges[0]) > 0:
            self.distances[self.invalid_edges] = invalid_value

    def get_visited(self, world):
        """Returns two boolean visited arrays (visited=True): one for the landmarks, one for the agent spawns."""
        visited = [landmark.visited for landmark in world.landmarks]
        return visited[0:self.num_landmarks], visited[-self.num_spawns:]

    def get_agent_location_indices(self, world):
        """Returns a boolean array with a flag for each world location whether an agent is present there."""
        agents_present = [False for _ in range(self.num_landmarks + self.num_spawns)]
        locs = self.get_agent_locations(world)
        for loc in locs:
            agents_present[loc] = True
        return agents_present

    @staticmethod
    def get_agent_locations(world):
        """Returns landmark indices for each agent"""
        return [agent.target_history[-1] for agent in world.agents]

    @staticmethod
    def get_locations(world):
        """Returns environment coordinates for all landmarks and spawns, indexed as in world.landmarks."""
        return [loc.state.p_pos for loc in world.landmarks]

    @staticmethod
    def inside_boundary(entity, world):
        """Returns True if an entity (agent or landmark) is currently inside the world boundary."""
        entity_pol = Polygon(entity.corners)
        return world.boundary.polygon.contains(entity_pol)

    @staticmethod
    def inside_walls(entity, world):
        """Returns True if an entity (agent or landmark) is currently (partially) inside one of the obstacles."""
        entity_pol = Polygon(entity.corners)
        for wall in world.walls:
            if entity_pol.intersects(wall.polygon):  # use intersect instead of contains for walls.
                return True
        return False

    def valid_position(self, entity, world, ignore_grid=False):
        """Returns True if this entity's position is legal - inside the boundary and not inside obstacles."""
        gridifyable = True
        if not ignore_grid:
            gridifyable = len(self.grid.nearest_grid_point(entity, world, self)) > 0
        return Scenario.inside_boundary(entity, world) and not Scenario.inside_walls(entity, world) and gridifyable

    def valid_edge(self, loc, dest, world):
        """
        Returns true if there is an uninterrupted straight route an agent can take from loc to dest.
        Assumes all agents are the same type (either ground or aerial).
        """
        location = loc.state.p_pos if isinstance(loc, Entity) else np.array(loc)
        destination = dest.state.p_pos if isinstance(dest, Entity) else np.array(dest)
        edge = LineString([location, destination])
        for wall in world.walls:
            if not wall.hard and not self.topology.flags['ground_flag']:  # match flag
                continue  # edge valid, agents can pass through/over.
            if edge.intersects(wall.polygon):
                return False  # invalid edge
        if edge.crosses(world.boundary.polygon):
            return False  # invalid edge
        return True

    def spawn_at_random_position(self, entity, world, b):
        """Spawns an entity at a random valid position in the world."""
        offset = np.array((b[0]))
        scaling = np.array((b[1][0] - b[0][0], b[1][1] - b[0][1]))
        entity.state.p_pos = np.random.random(2) * scaling + offset
        if not self.valid_position(entity, world):
            self.spawn_at_random_position(entity, world, b)

    def spawn_entity_near_position(self, entity, position, world, radius=None):
        """Spawns an entity randomly somewhere near the provided position. The new position is a valid position."""
        if radius is None:
            radius = entity.size*len(world.agents)/4
        r1, r2 = np.random.rand(), np.random.rand()
        d = radius * np.sqrt(r1)
        theta = 2*np.pi * r2
        x, y = d*np.cos(theta), d*np.sin(theta)
        entity.state.p_pos = np.array([position[0] + x, position[1] + y])
        if not self.valid_position(entity, world):
            self.spawn_entity_near_position(entity, position, world)

    @staticmethod  # TODO check if this method is useful. It is probably inefficient (based on agent.colliding).
    def check_collide(agent, entity):
        """Returns True if an agent is colliding with another entity or with an object/wall."""
        dist = distance(agent, entity)
        if agent.colliding:
            return True
        if dist < agent.size + entity.size:
            return True
        return False

    def get_targets_kmeans(self, world):
        """Create a k-means cluster for each agent, and return the landmarks in this cluster."""
        # Agent nest is not necessarily close to cluster - when using multiple spawns this could be improved
        locs = self.get_locations(world)
        kmeans = KMeans(n_clusters=self.num_agents, init='k-means++')
        kmeans.fit(np.asarray(locs))
        pred = kmeans.predict(np.asarray(locs))
        targets = [[] for _ in range(self.num_agents)]
        for i, prediction in enumerate(pred):
            targets[prediction].append(i)
        return targets

    # sets scenario targets to a certain value, one list of targets per agent.
    def set_targets(self, world, targets=None, target_init=''):
        """Sets scenario targets to the targets value, or to a preset value (currently only kmeans supported)"""
        if target_init == 'kmeans':
            self.targets = self.get_targets_kmeans(world)
        else:
            self.targets = targets
        for i, agent, in enumerate(world.agents):
            agent.target_iterator = 0
            agent.state.target = self.targets[i][agent.target_iterator]

    def get_next_target(self, agent):
        """Get next target off agent target list - returns True if the target list has been completed."""
        if agent.target_iterator + 1 >= len(self.targets[agent.index]):
            return True  # end of target list reached
        agent.target_iterator += 1
        agent.state.target = self.targets[agent.index][agent.target_iterator]
        return False

    def get_closest_indices(self, world, location, n):
        """Gets n closest unvisited points. If less than n points are available, append home loc."""
        all_dists = self.distances[location]
        visited = self.get_visited(world)[0]
        idxs = np.argsort(all_dists)[1:]
        target_idxs = []
        for idx in idxs:
            if idx == self.num_landmarks:
                continue  # skip home location for now
            if not visited[idx] and all_dists[idx] < self.inf_dist:
                target_idxs.append(idx)
            if len(target_idxs) == n:
                break
        while len(target_idxs) < n:
            target_idxs.append(self.num_landmarks)  # append home location at end
        return target_idxs, [all_dists[idx] for idx in target_idxs]

    def get_closest_agents(self, agent_index, agent_locations, n=None):
        """Returns the indexes of the locations of the n closest agents to the agent with agent_index."""
        if n is None:
            n = self.view_range
        if n > self.num_agents:
            n = self.num_agents

        cur_loc = agent_locations[agent_index]
        all_dists = self.distances[cur_loc][agent_locations]
        idxs = np.argsort(all_dists)[1:n + 1]
        return idxs

    def make_grid(self, world, step_size):
        """Constructs a grid of size step_size in the world, based on world boundaries, with only valid points."""
        self.grid = Grid(self, world, step_size, self.topology.flags['grid_rotation'])

    def update_nonstraight_edges(self, world):
        """Constructs a list of workaround paths to avoid obstacles. Same layout as the distance LUT."""
        tab_size = self.num_landmarks + self.num_spawns
        self.curved_paths = {}
        for i in range(tab_size):
            for j in range(tab_size):
                if i < j:
                    if self.distances[i][j] >= self.inf_dist:  # if self.distances[i][j] > self.max_dist:
                        new_path, length = self.find_workaround(world.landmarks[i], world.landmarks[j], world)
                        if len(new_path) > 0:  # A* workaround found
                            self.curved_paths[i, j] = new_path
                            self.distances[i][j] = length
                            self.distances[j][i] = length
                        else:  # A* failed to find a path
                            self.distances[i][j] = self.inf_dist
                            self.distances[j][i] = self.inf_dist

    def workaround_getter(self, i, j):
        """Function that checks whether there is a workaround path between two points and returns that path."""
        res = self.curved_paths.get((i, j), [])
        if len(res) == 0:
            res = self.curved_paths.get((j, i), [])
            if len(res) > 0:
                res = res[::-1]  # get reverse path
        return res

    def find_workaround(self, loc, dest, world):
        """Perform A* search on the underlying grid to find a workaround path around obstacles."""
        # first move to closest valid grid point for both loc and dest - translate
        eq_loc = self.grid.nearest_grid_point(loc, world, self)
        eq_dest = self.grid.nearest_grid_point(dest, world, self)
        # assert len(eq_loc) == 2 and len(eq_dest) == 2, 'Invalid point for path search'
        if len(eq_loc) != 2 or len(eq_dest) != 2:
            return [], self.inf_dist
        # perform a* search to navigate to point
        node_path_idxs, path_length = self.grid.shortest_path(eq_loc, eq_dest, self.max_dist)
        length = distance(self.grid.node_at_index(*eq_loc).real_pos, loc) + \
            distance(self.grid.node_at_index(*eq_dest).real_pos, dest) + path_length

        path = []
        for idx in node_path_idxs:
            path.append(self.grid.node_at_index(*idx).real_pos)
        return path, length

    def switch_to_aerial(self, world):
        """Switches agents from ground-based to aerial. Needs a call to scenario.reset_world() afterwards..."""
        self.topology.flags['ground_flag'] = False
        for agent in world.agents:
            agent.ghost = True
        for i in range(len(self.distances)):
            for j in range(len(self.distances)):
                if i < j:
                    if self.valid_edge(world.landmarks[i], world.landmarks[j], world):
                        self.distances[i][j] = distance(world.landmarks[i], world.landmarks[j])
                        self.distances[j][i] = distance(world.landmarks[i], world.landmarks[j])
                        if len(self.workaround_getter(i, j)) > 0:
                            del self.curved_paths[i, j]

    def get_target_coordinate_lists(self, world, targets=None):
        """Assumes targets does not contain any empty target lists!"""
        if targets is None:
            targets = self.targets
        t_lengths = [len(target_list) for target_list in targets]

        target_coordinates = [[world.landmarks[target_list[0]].state.p_pos] for target_list in targets]
        for i in range(max(t_lengths)):
            for j in range(len(targets)):
                if 0 < i < len(targets[j]):
                    for loc in self.workaround_getter(targets[j][i-1], targets[j][i]):
                        target_coordinates[j].append(loc)
                    target_coordinates[j].append(world.landmarks[targets[j][i]].state.p_pos)
        return target_coordinates

    def check_landmark_reachability(self, world, tolerance=0.25, random_respawn_attempts=10):
        """
        Checks if all spawned landmarks have a valid, reachable location according to the distance LUT.
        A location is considered valid/reachable if at least (1 - tolerance) of its edges towards other nodes are valid.
        If landmarks are spawned from a grid, invalid ones are simply removed.
        Otherwise, they are randomly respawned until valid, or until random_respawn_attempts attempts have passed.
        """
        nsum = np.sum(self.distances, axis=0)
        idxs = np.where(nsum/len(nsum) > self.inf_dist * tolerance)[0]  # find index w/at least 25% of the edges invalid
        if len(idxs) == 0:
            return True  # no 'unreachable' landmarks found
        # print('%d of %d landmarks are unreachable. Removing/replacing invalid landmarks...' % (len(idxs), len(nsum)))
        for idx in idxs[::-1]:
            if self.topology.flags['from_grid_l'] and idx < self.num_landmarks:
                self.remove_landmark(world, idx)  # remove (non-spawn) landmark from grid
            else:  # try random respawn until proper location found
                s = len(nsum)
                attempts = random_respawn_attempts  # number of times to attempt random respawn
                while s > len(nsum) * tolerance and attempts != 0:
                    s = 0
                    attempts -= 1
                    self.spawn_at_random_position(world.landmarks[idx], world, self.env_bounds)
                    for i in range(len(nsum)):
                        if i != idx:
                            s += self.valid_edge(world.landmarks[idx], world.landmarks[i], world)
        return False

    def remove_landmark(self, world, idx):
        """Remove a landmark from the world and scenario."""
        self.num_landmarks -= 1
        self.topology.flags['landmark_amount'] -= 1
        del world.landmarks[idx]

    def store_environment(self, world, file_path):
        """Save the current scenario and world information to a .dictionary file."""
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path + '_scenario.dictionary', 'wb') as scenario_dict:
            pickle.dump(self.__dict__, scenario_dict)
        with open(file_path + '_world.dictionary', 'wb') as world_dict:
            pickle.dump(world.__dict__, world_dict)

    def load_environment(self, world, file_path):
        """Load a scenario and world object from a .dictionary file."""
        with open(file_path + '_scenario.dictionary', 'rb') as scenario_dict:
            self.__dict__ = pickle.load(scenario_dict)
        with open(file_path + '_world.dictionary', 'rb') as world_dict:
            world.__dict__ = pickle.load(world_dict)


