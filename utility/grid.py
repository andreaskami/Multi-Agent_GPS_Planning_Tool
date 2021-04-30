from multiagent.core import Landmark, Entity  # for valid position check
from utility.misc import distance
from utility.path_search import a_star_search

import numpy as np
import heapq as hq


class Grid:
    """Grid representation class. Subclass in Scenario."""
    # TODO re-evaluate A* search; it might take too long for large no of landmarks? Maybe repeat paths or path sections?
    def __init__(self, scenario, world, step_size, rotation=0):
        self.bounds = scenario.env_bounds if rotation == 0 else scenario.topology.flags['size']
        self.step_size = step_size
        self.rotation = rotation
        self.origin = np.zeros(2)
        self.n_nodes = 0
        self.index_map = []  # NOTE: index map indices are swapped; y first, x second
        self.nodes = {}  # maps (integer) indices to nodes
        self.distances = {}  # maps node pairs (their hashes) to distances
        self.valid_idx_bounds = None

        self.fill_grid(world, scenario)
        # self.calc_distances()  # currently not creating distance lookup dict: WAY too memory-intensive
        self.set_node_neighbours(world, scenario)

        # offset is not used
        self.offset = np.array((self.bounds[0][0], self.bounds[1][1]))  # (0, 0) corresponds to (minx, maxy)

    def node_at_index(self, x, y):
        """Gets a Node at an index location. Returns None if no node is found at that index."""
        return self.nodes.get((x, y), None)

    def node_distance(self, loc, dest):
        """Get distance between two Node objects. Returns -1 for invalid distances."""
        # Currently self.distances is not initialized, so this will always return 0
        return self.distances.get((loc.hash, dest.hash), -1)

    def fill_grid(self, world, scenario):
        """Fill the environment with valid grid nodes."""
        x_coor = np.arange(self.bounds[0][0], self.bounds[1][0], self.step_size)
        y_coor = np.arange(self.bounds[0][1], self.bounds[1][1], self.step_size)
        points = [[] for _ in range(len(x_coor))]
        for x, real_x in enumerate(x_coor):
            for y, real_y in enumerate(y_coor):
                points[x].append(self._rotate_point(real_x, real_y, self.rotation))

        node_idx = 0
        locations = []
        self.index_map = [[0 for _ in range(len(x_coor))] for _ in range(len(y_coor))]
        for x, row in enumerate(points):
            for y_reversed, real_pos in enumerate(points[x]):
                y = len(points[x]) - y_reversed - 1
                point = Landmark()  # created for valid position check, afterwards immediately discarded
                point.state.p_pos = real_pos
                if scenario.valid_position(point, world, ignore_grid=True):
                    locations.append((x, y))
                    self.index_map[y][x] = 1
                    self.nodes[(x, y)] = Node(x, y, real_pos, node_idx)
                    node_idx += 1
        self.n_nodes = node_idx
        locations = np.asarray(locations)
        self.valid_idx_bounds = np.concatenate((np.min(locations, axis=0), np.max(locations, axis=0)))

    def calc_distances(self):
        """Calculate the distance between each node in the grid. Stored in the distances hash table."""
        for loc in self.nodes.values():
            for dest in self.nodes.values():
                if loc == dest:
                    self.distances[(loc.hash, dest.hash)] = 0
                self.distances[(loc.hash, dest.hash)] = distance(loc.real_pos, dest.real_pos)

    def set_node_neighbours(self, world, scenario):
        """Sets the neighbours of all grid nodes."""
        for node in self.nodes.values():
            node.get_valid_neighbours(self, world, scenario)

    def print_to_text(self):
        """Print grid indices to output. Only used for debugging purposes."""
        for y in self.index_map:
            print(y)

    def nearest_grid_point(self, location, world, scenario):
        """Finds the nearest valid grid location to a continuous point location."""
        coor = location.state.p_pos if isinstance(location, Entity) else np.array(location)
        dists = []
        for node in self.nodes.values():
            if scenario.valid_edge(coor, node.real_pos, world):
                hq.heappush(dists, (distance(coor, node.real_pos), (node.x, node.y), node))
        if len(dists) == 0:
            return []
        return hq.heappop(dists)[1]

    def shortest_path(self, start_idx, goal_idx, search_distance):
        """Find shortest path using A* search."""
        start = self.node_at_index(*start_idx)
        goal = self.node_at_index(*goal_idx)

        assert start is not None and goal is not None, 'Start and/or goal node not found in grid (invalid pos)'
        # assert start != goal, 'Start and goal should be different nodes!'
        if start == goal:
            return [], 0

        self.soft_reset()  # reset nodes and parent nodes

        path, length = a_star_search(self, start, goal, search_distance)

        self.soft_reset()  # reset nodes and parent nodes - again, just for security
        # for idxs in path:
        #     self.index_map[idxs[1]][idxs[0]] = 2
        return path, length

    def soft_reset(self):
        """Resets index map and node search attributes."""
        for node in self.nodes.values():
            if str(self.index_map[node.y][node.x]) > '1':
                self.index_map[node.y][node.x] = 1
            node.f = float('inf')
            node.g = float('inf')
            node.parent = None

    def get_rows(self):
        """Create rows on the underlying grid."""
        all_rows = []
        min_valid_y, max_valid_y = self.valid_idx_bounds[1], self.valid_idx_bounds[3]

        for y in range(min_valid_y + 1, max_valid_y - 1, 3):
            rows = self._construct_rows(y)
            if len(rows) != 0:  # if row is empty, do not add it
                all_rows += rows
        return all_rows

    def _construct_rows(self, y):
        """Valid grid row creation function."""
        rows = []
        start_points, end_points = self._search_row(y)
        for s in range(len(start_points)):
            corners = [start_points[s][0], end_points[s][0], end_points[s][1], start_points[s][1]]
            row = [self.node_at_index(*corner).real_pos for corner in corners]
            rows.append(row)
        return rows

    def _search_row(self, y):
        """Find start points and end points for row to construct."""
        start_points, end_points = [], []
        y_bits = np.asarray([self.index_map[y], self.index_map[y + 1]])
        start_switch = True
        for x in range(0, len(y_bits[0]) - 2):
            if start_switch:  # search for [not 1, 1, 1], [not 1, 1, 1] and report indices of last search point
                if not (y_bits[0][x] and y_bits[1][x]) and (y_bits[0][x + 1] and y_bits[1][x + 1]) \
                        and (y_bits[0][x + 2] and y_bits[1][x + 2]):
                    start_points.append([(x + 2, y), (x + 2, y + 1)])
                    start_switch = False
            else:  # search for [1, 1, not 1], [1, 1, not 1] and report indices of first search point
                if (y_bits[0][x] and y_bits[1][x]) and (y_bits[0][x + 1] and y_bits[1][x + 1]) \
                        and not (y_bits[0][x + 2] and y_bits[1][x + 2]):
                    end_points.append([(x, y), (x, y + 1)])
                    start_switch = True
        if len(start_points) > len(end_points):  # there should be an equal number of start and end points
            start_points.pop(-1)
        for s in range(len(start_points) - 1, -1, -1):
            if start_points[s] == end_points[s]:  # if start point is also end point, the row can be discarded
                start_points.pop(s)
                end_points.pop(s)
        return start_points, end_points

    def _rotate_point(self, x, y, angle):
        """Rotates a point around the origin."""
        point = np.array((x, y))
        translated = point - self.origin
        new_x = translated[0]*np.cos(angle) - translated[1]*np.sin(angle)
        new_y = translated[0]*np.sin(angle) + translated[1]*np.cos(angle)
        return np.array((new_x, new_y)) + self.origin


class Node:
    """Grid Node representative class."""
    def __init__(self, x, y, real_pos, idx):
        self.x, self.y = x, y
        self.real_pos = real_pos
        self.hash = hash('%d_%d' % (self.x, self.y))
        self.idx = idx

        self.f = float('inf')
        self.g = float('inf')
        self.parent = None
        self.neighbours = []
        self.dists = []

    def __eq__(self, other):
        """Allows for comparing nodes."""
        if not isinstance(other, Node):
            raise NotImplementedError
        return self.hash == other.hash

    def get_valid_neighbours(self, grid, world, scenario):
        """Find this node's valid neighbours."""
        perm = [[-1, -1, 0, 1, 1, 1, 0, -1], [0, 1, 1, 1, 0, -1, -1, -1]]
        dists = [grid.step_size, np.sqrt(2)*grid.step_size]
        for i in range(8):
            neighbour = grid.node_at_index(self.x + perm[0][i], self.y + perm[1][i])
            if neighbour is None:
                continue
            if scenario.valid_edge(self.real_pos, neighbour.real_pos, world):
                self.dists.append(dists[i % len(dists)])
                self.neighbours.append((self.x + perm[0][i], self.y + perm[1][i]))
