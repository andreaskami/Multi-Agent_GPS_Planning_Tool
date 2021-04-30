import heapq as hq
from utility.misc import distance


def a_star_search(grid, start, goal, max_dist):
    """
    Performs A* search on a grid, from start to goal.
    start and goal should be of the Node object type (see utility/grid.py)
    If start is more than max_dist away from the goal, the path is not considered and an empty route is returned.
    """
    open_nodes = []  # list of open nodes with f_costs - is a priority queue implemented with heapq
    open_nodes_searchmap = {}  # parallel dict to open_nodes for faster searching using a hashmap
    start.g = 0
    start.f = _heuristic(start, goal)
    if start.f > max_dist:
        return [], max_dist

    hq.heappush(open_nodes, (start.f, start.hash, start))
    open_nodes_searchmap[start.hash] = True

    while open_nodes:
        _, _, current_node = hq.heappop(open_nodes)
        open_nodes_searchmap[current_node.hash] = False

        if current_node == goal:
            return _construct_path(current_node)  # goal found

        neighbours = [grid.node_at_index(*n) for n in current_node.neighbours]
        for n, neighbour in enumerate(neighbours):
            temp_g = _heuristic(current_node, neighbour, neighbour_dist=True, neighbour_idx=n)
            if temp_g < neighbour.g:
                neighbour.parent = current_node
                neighbour.g = temp_g
                neighbour.f = _heuristic(neighbour, goal)
                if not open_nodes_searchmap.get(start.hash, False):
                    hq.heappush(open_nodes, (neighbour.f, neighbour.hash, neighbour))
                    open_nodes_searchmap[neighbour.hash] = True

    # print("A* search failed to find a path to the goal location.")
    return [], max_dist


def _heuristic(node, other_node, neighbour_dist=False, neighbour_idx=0):
    """Heuristic utility function for A* search."""
    if neighbour_dist:
        return node.g + node.dists[neighbour_idx]
    else:
        return node.g + distance(node.real_pos, other_node.real_pos)


def _construct_path(current_node):
    """Constructs the resulting path by retracing parent nodes. Returns a list of index pairs."""
    path = [(current_node.x, current_node.y)]
    length = current_node.g
    while current_node.parent is not None:
        current_node = current_node.parent
        path.insert(0, (current_node.x, current_node.y))
    return path, length
