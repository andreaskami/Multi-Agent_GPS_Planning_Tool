"""
Downloaded from https://github.com/Retsediv/ChristofidesAlgorithm/blob/master/christofides.py
By * Andrew Zhuravchak - student of CS@UCU

Modficiations by Lior Sinai, August 2019:
1) Minimum weight matching using Joris van Rantwijk's mwmatching.py code. This is much more optimal than Zhuravchak's
   original nearest neighbour algorithm
2) Verification of results using matplotlib.pyplot
3) Choose short cut from the best short cut
4) Return arguments from tsp in a dictionary

"""

from algorithms.Christofides.mwmatching import maxWeightMatching
import random


def ChristofidesApprox(distance_matrix):
    # build a graph of for each vertex of distances to all other vertices
    num_points = len(distance_matrix)
    G = build_graph(distance_matrix)
    # print("Graph: ", G)

    # build a minimum spanning tree
    MSTree, MSTlength = minimum_spanning_tree(G)
    # print("MSTree: ", MSTree)
    MSTbranches = [[branch[0], branch[1]] for branch in MSTree]

    # find odd vertexes
    odd_vertices = find_odd_vertexes(MSTree)
    # print("Odd vertexes in MSTree: ", odd_vertexes)

    # add minimum weight matching edges to MST
    matching_cost = minimum_weight_matching_blossom(MSTree, G, odd_vertices)
    # print("Minimum weight matching tree: ", MSTree)
    matched_pairs = []
    for pair in range(num_points-1, len(MSTree)):
        matched_pairs.append(list(MSTree[pair][0:2]))
    # print("Matching cost for %d edges: %.4f"%(len(matchedPairs), matching_cost))

    # find an Eulerian tour
    eulerian_tour = find_eulerian_tour(MSTree, G)
    # print("Eulerian tour: ", eulerian_tour)

    path, length = shortcut_repeat_vertices(eulerian_tour, G, num_points)

    return {'path': path,
            'length': length,
            'graph': G,
            'minimum_spanning_tree': MSTbranches,
            'minimum_spanning_tree_length': MSTlength,
            'eulerian_tour': eulerian_tour,
            'odd_vertices': odd_vertices,
            'minimum_weight_matches': matched_pairs
            }


def get_length(x1, y1, x2, y2):
    return ((x1 - x2) ** 2 + (y1 - y2) ** 2) ** (1 / 2)


def build_graph(data):
    graph = {}
    for this in range(len(data)):
        for another_point in range(len(data)):
            if this != another_point:
                if this not in graph:
                    graph[this] = {}

                graph[this][another_point] = data[this][another_point]
                # get_length(data[this][0], data[this][1], data[another_point][0], data[another_point][1])

    return graph


class UnionFind:
    def __init__(self):
        self.weights = {}
        self.parents = {}

    def __getitem__(self, object):
        if object not in self.parents:
            self.parents[object] = object
            self.weights[object] = 1
            return object

        # find path of objects leading to the root
        path = [object]
        root = self.parents[object]
        while root != path[-1]:
            path.append(root)
            root = self.parents[root]

        # compress the path and return
        for ancestor in path:
            self.parents[ancestor] = root
        return root

    def __iter__(self):
        return iter(self.parents)

    def union(self, *objects):
        roots = [self[x] for x in objects]
        heaviest = max([(self.weights[r], r) for r in roots])[1]
        for r in roots:
            if r != heaviest:
                self.weights[heaviest] += self.weights[r]
                self.parents[r] = heaviest


def minimum_spanning_tree(G):
    """Kruskal's minimum spanning tree algorithm """
    tree = []
    total_weight = 0
    subtrees = UnionFind()
    for Weight, u, v in sorted((G[u][v], u, v) for u in G for v in G[u]):
        if subtrees[u] != subtrees[v]:
            tree.append((u, v, Weight))
            subtrees.union(u, v)
            total_weight += Weight

    return tree, total_weight


def find_odd_vertexes(MST):
    tmp_g = {}
    vertexes = []
    for edge in MST:
        if edge[0] not in tmp_g:
            tmp_g[edge[0]] = 0

        if edge[1] not in tmp_g:
            tmp_g[edge[1]] = 0

        tmp_g[edge[0]] += 1
        tmp_g[edge[1]] += 1

    for vertex in tmp_g:
        if tmp_g[vertex] % 2 == 1:
            vertexes.append(vertex)

    return vertexes


def minimum_weight_matching(MST, G, odd_vert):
    """ Zhuravchak's original algorithm: select an odd vertex, give it the closest neighbour, and then go to the
    next vertex. The last point will probably get a high cost, so this algorithm is not that good"""
    random.shuffle(odd_vert)

    weight = 0
    while odd_vert:
        v = odd_vert.pop()
        length = float("inf")
        closest = 0
        for u in odd_vert:
            if v != u and G[v][u] < length:
                length = G[v][u]
                closest = u

        MST.append((v, closest, length))
        odd_vert.remove(closest)
        weight += length
        # print(f'({v}, {closest}) -> {length}')

    return weight


def minimum_weight_matching_blossom(MST, G, vertices):
    """ Uses the blossom method for matching pairs. Runs in O(n**3) time, for n!/[2**(n/2)*(n/2)!] possible pair sets"""

    subgraph = []  # the sub-graph to be used in matching. Must include each pair only once.
    n = len(vertices)
    for i in range(n):
        for j in range(i + 1, n):
            u = vertices[i]
            v = vertices[j]
            subgraph.append(
                (i, j, -round(G[u][v])))  # Negative weight to turn maximising problem into a minimisation problem. Crashes if don't use round.
    # assert len(subgraph) == n*(n-1)/2  # number of edges to choose from
    # print("subgraph with unique edges of odd vertices:", subgraph)

    # apply weighting algorithm
    pairs_to_use = maxWeightMatching(subgraph, True)

    # recover pairs in the main graph
    weight = 0
    edge_used = []
    for i, j in enumerate(pairs_to_use):
        if j == -1:
            raise Warning("Unused odd vertex. Not possible in Christofides algorithm")
        if i not in edge_used and j not in edge_used:
            u = vertices[i]
            v = vertices[j]
            w = G[u][v]
            MST.append((u, v, w))
            weight += w
            edge_used.extend([i, j])

    return weight


def find_eulerian_tour(MatchedMSTree, G):
    # find neighbours
    neighbours = {}
    for edge in MatchedMSTree:
        if edge[0] not in neighbours:
            neighbours[edge[0]] = []

        if edge[1] not in neighbours:
            neighbours[edge[1]] = []

        neighbours[edge[0]].append(edge[1])
        neighbours[edge[1]].append(edge[0])

    # print("Neighbours: ", neighbours)

    # finds the Eulerian tour
    start_vertex = MatchedMSTree[0][0]
    EP = [neighbours[start_vertex][0]]

    while len(MatchedMSTree) > 0:
        for i, v in enumerate(EP):
            if len(neighbours[v]) > 0:
                break

        while len(neighbours[v]) > 0:
            w = neighbours[v][0]

            remove_edge_from_matchedMST(MatchedMSTree, v, w)

            del neighbours[v][(neighbours[v].index(w))]
            del neighbours[w][(neighbours[w].index(v))]

            i += 1
            EP.insert(i, w)

            v = w

    return EP


def remove_edge_from_matchedMST(MatchedMST, v1, v2):

    for i, item in enumerate(MatchedMST):
        if (item[0] == v2 and item[1] == v1) or (item[0] == v1 and item[1] == v2):
            del MatchedMST[i]

    return MatchedMST


def shortcut_repeat_vertices(tour, G, n: int):
    # get all repeated vertices first
    visited = [-1] * n
    repeats = {}  # dictionary of repeated values
    for i, v in enumerate(tour):
        if visited[v] == -1 or (i == 0 or i == len(tour) - 1):
            visited[v] = i
        else:
            if v in repeats:
                repeats[v] += [i]
            else:
                repeats[v] = [visited[v], i]

    # check which repeat indices are best to use
    keep = {tour[0]: len(tour) - 1}  # repeat vertices to keep. Always keep the end point
    for v, repeat_indices in repeats.items():
        repeat_indices = repeats[v]
        # keep[v] = repeat_indices[0] # keep the first index
        # continue
        if any(i == 0 for i in repeat_indices):
            continue
        elif any(tour[i - 1] in repeats or tour[i + 1] in repeats for i in repeat_indices):
            # there's a loop and interdependency, and it's difficult to check which set of indices is better.
            # Also the loop might need to be reversed if it turns away from the main branch
            keep[v] = repeat_indices[0]
            # print('%s interdependency loops'%(str(repeat_indices)))
        else:
            cost = float('inf')
            for i in repeat_indices:
                cost_to_keep = 0
                for j in repeat_indices:
                    if i == j:
                        # cost to keep is the cost through this vertex to the next vertex
                        cost_to_keep += G[tour[i - 1]][tour[i]] + G[tour[i]][tour[i + 1]]
                    else:
                        # short cut the vertex
                        cost_to_keep += G[tour[j - 1]][tour[j + 1]]
                if cost_to_keep < cost:
                    keep[v] = i
                    # print("%d Cost to keep vs change: %.4f" % (v, cost - cost_to_keep))
                    cost = cost_to_keep

    # add vertices until get to repeats, then check if it is better to add or not
    current = tour[0]
    path = [current]
    length = 0
    for i, v in enumerate(tour[1:]):
        if v not in keep:
            # not repeated, append normally
            path.append(v)
            length += G[current][v]
            current = v
        else:
            if keep[v] == (i + 1):  # we have chosen to keep this one. +1 since starting at tour[1:]
                path.append(v)
                length += G[current][v]
                current = v

    return path, length
