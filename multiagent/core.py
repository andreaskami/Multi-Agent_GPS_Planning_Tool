import numpy as np
from copy import deepcopy


# physical/external base state of all entites
class EntityState(object):
    def __init__(self):
        # physical position
        self.p_pos = None
        # physical velocity
        self.p_vel = None


# state of agents (including communication and internal/mental state)
class AgentState(EntityState):
    def __init__(self):
        super(AgentState, self).__init__()
        # communication utterance
        self.c = None
        # index of target landmark
        self.target = None


# action of the agent
class Action(object):
    def __init__(self):
        # physical action
        self.u = None
        # communication action
        self.c = None


# properties and state of walls - CURRENTLY NOT USED
class Wall(object):
    def __init__(self, orientation='H', axis_pos=0.0, endpoints=(-1, 1), width=0.1, hard=True):
        self.orientation = orientation
        self.axis_pos = axis_pos
        self.endpoints = np.array(endpoints) if endpoints[0] < endpoints[1] else np.flip(np.array(endpoints))
        self.width = width
        self.hard = hard
        self.color = np.array([0.0, 0.0, 0.0])
        # tuple to store associated corner with this wall
        self.corner = (endpoints[0], axis_pos) if self.orientation == 'H' else (axis_pos, endpoints[0])


class Boundary(object):
    def __init__(self, polygon, width=0.2):
        self.polygon = polygon
        self.corners = polygon.exterior.coords
        self.center = polygon.centroid.coords[0]
        self.width = width
        self.hard = False
        self.color = np.array([0.0, 0.0, 0.0])
        self.index = 0
        self.name = ''


# properties and state of physical world entity
class Entity(object):
    def __init__(self):
        # name and index
        self.name = ''
        self.index = 0
        # properties:
        self.size = 0.03
        # entity can move / be pushed
        self.movable = False
        # entity collides with others
        self.collide = True
        self.ghost = False
        # material density (affects mass)
        self.density = 50.0
        # color
        self.color = None
        # max speed and accel
        self.max_speed = None
        self.accel = None
        # state
        self.state = EntityState()
        # mass
        self.initial_mass = 50.0

    @property
    def mass(self):
        return self.initial_mass

    @property
    def corners(self):
        if self.state.p_pos is None:
            return None
        # return octagon representation of entity for polygon collision detection and valid location detection
        return [np.array((np.cos(i * 0.25 * np.pi), np.sin(i * 0.25 * np.pi)))
                * self.size + self.state.p_pos for i in range(8)]


# properties of landmark entities
class Landmark(Entity):
    def __init__(self):
        super(Landmark, self).__init__()
        # keep track if landmark has been visited by an agent
        self.visited = False


# properties of agent entities
class Agent(Entity):
    def __init__(self):
        super(Agent, self).__init__()
        # agents are movable by default
        self.movable = True
        # cannot send communication signals
        self.silent = False
        # cannot observe the world
        self.blind = False
        # physical motor noise amount
        self.u_noise = None
        # communication noise amount
        self.c_noise = None
        # control range
        self.u_range = 1.0
        # state
        self.state = AgentState()
        # action
        self.action = Action()
        # script behavior to execute
        self.action_callback = None
        # currently colliding?
        self.colliding = False
        # target index iterator
        self.target_iterator = 0
        # target history tracker
        self.target_history = []
        # nest / spawn
        self.spawn = 0


# multi-agent world
class World(object):
    def __init__(self, collaborative=False):
        # list of agents and entities (can change at execution-time!)
        self.agents = []
        self.landmarks = []
        self.walls = []
        self.agent_paths = []
        # world boundary and walls grouped into obstacle
        self.boundary = None
        # communication channel dimensionality
        self.dim_c = 0
        # position dimensionality
        self.dim_p = 2
        # color dimensionality
        self.dim_color = 3
        # simulation timestep
        self.dt = 0.1
        # physical damping
        self.damping = 0.25  # 0.25
        # contact response parameters
        self.contact_force = 5e+2
        self.contact_force_wall = 1e+3
        self.contact_margin = 1e-3
        self.collaborative = collaborative

    # return all entities in the world
    @property
    def entities(self):
        return self.agents + self.landmarks

    # return all agents controllable by external policies
    @property
    def policy_agents(self):
        return [agent for agent in self.agents if agent.action_callback is None]

    # return all agents controlled by world scripts
    @property
    def scripted_agents(self):
        return [agent for agent in self.agents if agent.action_callback is not None]

    # return all agent positions
    @property
    def agent_positions(self):
        return [agent.state.p_pos for agent in self.agents]

    # update state of the world
    def step(self, rendering):
        # set actions for scripted agents 
        for agent in self.scripted_agents:
            agent.action = agent.action_callback(agent, self)
        # gather forces applied to entities
        p_force = [None] * len(self.agents)  # set to self.entities in case of movable/collidable landmarks
        # apply agent physical controls
        p_force = self.apply_action_force(p_force)
        # apply environment forces
        p_force = self.apply_environment_force(p_force)
        # integrate physical state
        self.integrate_state(p_force)
        # update agent state
        for agent in self.agents:
            self.update_agent_state(agent)

        if rendering:
            self.agent_paths.append(deepcopy(self.agent_positions))  # add to agent movement history for path rendering

    # gather agent action forces
    def apply_action_force(self, p_force):
        # set applied forces
        for i,agent in enumerate(self.agents):
            if agent.movable:
                noise = np.random.randn(*agent.action.u.shape) * agent.u_noise if agent.u_noise else 0.0
                p_force[i] = agent.action.u + noise
        return p_force

    # gather physical forces acting on entities
    def apply_environment_force(self, p_force):
        # simple (but inefficient) collision response
        for a, entity_a in enumerate(self.agents):  # set to self.entities in case of movable/collidable landmarks
            for b, entity_b in enumerate(self.agents):  # set to self.entities in case of movable/collidable landmarks
                if b <= a: continue
                [f_a, f_b] = self.get_collision_force(entity_a, entity_b)
                if f_a is not None:
                    if p_force[a] is None: p_force[a] = 0.0
                    p_force[a] = f_a + p_force[a]
                if f_b is not None:
                    if p_force[b] is None: p_force[b] = 0.0
                    p_force[b] = f_b + p_force[b]
            if entity_a.movable:
                for wall in self.walls:
                    wf = self.get_wall_collision_force(entity_a, wall)
                    if wf is not None:
                        if p_force[a] is None:
                            p_force[a] = 0.0
                        p_force[a] = p_force[a] + wf
        return p_force

    # integrate physical state
    def integrate_state(self, p_force):
        for i,entity in enumerate(self.agents):  # change back to self.entities in case of moving/collidable landmarks
            if not entity.movable: continue
            entity.state.p_vel = entity.state.p_vel * (1 - self.damping)
            if p_force[i] is not None:
                entity.state.p_vel += (p_force[i] / entity.mass) * self.dt
            if entity.max_speed is not None:
                speed = np.sqrt(np.square(entity.state.p_vel[0]) + np.square(entity.state.p_vel[1]))
                if speed > entity.max_speed:
                    entity.state.p_vel = entity.state.p_vel / np.sqrt(np.square(entity.state.p_vel[0]) +
                                                              np.square(entity.state.p_vel[1])) * entity.max_speed
            entity.state.p_pos += entity.state.p_vel * self.dt

    def update_agent_state(self, agent):
        # set communication state (directly for now)
        if agent.silent:
            agent.state.c = np.zeros(self.dim_c)
        else:
            noise = np.random.randn(*agent.action.c.shape) * agent.c_noise if agent.c_noise else 0.0
            agent.state.c = agent.action.c + noise

    # get collision forces for any contact between two entities
    def get_collision_force(self, entity_a, entity_b):
        if (not entity_a.collide) or (not entity_b.collide):
            return [None, None]  # not a collider
        if entity_a is entity_b:
            return [None, None]  # don't collide against itself
        # compute actual distance between entities
        delta_pos = entity_a.state.p_pos - entity_b.state.p_pos
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist = dist if dist != 0 else 1e-9  # catch divide by zero
        # minimum allowable distance
        dist_min = entity_a.size + entity_b.size
        # softmax penetration
        k = self.contact_margin
        penetration = np.logaddexp(0, -(dist - dist_min)/k)*k
        force = self.contact_force * delta_pos / dist * penetration
        force_a = +force if entity_a.movable else None
        force_b = -force if entity_b.movable else None
        entity_a.colliding = True if force[0] != 0.0 or force[1] != 0.0 else False
        entity_b.colliding = entity_a.colliding
        return [force_a, force_b]

    def get_wall_collision_force(self, agent, boundary):
        overlap = np.inf
        for i in range(len(boundary.corners)):
            j = (i + 1) % len(boundary.corners)
            c1, c2 = np.array(boundary.corners[i]), np.array(boundary.corners[j])
            axis_proj = (c2 - c1)[::-1] * np.array((-1, 1))
            min_r1, max_r1 = np.inf, -np.inf
            for corner in boundary.corners:
                q = np.dot(corner, axis_proj)
                min_r1 = min(min_r1, q)
                max_r1 = max(max_r1, q)
            min_r2, max_r2 = np.inf, -np.inf
            for corner in agent.corners:
                q = np.dot(corner, axis_proj)
                min_r2 = min(min_r2, q)
                max_r2 = max(max_r2, q)
            if not (max_r2 >= min_r1 and max_r1 >= min_r2):
                agent.colliding = False  # no collision
                return None
            overlap = min(min(max_r1, max_r2) - max(min_r1, min_r2), overlap)
        agent.colliding = True
        if not (agent.ghost and not boundary.hard):
            d = agent.state.p_pos - np.array(boundary.center)
            k = self.contact_margin * 100  # higher contact margin for walls
            f = self.contact_force
            penetration = np.logaddexp(0, -overlap / k) * k
            force = f * d / np.sqrt(np.dot(d, d)) * penetration
            return force
        return None
