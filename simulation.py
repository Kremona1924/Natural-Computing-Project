from matplotlib import pyplot as plt
import pygame
import random
import math
import numpy as np
import time
import NN

# Define constants
WIDTH = 450
HEIGHT = 450
WALL_MARGIN = 50
NUM_AGENTS = 15
AGENT_SIZE = 5
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 0)
AGENT_SPEED = 2  # Adjust speed as needed

# Define Agent class
class Agent:
    def __init__(self, x, y, params):
        self.params = params
        self.xpos = x
        self.ypos = y
        # Generate random angle
        angle = random.uniform(0, 2*math.pi)
        # Calculate velocity components based on angle and speed
        self.xvel = AGENT_SPEED * math.cos(angle)
        self.yvel = AGENT_SPEED * math.sin(angle)
        
        # Boid hyperparameters
        self.neighbor_dist = 100  # Adjust neighbor distance as needed
        self.fov_angle = 100 # How far back it can look up to 180 degrees
        self.turnfactor = 0.2
        self.max_ang_vel = np.pi / 180

    def set_agents(self, agents):
        self.agents = agents

    def angle_between_agents(self, agent_pos):
        # Calculate vectors between the agents
        vec_agent1 = np.array([self.xvel, self.yvel])  # Velocity vector of agent 1
        vec_agent2 = np.array(agent_pos) - np.array([self.xpos, self.ypos])  # Vector from agent 1 to agent 2
        
        # If the agents are on top of each other, return 0
        if np.linalg.norm(vec_agent2) < AGENT_SIZE: 
            return 0

        # Calculate the angle between the two vectors
        dot_product = np.dot(vec_agent1, vec_agent2)
        norms = np.linalg.norm(vec_agent1) * np.linalg.norm(vec_agent2)

        angle_radians = np.arccos(np.clip(dot_product / norms, -1, 1)) # Clip between -1 and 1. Needed due to numerical inaccuracies

        # Convert angle from radians to degrees
        angle_degrees = np.degrees(angle_radians)
        
        return angle_degrees

    def get_neighbors(self, agents):
        neighbors = []
        for agent in agents:
            distance = math.sqrt((self.xpos - agent.xpos)**2 + (self.ypos - agent.ypos)**2)
            if distance < self.neighbor_dist:
                angle = self.angle_between_agents([agent.xpos, agent.ypos])
                if angle < self.fov_angle:
                    neighbors.append(agent)
        return neighbors 

    def move(self):
        neighbors = self.get_neighbors(self.agents)
        avg_neighbor_pos = np.zeros(2)
        avg_neighbor_vel = np.zeros(2)

        if len(neighbors) == 0:
            inputs = np.zeros(4)

        else:
            for neigbor in neighbors:
                avg_neighbor_pos += np.array([neigbor.xpos, neigbor.ypos])
                avg_neighbor_vel += np.array([neigbor.xvel, neigbor.yvel])
            
            avg_neighbor_pos = avg_neighbor_pos/len(neighbors)
            avg_neighbor_vel = avg_neighbor_vel/len(neighbors)

            positiion_vec = avg_neighbor_pos - np.array([self.xpos, self.ypos]) # Vector from the agent to the mean neighbor position
            velocity_vec = avg_neighbor_vel - np.array([self.xvel, self.yvel]) # Difference in velocity between agent and neighbors
        
            inputs = np.concatenate((positiion_vec, velocity_vec))

        angular_vel = NN.feed_forward(self.params, inputs) # Angular velocity of angent, as determined by the NN. Range = (-1, 1).
        theta = angular_vel * self.max_ang_vel

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_velocity = np.matmul(rotation_matrix, np.array([self.xvel, self.yvel]))

        self.xvel = rotated_velocity[0]
        self.yvel = rotated_velocity[1]

        # Update position
        self.xpos += self.xvel
        self.ypos += self.yvel

        if self.xpos > WIDTH:
            self.xpos -= WIDTH
        if self.xpos < 0:
            self.xpos += WIDTH
        if self.ypos > HEIGHT:
            self.ypos -= HEIGHT
        if self.ypos < 0:
            self.ypos += HEIGHT

    def draw(self, screen):
        direction_angle = math.atan2(self.yvel, self.xvel)
        
        # Punt van de driehoek in de richting van beweging
        front_point = (self.xpos + AGENT_SIZE * 2 * math.cos(direction_angle),
                    self.ypos + AGENT_SIZE * 2 * math.sin(direction_angle))
        
        # Achterpunten van de driehoek
        back_left = (self.xpos + AGENT_SIZE * math.cos(direction_angle + math.pi * 3/4),
                    self.ypos + AGENT_SIZE * math.sin(direction_angle + math.pi * 3/4))
        back_right = (self.xpos + AGENT_SIZE * math.cos(direction_angle - math.pi * 3/4),
                    self.ypos + AGENT_SIZE * math.sin(direction_angle-math.pi*3/4))
        
        pygame.draw.polygon(screen, AGENT_COLOR, [front_point, back_left, back_right])

class boids_sim:
    def __init__(self, pop_size, layer_sizes) -> None:
        #random.seed(1) # Ensure each sim starts the same
        self.pop_size = pop_size
        self.agents = np.array([Agent(random.randint(0, WIDTH), random.randint(0, HEIGHT), NN.initialise_network(layer_sizes)) for i in range(self.pop_size)])
        for i, agent in enumerate(self.agents):
            agent.set_agents(self.agents[np.arange(pop_size) != i])
       
    def run(self, steps):
        order = []
        for _ in range(steps):
            for agent in self.agents:
                agent.move()
            order.append(self.compute_order(self.agents))
        return order

    def compute_order(self, agents):
        vx = 0
        vy = 0
        for agent in agents:
            vel_magnitude = math.sqrt(agent.xvel**2 + agent.yvel**2)
            vx += agent.xvel/vel_magnitude
            vy += agent.yvel/vel_magnitude
        return math.sqrt(vx**2 + vy**2)/len(agents)


    def run_with_screen(self, steps):
        # Initialize pygame
        pygame.init()
        screen = pygame.display.set_mode((WIDTH, HEIGHT))
        pygame.display.set_caption("Simple Agent Simulation")
        clock = pygame.time.Clock()
        
        order = []
        for _ in range(steps):
            screen.fill(BG_COLOR)
            for agent in self.agents:
                agent.move()
                agent.draw(screen)

            order.append(self.compute_order(self.agents))

            # Event handling
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            pygame.display.flip()
            clock.tick(60)

        plt.plot(order)
        plt.show()
        pygame.quit()

# Uncomment to run with screen
sim = boids_sim(20, [4,1])
sim.run_with_screen(10000)
