from matplotlib import pyplot as plt
import pygame
import random
import math
import numpy as np
import time
import NN
import json
import os


# Define constants
WIDTH = 500
HEIGHT = 500
WALL_MARGIN = 50
NUM_AGENTS = 25
AGENT_SIZE = 5
BG_COLOR = (255, 255, 255)
AGENT_COLOR = (0, 0, 0)
AGENT_SPEED = 2  # Adjust speed as needed

def angle(v1, v2):
    # Calculate the angle between the two vectors
    dot_product = np.dot(v1, v2)
    norms = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle_radians = np.arccos(np.clip(dot_product / norms, -1, 1)) # Clip between -1 and 1. Needed due to numerical inaccuracies

    # Convert angle from radians to degrees
    angle_degrees = np.degrees(angle_radians)
    return angle_degrees

# Define Agent class
class Agent:
    def __init__(self, id, x, y, params):
        self.id = id
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
        self.max_ang_vel = np.pi / 60

    def angle_between_agents(self, agent_pos):
        # Calculate vectors between the agents
        vec_agent1 = np.array([self.xvel, self.yvel])  # Velocity vector of agent 1
        vec_agent2 = np.array(agent_pos) - np.array([self.xpos, self.ypos])  # Vector from agent 1 to agent 2
        
        # If the agents are on top of each other, return 0
        if np.linalg.norm(vec_agent2) < AGENT_SIZE: 
            return 0

        return angle(vec_agent1, vec_agent2)

    def get_neighbors(self, agents):
        neighbors = []
        for agent in agents:
            distance = math.sqrt((self.xpos - agent.xpos)**2 + (self.ypos - agent.ypos)**2)
            if distance < self.neighbor_dist:
                angle = self.angle_between_agents([agent.xpos, agent.ypos])
                if angle < self.fov_angle:
                    neighbors.append(agent)
        return neighbors 

    def move(self, neighbors):
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

            position_vec = avg_neighbor_pos - np.array([self.xpos, self.ypos]) # Vector from the agent to the mean neighbor position
            velocity_vec = avg_neighbor_vel - np.array([self.xvel, self.yvel]) # Difference in velocity between agent and neighbors

            position_vec /= self.neighbor_dist
            velocity_vec /= AGENT_SPEED
        
            inputs = np.concatenate((position_vec, velocity_vec))

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
    def __init__(self, pop) -> None:
        #random.seed(1) # Ensure each sim starts the same
        self.pop_size = len(pop)
        self.agents = np.array([Agent(agent_params["id"], random.uniform(0, WIDTH), random.uniform(0, HEIGHT), agent_params["params"]) for agent_params in pop])
        
    def log_state(self, agents, filename="simulation_log.json"):
        state_data = []
        for agent in agents:
            agent_data = {
                "id": agent.id,
                "xpos": agent.xpos,
                "ypos": agent.ypos,
                "xvel": agent.xvel,
                "yvel": agent.yvel
            }
            state_data.append(agent_data)
        
        with open(filename, "a") as file:
            json.dump(state_data, file)
            file.write("\n")  # new line each timestamp

    def evaluate_alignment(self, agent, neighbors):

        # If there are no neighbors, alignment is undefined, return 1 in that case (worst possible value)
        if len(neighbors) == 0:
            return 1.0
    
        neighbors_xvel = 0.0
        neighbors_yvel = 0.0 
        for n in neighbors:
            neighbors_xvel += n.xvel
            neighbors_yvel += n.yvel
        neighbors_xvel /= len(neighbors)
        neighbors_yvel /= len(neighbors)

        # Return alignment as the angle between the agent velocity and the mean neighbor velocity, scaled to [0,1], 0 being perfect alignment
        return angle((agent.xvel, agent.yvel), (neighbors_xvel, neighbors_yvel)) / 180.0
        
    def evaluate_cohesion(self):
        # Cohesion is now defined as the mean distance to all boids per agent
        # TODO: evaluate and tune this measure of cohesion
        agent_positions = np.array([[agent.xpos, agent.ypos] for agent in self.agents])
        agent_positions /= float(max(WIDTH, HEIGHT)) # Scale positions by screen size to fall in range [0,1]
        dist_matrix = self.compute_dist_matrix(agent_positions)
        cohesion = np.mean(np.sort(dist_matrix, axis=1)[:, 1:7], axis=1) # cohesion is the mean dist to the 5 closest boids (excluding itself)
        return cohesion
        
    def compute_dist_matrix(self, X):
        # Fast distance matrix calculation using numpy, based on https://jaykmody.com/blog/distance-matrices-with-numpy/
        x2 = np.sum(X**2, axis=1)
        xy = np.matmul(X, X.T)
        dist_sq = x2.reshape(-1, 1) - 2*xy + x2
        return np.sqrt(np.maximum(dist_sq, 0))

    def run_with_screen(self, steps, show_screen = True, plot_chart=False, log=False, filename="simulation_log.json"):
        # # This does not work well, it removes the data in the file from the previous generations
        # # For now remove the log manually
        # if log:
        #     open(filename, 'w').close()

        # Initialize pygame

        if show_screen:
            pygame.init()
            screen = pygame.display.set_mode((WIDTH, HEIGHT))
            pygame.display.set_caption("Simple Agent Simulation")
            clock = pygame.time.Clock()
        
        alignment_metric = np.zeros((steps, len(self.agents)))
        cohesion_metric = np.zeros((steps, len(self.agents)))

        for t in range(steps):

            if show_screen:
                screen.fill(BG_COLOR)
                for agent in self.agents:
                    agent.draw(screen)

                # Event handling
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        running = False

                pygame.display.flip()
                clock.tick(200)

            if log:
                self.log_state(self.agents, filename)
            
            cohesion_metric[t] = self.evaluate_cohesion()

            for i, agent in enumerate(self.agents):
                neighbors = agent.get_neighbors(self.agents[np.arange(self.pop_size) != i]) # Exclude the agent itself from the neighbors
                alignment_metric[t][i] = self.evaluate_alignment(agent, neighbors)
                agent.move(neighbors)
        
        if plot_chart:
            # TODO: add appropriate plots
            #plt.plot()
            #plt.show()
            pass
        
        if show_screen:
            pygame.quit()
        
        return alignment_metric.T, cohesion_metric.T

# Uncomment to run with screen
# sim = boids_sim(20, [4,1])
# sim.run_with_screen(10000)

# just some random text for me to understand github
