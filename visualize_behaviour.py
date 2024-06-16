import pygame
import math
import numpy as np
import NN
import os


pygame.init()
WIDTH = 400
HEIGHT = 400
AGENT_SIZE = 10
VELOCITY_COLOR = (100, 100, 255)
VELOCITY_LENGTH = 100
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

def draw_agent(screen, xpos, ypos, xvel, yvel, color):
    direction_angle = math.atan2(yvel, xvel)
        
    front_point = (xpos + AGENT_SIZE * 2 * math.cos(direction_angle), ypos + AGENT_SIZE * 2 * math.sin(direction_angle))
    back_left = (xpos + AGENT_SIZE * math.cos(direction_angle + math.pi * 3/4), ypos + AGENT_SIZE * math.sin(direction_angle + math.pi * 3/4))
    back_right = (xpos + AGENT_SIZE * math.cos(direction_angle - math.pi * 3/4), ypos + AGENT_SIZE * math.sin(direction_angle-math.pi*3/4))
    
    pygame.draw.polygon(screen, color, [front_point, back_left, back_right])

def draw_velocity_vectors(screen, xpos, ypos, velocities):
    for vel in velocities:
        xvel, yvel = vel
        # Draw velocity vectors as lines
        end_pos = (xpos + xvel * VELOCITY_LENGTH, ypos + yvel * VELOCITY_LENGTH)
        pygame.draw.line(screen, VELOCITY_COLOR, (xpos, ypos), end_pos, 2)

def get_params(filename):
    params = np.load(filename, allow_pickle=True)

    return zip(params["weights"], params["biases"])

def get_velocities(allparams, input):
    velocities = []

    for params in allparams:
        theta, speed = NN.feed_forward(params, input)

        rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
        rotated_velocity = np.matmul(rotation_matrix, np.array([0.0, -1.0]))
        velocities.append(rotated_velocity)
    return velocities

def get_input(main, agents):
    tot_neighbor_pos = np.zeros(2)
    tot_neighbor_vel = np.zeros(2)

    for agent in agents:
        tot_neighbor_pos += np.array([agent[0], agent[1]])
        tot_neighbor_vel += np.array([agent[2], agent[3]])
    
    avg_neighbor_pos = tot_neighbor_pos/len(agents)
    avg_neighbor_vel = tot_neighbor_vel/len(agents)

    position_vec = avg_neighbor_pos - np.array([main[0], main[1]]) # Vector from the agent to the mean neighbor position
    velocity_vec = avg_neighbor_vel - np.array([main[2], main[3]]) # Difference in velocity between agent and neighbors

    position_vec /= 100
    velocity_vec /= 2

    inputs = np.concatenate((position_vec, velocity_vec))

    return inputs

def draw_input(screen, xpos, ypos, input):
    # Extract position and velocity vectors from the input
    position_vec = (input[0], input[1])
    velocity_vec = (input[2], input[3])

    # Scale vectors for visualization
    position_end_pos = (xpos + position_vec[0] * 100, ypos + position_vec[1] * 100)
    velocity_end_pos = (xpos + velocity_vec[0] * 100, ypos + velocity_vec[1] * 100)

    # Draw the vectors on the screen
    pygame.draw.line(screen, (255, 0, 0), (xpos, ypos), position_end_pos, 2)
    pygame.draw.line(screen, (0, 255, 0), (xpos, ypos), velocity_end_pos, 2)

paths = ['logsv2\official_run01\experiment0001\pop_info_steps500_pop25_gens50_cot=none_mr1e-02_ms1e-02_ts1.npz',
         'logsv2\official_run01\experiment0166\pop_info_steps500_pop25_gens50_cot=uniform_mr2e-01_ms2e-01_ts2.npz',
         'logsv2\official_run01\experiment0075\pop_info_steps500_pop25_gens50_cot=single_point_mr1e-01_ms1e-02_ts5.npz',
         'logsv2\official_run01\experiment0183\pop_info_steps500_pop25_gens50_cot=single_point_mr2e-01_ms4e-01_ts2.npz',
         'logsv2\official_run01\experiment0236\pop_info_steps500_pop25_gens50_cot=mean_mr4e-01_ms2e-01_ts5.npz']

if __name__ == "__main__":
    agent_pos = (200, 200, 0, -2)
    x0, y0, _, _ = agent_pos
    
    other_agents = [(150, 160, -1.96, -0.37), 
                    (240, 130, -0.97, 0.2), 
                    (180, 130, -1.5, -0.6),
                    (160, 220, -2, 0)]
    
    NN_input = get_input(agent_pos, other_agents)

    dir = 'logsv2\official_run01'

    for filepath in paths:    
        print(filepath)
        
        agents_params = get_params(filepath)
        velocities = get_velocities(agents_params, NN_input)

        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            screen.fill((255, 255, 255))
            
            xpos, ypos, xvel, yvel = agent_pos
            draw_agent(screen, xpos, ypos, xvel, yvel, (200, 0, 0))
            for ag in other_agents:
                draw_agent(screen, ag[0], ag[1], ag[2], ag[3], (0,0,0))

            draw_velocity_vectors(screen, xpos, ypos, velocities)
            #draw_input(screen, xpos, ypos, NN_input)  # Call the new draw_input function

            pygame.display.flip()
            clock.tick(60)

    pygame.quit()

