import pygame
import json
import math

'''
This file replays the logged simulation(s) from a json file. Either the last simulation or all simulations, dependent on
which logging option you used. 
'''

pygame.init()
WIDTH = 500
HEIGHT = 500
AGENT_SIZE = 5
AGENT_COLOR = (0, 0, 0)
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

def draw(screen, xpos, ypos, xvel, yvel):
        direction_angle = math.atan2(yvel, xvel)
        
        # Punt van de driehoek in de richting van beweging
        front_point = (xpos + AGENT_SIZE * 2 * math.cos(direction_angle),
                    ypos + AGENT_SIZE * 2 * math.sin(direction_angle))
        
        # Achterpunten van de driehoek
        back_left = (xpos + AGENT_SIZE * math.cos(direction_angle + math.pi * 3/4),
                    ypos + AGENT_SIZE * math.sin(direction_angle + math.pi * 3/4))
        back_right = (xpos + AGENT_SIZE * math.cos(direction_angle - math.pi * 3/4),
                    ypos + AGENT_SIZE * math.sin(direction_angle-math.pi*3/4))
        
        pygame.draw.polygon(screen, AGENT_COLOR, [front_point, back_left, back_right])

def replay_simulation(filename="simulation_log.json"):
    with open(filename, "r") as file:
        for line in file:
            state_data = json.loads(line)
            screen.fill((255, 255, 255))  # Wis het scherm

            for agent_data in state_data:
                draw(screen, agent_data["xpos"], agent_data["ypos"], agent_data["xvel"], agent_data["yvel"])

            pygame.display.flip()
            clock.tick(100)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return  # Stop de replay als het venster wordt gesloten
                
if __name__ == "__main__":
    # Path to the simulation you want to replay. It should be a json file that starts with 'all_states_..' or 'last_states_..'
    sim_name = r"logs/replay_example/last_states_steps500_pop25_gens50_cot=uniform_mr1e-01_ms2e-01_ts5.json" 
    replay_simulation(sim_name)  # Call the function to start the replay