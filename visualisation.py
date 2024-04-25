import numpy as np
import pygame
import NN

"""
This script lets you play around with the input to the neural network and see the output. Run the script and drag the pos and vel arrows to change
the input the NN gets. The red arrow indicates the output of the network.
The output is a value between [-1,1] which is represented by rotating the red arrow indicating the direction of the boid.
"""

def draw_arrow(
        surface: pygame.Surface,
        start: pygame.Vector2,
        end: pygame.Vector2,
        color: pygame.Color,
        body_width: int = 2,
        head_width: int = 4,
        head_height: int = 2,
    ):
    """Draw an arrow between start and end with the arrow head at the end.

    Args:
        surface (pygame.Surface): The surface to draw on
        start (pygame.Vector2): Start position
        end (pygame.Vector2): End position
        color (pygame.Color): Color of the arrow
        body_width (int, optional): Defaults to 2.
        head_width (int, optional): Defaults to 4.
        head_height (float, optional): Defaults to 2.
    """
    arrow = start - end
    angle = arrow.angle_to(pygame.Vector2(0, -1))
    body_length = arrow.length() - head_height

    # Create the triangle head around the origin
    head_verts = [
        pygame.Vector2(0, head_height / 2),  # Center
        pygame.Vector2(head_width / 2, -head_height / 2),  # Bottomright
        pygame.Vector2(-head_width / 2, -head_height / 2),  # Bottomleft
    ]
    # Rotate and translate the head into place
    translation = pygame.Vector2(0, arrow.length() - (head_height / 2)).rotate(-angle)
    for i in range(len(head_verts)):
        head_verts[i].rotate_ip(-angle)
        head_verts[i] += translation
        head_verts[i] += start

    pygame.draw.polygon(surface, color, head_verts)

    # Stop weird shapes when the arrow is shorter than arrow head
    if arrow.length() >= head_height:
        # Calculate the body rect, rotate and translate into place
        body_verts = [
            pygame.Vector2(-body_width / 2, body_length / 2),  # Topleft
            pygame.Vector2(body_width / 2, body_length / 2),  # Topright
            pygame.Vector2(body_width / 2, -body_length / 2),  # Bottomright
            pygame.Vector2(-body_width / 2, -body_length / 2),  # Bottomleft
        ]
        translation = pygame.Vector2(0, body_length / 2).rotate(-angle)
        for i in range(len(body_verts)):
            body_verts[i].rotate_ip(-angle)
            body_verts[i] += translation
            body_verts[i] += start

        pygame.draw.polygon(surface, color, body_verts)

def get_params(filename):
    return np.load(filename, allow_pickle=True)


params = get_params('params.npy')

if params is None:
    print('No params found in params.npy')
    exit()

pygame.init()
WIDTH = 1000
HEIGHT = 600
center = pygame.Vector2(WIDTH/2, HEIGHT/2)
win = pygame.display.set_mode((WIDTH, HEIGHT))

DRAGBOX_WIDTH = 50
pos_drag = pygame.rect.Rect(WIDTH/2, HEIGHT/2 - 100, DRAGBOX_WIDTH, DRAGBOX_WIDTH)
pos_dragging = False

vel_drag = pygame.rect.Rect(WIDTH/2, HEIGHT/2 + 100, DRAGBOX_WIDTH, DRAGBOX_WIDTH)
vel_dragging = False

running = True
font = pygame.font.SysFont(None, 24)
pos_label = font.render('Pos', True, (0,0,0))
vel_label = font.render('Vel', True, (0,0,0))

while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:            
                if pos_drag.collidepoint(event.pos):
                    pos_dragging = True
                    mouse_x, mouse_y = event.pos
                    offset_x = pos_drag.x - mouse_x
                    offset_y = pos_drag.y - mouse_y
                elif vel_drag.collidepoint(event.pos):
                    vel_dragging = True
                    mouse_x, mouse_y = event.pos
                    offset_x = vel_drag.x - mouse_x
                    offset_y = vel_drag.y - mouse_y

        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:            
                pos_dragging = False
                vel_dragging = False

        elif event.type == pygame.MOUSEMOTION:
            if pos_dragging:
                mouse_x, mouse_y = event.pos
                pos_drag.x = mouse_x + offset_x
                pos_drag.y = mouse_y + offset_y
            elif vel_dragging:
                mouse_x, mouse_y = event.pos
                vel_drag.x = mouse_x + offset_x
                vel_drag.y = mouse_y + offset_y

    win.fill((255,255,255))

    pygame.draw.rect(win, (255,255,255), pos_drag)
    pygame.draw.rect(win, (255,255,255), vel_drag)

    pos = pygame.Vector2(pos_drag.x + DRAGBOX_WIDTH/2, pos_drag.y+DRAGBOX_WIDTH/2)
    vel = pygame.Vector2(vel_drag.x + DRAGBOX_WIDTH/2, vel_drag.y+DRAGBOX_WIDTH/2)
    draw_arrow( win, pygame.Vector2(WIDTH/2, HEIGHT/2), pos, (0,0,0), 5, 20, 10)
    draw_arrow( win, pygame.Vector2(WIDTH/2, HEIGHT/2), vel, (0,0,0), 5, 20, 10)

    pos_dir = pos - center
    pos_dir = pos_dir.normalize()
    win.blit(pos_label, pos + pos_dir * 15 - (12, 5))

    vel_dir = vel - center
    vel_dir = vel_dir.normalize()
    win.blit(vel_label, vel + vel_dir * 15 - (12,5))

    NN_input = np.array([pos.x - center.x , center.y - pos.y , vel.x - center.x, center.y - vel.y]) / 100.0
    theta = NN.feed_forward(params, NN_input)

    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    rotated_velocity = np.matmul(rotation_matrix, np.array([0.0, -1.0]))

    draw_arrow(win, center, center + rotated_velocity * 100, (255,0,0),5,20,10)

    pygame.display.flip()
