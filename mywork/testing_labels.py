'''
The script is present how to using labe to detect
'''


from __future__ import print_function
import vizdoom as vzd

from random import choice
from time import sleep
import settings
import cv2

game = settings.game_default(True)
game.init()

actions = [
    [True, False, False, False, False], 
    [False, True, False, False, False], 
    [False, False, True, False, False],
    [False, False, False, True, False],
    [False, False, False, False, True]
    ]
    
episodes = 10
doom_red_color = [0,0,203]
doom_blue_color = [203,0,0]
doom_white_color = 255
sleep_time = 28


def draw_bounding_box(buffer, x, y, width, height, color):
    for i in range(width):
        #buffer[y, x + i, :] = color
        #buffer[y + height, x + i, :] = color
        buffer[y, x + i] = color
        buffer[y + height, x + i] = color

    for i in range(height):
        #buffer[y + i, x, :] = color
        #buffer[y + i, x + width, :] = color
        buffer[y + i, x] = color
        buffer[y + i, x + width] = color


for i in range(episodes):
    print("Episode #" + str(i + 1))
    seen_in_this_episode = set()


    # Starts a new episode. It is not needed right after init() but it doesn't cost much. At least the loop is nicer.
    game.new_episode()

    while not game.is_episode_finished():

        # Gets the state
        state = game.get_state()

        # Which consists of:
        n = state.number
        vars = state.game_variables
        screen_buf = state.screen_buffer
        depth_buf = state.depth_buffer
        labels_buf = state.labels_buffer
        automap_buf = state.automap_buffer
        labels = state.labels

        #if labels_buf is not None:
        #    cv2.imshow('ViZDoom Labels Buffer', labels_buf)

        for l in state.labels:
            if l.object_name in ["Cacodemon", "GreenArmor"]:
                draw_bounding_box(labels_buf, l.x, l.y, l.width, l.height, doom_white_color)
            else:
                draw_bounding_box(labels_buf, l.x, l.y, l.width, l.height, doom_white_color)
        cv2.imshow('ViZDoom Screen Buffer', labels_buf)

        cv2.waitKey(sleep_time)

        
        game.make_action(choice(actions))
        # Prints state's game variables and reward.
        #print("State #" + str(n))
        #print("Player position X:", state.game_variables[0], "Y:", state.game_variables[1], "Z:", state.game_variables[2])
        #print("Labels:")
        #print("=====================")
        
        for l in state.labels:
            seen_in_this_episode.add(l.object_name)
            # print("---------------------")
            print("Label:", l.value, "object id:", l.object_id, "object name:", l.object_name)
            print("Object position x:", l.object_position_x, "y:", l.object_position_y, "z:", l.object_position_z)
        

            # Other available fields:
            #print("Object rotation angle", l.object_angle, "pitch:", l.object_pitch, "roll:", l.object_roll)
            #print("Object velocity x:", l.object_velocity_x, "y:", l.object_velocity_y, "z:", l.object_velocity_z)
                


# It will be done automatically anyway but sometimes you need to do it in the middle of the program...
game.close()

