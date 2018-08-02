#!/usr/bin/env python3

from __future__ import print_function
import vizdoom as vzd
import numpy as np
import pandas as pd

from random import choice
from time import sleep

ACTIONS = [[True, False, False], [False, True, False], [False, False, True]]
ACTIONS_NAME = ['left','right','shot']
STATAE = 200
EPISODES = 100
LEARNING_RATE = .8
EXPLORE_RATE = .9
SLEEP = 1.0 / vzd.DEFAULT_TICRATE
game = vzd.DoomGame()

Q  = pd.DataFrame(
        np.zeros((STATAE, len(ACTIONS))),
        columns= ACTIONS_NAME
    )
DEBUG_MODE = True


def game_default():
    # Create DoomGame instance. It will run the game and communicate with you.
    #game = vzd.DoomGame()

    # Now it's time for conset_available_game_variablesfiguration!
    # load_config could be used to load configuration instead of doing it here with code.
    # If load_config is used in-code configuration will also work - most recent changes will add to previous ones.
    # game.load_config("../../scenarios/basic.cfg")

    # Sets path to additional resources wad file which is basically your scenario wad.
    # If not specified default maps will be used and it's pretty much useless... unless you want to play good old Doom.
    game.set_doom_scenario_path("../../scenarios/basic.wad")

    # Sets map to start (scenario .wad files can contain many maps).
    game.set_doom_map("map01")

    # Sets resolution. Default is 320X240
    game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)

    # Sets the screen buffer format. Not used here but now you can change it. Defalut is CRCGCB.
    game.set_screen_format(vzd.ScreenFormat.RGB24)

    # Enables depth buffer.set_screen_resolution
    game.set_depth_buffer_enabled(True)

    # Enables labeling of in game objects labeling.
    game.set_labels_buffer_enabled(True)

    # Enables buffer with top down map of the current episode/level.
    game.set_automap_buffer_enabled(True)

    # Sets other rendering options (all of these options except crosshair are enabled (set to True) by default)
    game.set_render_hud(False)
    game.set_render_minimal_hud(False)  # If hud is enabled
    game.set_render_crosshair(False)
    game.set_render_weapon(True)
    game.set_render_decals(False)  # Bullet holes and blood on the walls
    game.set_render_particles(False)
    game.set_render_effects_sprites(False)  # Smoke and blood
    game.set_render_messages(False)  # In-game messages
    game.set_render_corpses(False)
    game.set_render_screen_flashes(True)  # Effect upon taking damage or picking up items

    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)

    # Adds game variables that will be included in state.
    game.add_available_game_variable(vzd.GameVariable.AMMO2)

    # Causes episodes to finish after 200 tics (actions)
    game.set_episode_timeout(STATAE-1)

    # Makes episodes start after 10 tics (~after raising the weapon)
    game.set_episode_start_time(10)

    # Makes the window appear (turned on by default)
    game.set_window_visible(True)

    # Turns on the sound. (turned off by default)
    game.set_sound_enabled(True)

    # Sets the livin reward (for each move) to -1
    game.set_living_reward(-1)

    # Sets ViZDoom mode (PLAYER, ASYNC_PLAYER, SPECTATOR, ASYNC_SPECTATOR, PLAYER mode is default)
    game.set_mode(vzd.Mode.PLAYER)

    # Enables engine output to console.
    #game.set_console_enabled(True)

    # Initialize the game. Further configuration won't take any effect from now on.
    game.init()

def Update_function():
    for i in range(EPISODES):
        print("Episode #", str(i+1))
        game.new_episode()

        while not game.is_episode_finished():

            state = game.get_state()

            #Which consists of:
            stateN = state.number
            vars = state.game_variables
            screen_buf = state.screen_buffer
            depth_buf = state.depth_buffer
            labels_buf = state.labels_buffer
            automap_buf = state.automap_buffer
            labels = state.labels

            #get reward

            action = Select_action(stateN)



            reward = game.make_action(action)
            #state = game.get_state()


            RL_Brain(stateN,action,reward,stateN+1)
            if(DEBUG_MODE):
                print('action = ',action)
                print('reward = ',reward)
                print('state = ', stateN )

    if int(i) is 10:
        np.save('model/10times.npy', Q)
        #record_rate(10, explore_rate)
        sleep(0.25)

    elif int(i) is 100:
        np.save('model/100times.npy', Q)
        #record_rate(100, explore_rate)
        sleep(0.25)

        print("Episode finished.")
        print("Total reward:", game.get_total_reward())
        print("************************")
    game.close()


def RL_Brain(s,a,r,s_):

    q_predict = Q.loc[s,a]


    if r == 100:
        #q_target = r * EXPLORE_RATE * Q.iloc[s_,:].max()
        q_target = r  * Q.iloc[s_, :].max()
    else:
        q_target = r
    Q.loc[s,a] += LEARNING_RATE*(q_target - q_predict)

    #Q Algorithm
    pass
def Select_action(state):

    state_actions = Q.iloc[state, :]
    if (np.random.uniform() > EXPLORE_RATE) or \
            (state_actions.all()==0):
        action_name = np.random.choice(ACTIONS_NAME)
    else:
        action_name = state_actions.argmax()

    action_index = ACTIONS_NAME.index(action_name)


    return ACTIONS[action_index]

def show_table(talbe_name):
    # load...
    load_table = np.load(talbe_name)

    print(load_table)

    

if __name__ == "__main__" :

    #init
    game_default()
    #init Q_table
    Update_function()
    #show_table('model/10times.npy')


