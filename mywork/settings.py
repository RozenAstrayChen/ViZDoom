
from __future__ import print_function
import vizdoom as vzd
import skimage.color, skimage.transform
from scipy import  misc
from skimage import io
import numpy as np
import time
from os import walk


resolution = (108, 60)
STATAE = 200

path = './img/enemy1'

def game_default(bool):

    print("Initializing doom...")
    game = vzd.DoomGame()
    game.set_doom_map("map01")
    #game.load_config("../scenarios/simpler_basic.cfg")
    game.set_doom_scenario_path("../scenarios/deathmatch")
    '''
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
    '''
    game.add_available_game_variable(vzd.GameVariable.POSITION_X)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Y)
    game.add_available_game_variable(vzd.GameVariable.POSITION_Z)
    game.set_labels_buffer_enabled(True)
    # Adds buttons that will be allowed.
    game.add_available_button(vzd.Button.MOVE_FORWARD)
    game.add_available_button(vzd.Button.MOVE_BACKWARD)
    game.add_available_button(vzd.Button.MOVE_LEFT)
    game.add_available_button(vzd.Button.MOVE_RIGHT)
    game.add_available_button(vzd.Button.ATTACK)
    

    game.set_window_visible(bool)
    game.add_available_game_variable(vzd.GameVariable.WEAPON2)
    game.set_mode(vzd.Mode.PLAYER)
    #game.set_screen_format(vzd.ScreenFormat.GRAY8)
    game.set_screen_format(vzd.ScreenFormat.RGB24)
    #game.set_screen_resolution(vzd.ScreenResolution.RES_640X480)
    game.set_screen_resolution(vzd.ScreenResolution.RES_512X288)
    
    print("Doom initialized.")
    game.init()

    return game


# Converts and down-samples the input image
def preprocess(img):
    #img = skimage.transform.resize(img, resolution)
    #img = img.astype(np.float32)
    #img = skimage.color.rgb2gray(img)
    return img
def saveImg(img,tf):

    strtime=  time.strftime("%H_%M_%S", time.localtime()) +'.jpg'
    filePath = path+tf+strtime
    print(filePath)
    io.imsave(filePath,img)

def readImg(imgName):
    filepath = path +imgName

    img = misc.imread(filepath)


    return img

def saveEnemy(game,delay,epsoid,bool):
    for i in range(0,epsoid):
        print('epsoid ->',i)
        game.init()
        s1 = preprocess(game.get_state().screen_buffer)
        game.close()
        saveImg(s1,bool)

        time.sleep(delay)

    print('finish!')

def findAllFile():
    table = np.zeros((100, 84, 84))
    tableName = []
    i = 0
    for root, dirs, files in walk(path):
        for file in files:

            imgary = readImg(file)

            table[i] = imgary
            tableName.append(str(file))

            i += 1
            if i == 100:
                break

    return table,tableName

def SaveXModule(nparray,Names):
    np.save('Xepsoide1.npy',nparray)
    '''
    with open('names.txt' ,'w') as file:
        #file.write(str(Names))
        for name in Names:
            file.write(str(name))
            file.write('\n')
        file.close()
    '''
def SaveYModule(nparray):
    np.save('Yepsoide1.npy',nparray)

def Load_Yvalue(path):
    text_file = open(path, 'r')
    lines = text_file.read().split('\n')
    # delete last ' '
    del lines[-1]

    y = np.array(lines)
    y = np.asfarray(y,float)

    return y
def Reshape_Y(arry):
    y = np.zeros((100,3))
    for i in range(0,100):
        index = int(arry[i])
        y[i][index] = 1

    return y
