import vizdoom as vzd
from skimage import io,data,color
from settings import *
POSTIVE = '/postive/'
NEGATIVE = '/negative/'

#%%
game = game_default(False)

s1 = preprocess(game.get_state().screen_buffer)
#%%

saveEnemy(game,1,1,NEGATIVE)