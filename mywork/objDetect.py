import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import skimage.measure
from skimage import io
from settings import *

path = './img/enemy1/sample/'


def Capture():
    game = game_default(False)
    
    game.init()
    tempPicture = game.get_state().screen_buffer
    
    #plt.imshow(tempPicture)
    #plt.show()

    game.close()
    saveImg(tempPicture)
    
    print('finish!')
def ORB(img):
    #Initiate ORB detector
    orb = cv.ORB_create()
    # find the keypoints with ORB
    kp = orb.detect(img,None)
    # compute the descriptors with ORB
    kp, des = orb.compute(img, kp)
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    #plt.imshow(img2,cmap='gray'), plt.show()
    #plt is show bgr, you need change to rgb
    plt.axis("off")
    plt.imshow(img2),plt.show()
    #plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)), plt.show()

def orb_compare():
    
    
    
    img1 = cv.imread(path+'e1_rgb.png')
    img2 = cv.imread(path+'7.jpg')
    #gray
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    
    orb = cv.ORB_create()

    kp1 = orb.detect(img1,None)
    kp2 = orb.detect(img2,None)
    kp1, des1 = orb.compute(img, kp1)
    kp2, des2 = orb.compute(img, kp2)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)), plt.show()


def maxPooling(X,x,Y,y,img):
    return  skimage.measure.block_reduce(img,(x,y),np.max)

def akaze_compare():
    img1 = cv.imread(path+'e1_rgb.png')
    img2 = cv.imread(path+'7.jpg')
    #gray
    gray1 = cv.cvtColor(img1, cv.COLOR_BGR2GRAY)
    gray2 = cv.cvtColor(img2, cv.COLOR_BGR2GRAY)
    akaze = cv.AKAZE_create()
    kp1, des1 = akaze.detectAndCompute(gray1, None)
    kp2, des2 = akaze.detectAndCompute(gray2, None)

    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    img3 = cv.drawMatches(img1, kp1, img2, kp2, matches[:10], None, flags=2)
    plt.imshow(cv.cvtColor(img3, cv.COLOR_BGR2RGB)), plt.show()
    
def monster_feature():
    #img = cv.imread(path+'7.jpg')
    img = cv.imread(path+'e1_rgb.png')
    #gray
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
    blurred = cv.GaussianBlur(gray,(17,25),0)
    canny = cv.Canny(blurred, 30 ,150)
    #orb = cv.ORB_create()
    akaze = cv.AKAZE_create()
    kp, des = akaze.detectAndCompute(blurred, None)
    # draw only keypoints location,not size and orientation
    img2 = cv.drawKeypoints(blurred, kp, None, color=(0,255,0), flags=0)
    
    result = np.hstack([gray, blurred, canny])
    plt.imshow(img2,cmap='gray'), plt.show()
    
    
    blurred = cv.GaussianBlur(gray, (15, 15), 0)
    canny = cv.Canny(blurred, 30, 150)


    result = np.hstack([gray, blurred, canny])
    plt.imshow(result,cmap='gray'), plt.show()

    '''
    akaze = cv.AKAZE_create()
    kp, des = akaze.detectAndCompute(gray, None)
    img2 = cv.drawKeypoints(img, kp, None, color=(0,255,0), flags=0)
    plt.imshow(cv.cvtColor(img2, cv.COLOR_BGR2RGB)), plt.show()
    '''

#Capture()
#img = cv.imread(path+'4.jpg')
#ORB(img)
#orb_compare()
#akaze_compare()
monster_feature()
'''
game = game_default()
tempPicture = game.get_state().depth_buffer
game.init()
game.close()'''
