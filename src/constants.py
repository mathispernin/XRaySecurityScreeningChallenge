import numpy as np

# global constants
CLASS_NAMES = {0: 'Hammer', 1: 'Knife', 2: 'Gun', 3: 'Wrench', 4: 'HandCuffs', 5: 'Bullet', 6: 'Background'}
NUM_CLASSES = 7
BACKGROUND_CLASS = 6
SCALES = [1.0, 0.75, 0.5]

# preprocessing
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_GRID_SIZE = (8, 8)
BILATERAL_D = 5
BILATERAL_SIGMA_COLOR = 75
BILATERAL_SIGMA_SPACE = 75

# HOG
HOG_WINDOW_SIZE = (96, 96)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (8, 8)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_BLOCK_NORM = 'L2-Hys'

# LBP
LBP_P = 8
LBP_R = [1, 3]
LBP_METHOD = 'uniform'
LBP_NUM_BINS = 59

# color histogram
HIST_HUE_BINS = 32
HIST_SAT_BINS = 16

# GLCM
GLCM_DISTANCES = [1]
GLCM_ANGLES = [0, np.pi/4, np.pi/2, 3*np.pi/4]
GLCM_LEVELS = 32
GLCM_PROPS = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
MAX_TEXTURE_SIZE = 256

# SIFT
SIFT_N_FEATURES = 100
BOVW_VOCAB_SIZE = 200
