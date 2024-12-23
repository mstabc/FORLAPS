
import os

DIRECTORY = '../evaluation_dataset'
PLOT_DIRECTORY = 'plots'

ALPHA = 0.2
GAMMA = 0.8
NUM_COP = 50

if not os.path.exists(PLOT_DIRECTORY):
    os.makedirs(PLOT_DIRECTORY)