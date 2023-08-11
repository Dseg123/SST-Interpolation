import sys
sys.path.append("..")

import xarray as xr 
import matplotlib.pyplot as plt

import numpy as np                 #for general calculations.

from matplotlib.animation import FuncAnimation  #creating animations and videos.
from IPython.display import Video, HTML         #embedding videos in the notebook

#this command allows interactive plots in noteboks
from pathlib import Path

def ids(my_params):
    space_bounds = my_params["spaceBounds"]
    block_size = my_params["blockSize"] // my_params["avgSize"]
    tile_size = my_params["tileSize"]
    num_tiles = my_params["numTiles"]
    train_length = my_params["trainLength"]
    val_length = my_params["valLength"]
    threshold = my_params["threshold"]
    data_dir = my_params["dataDir"]
    window_size = my_params["windowSize"]

    
    
    p = Path(data_dir + '/raw_data')
    pg = list(p.glob('*.nc'))
    num_blocks = len(pg) // (my_params["blockSize"])
    print(num_blocks)
    
    print("a")
    tile_nums = np.load(data_dir + '/masked_stats.npy')[:, :, 0]
    tile_nums = tile_nums / (block_size * tile_size ** 2)
    
    is_valid = np.zeros(tile_nums.shape)
    for i in range(tile_nums.shape[1]):
        if i + window_size - 1 < tile_nums.shape[1]:
            is_valid[:, i] = np.mean(tile_nums[:, i:i+window_size], axis=1)
    
    print("b")
    viable = is_valid > threshold
    schedule = np.zeros(num_blocks)
    counter = 0
    while counter < len(schedule):
        schedule[counter:counter + train_length - window_size] = 1
        schedule[counter + train_length:counter + train_length + val_length - window_size] = 2
        counter += val_length + train_length
    if window_size > 1:
        schedule[-window_size:] = 0 
    
    print("c")
    final = np.swapaxes(np.swapaxes(viable, 1, -1) * schedule, -1, 1)
    
    train_locs = np.transpose((final == 1).nonzero())
    val_locs = np.transpose((final == 2).nonzero())
    
    new_train_locs = np.zeros((train_locs.shape[0] * block_size, 3))
    for i in range(len(train_locs)):
        for j in range(block_size):
            new_train_locs[i * block_size + j, :2] = train_locs[i]
            new_train_locs[i * block_size + j, 2] = j
    
    print("d")
    new_val_locs = np.zeros((val_locs.shape[0] * block_size, 3))
    for i in range(len(val_locs)):
        for j in range(block_size):
            new_val_locs[i * block_size + j, :2] = val_locs[i]
            new_val_locs[i * block_size + j, 2] = j
    
    print("e")
    np.save(data_dir + '/train_ids.npy', new_train_locs)
    np.save(data_dir + '/val_ids.npy', new_val_locs)