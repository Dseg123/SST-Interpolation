import sys
sys.path.append("..")

import xarray as xr 
import matplotlib.pyplot as plt

import numpy as np                 #for general calculations.

from matplotlib.animation import FuncAnimation  #creating animations and videos.
from IPython.display import Video, HTML         #embedding videos in the notebook

#this command allows interactive plots in noteboks
from pathlib import Path

import random


def masked_data(my_params):
    space_bounds = my_params["spaceBounds"]
    block_size = my_params["blockSize"]
    tile_size = my_params["tileSize"]
    num_tiles = my_params["numTiles"]
    data_dir = my_params["dataDir"]
    rand_seed = my_params["randSeed"]
    
    
    p = Path(data_dir + '/raw_data')
    pg = list(p.glob('*.nc'))
    num_blocks = len(pg) // my_params["blockSize"]
    
    tile_str = data_dir + '/full_data/tile{tile_num}_block{block_num}.npy'
    clouds_str = data_dir + '/cloud_data/tile{tile_num}_block{block_num}.npy'
    masks_str = data_dir + '/masked_data/tile{tile_num}_block{block_num}.npy'
    indices = [i for i in range(num_tiles)]
    random.seed(rand_seed)
    random.shuffle(indices)
    
    stats = np.zeros((num_tiles, num_blocks, 3))
    print(num_tiles, num_blocks)
    for i in range(num_tiles):
        for j in range(num_blocks):
            print(i, j)
            masked_data = np.load(tile_str.format(tile_num=i, block_num=j))
            cloud_data = np.load(clouds_str.format(tile_num = indices[i], block_num=j))
            masked_data[cloud_data] = np.nan
            np.save(masks_str.format(tile_num=i, block_num=j), masked_data)
            
            stats[i, j, 0] = np.sum(~np.isnan(masked_data))
            stats[i, j, 1] = np.nansum(masked_data)
            stats[i, j, 2] = np.nansum(np.square(masked_data))
            
    np.save(data_dir + "/masked_stats.npy", stats)

    