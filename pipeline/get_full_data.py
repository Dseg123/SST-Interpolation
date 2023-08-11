import sys
sys.path.append("..")

import xarray as xr 
import matplotlib.pyplot as plt

import numpy as np                 #for general calculations.

from matplotlib.animation import FuncAnimation  #creating animations and videos.
from IPython.display import Video, HTML         #embedding videos in the notebook

#this command allows interactive plots in noteboks
from pathlib import Path

def avg_data(tile, win_size):
    i = 0
    avg_tile = np.zeros((tile.shape[0] // win_size, tile.shape[1], tile.shape[2]))
    while i < tile.shape[0]:
        avg_tile[i // win_size, :, :] = np.nanmean(tile[i:i+win_size, :, :], axis=0)
        i += win_size
    return avg_tile
        
    
def full_data(my_params):
    space_bounds = my_params["spaceBounds"]
    block_size = my_params["blockSize"]
    tile_size = my_params["tileSize"]
    num_tiles = my_params["numTiles"]
    data_dir = my_params["dataDir"]
    tile_stride = my_params["tileStride"]
    avg_size = my_params["avgSize"]

    
    
    print("HI")
    p = Path(data_dir + '/raw_data')
    pg = list(p.glob('*.nc'))
    pg.sort()
    num_blocks = len(pg) // my_params["blockSize"]
    
    block_ind = 0
    height = space_bounds[1] - space_bounds[0]
    width = space_bounds[3] - space_bounds[2]
    SST_block = np.zeros((block_size, height, width))
    
    stats = np.zeros((num_tiles, num_blocks, 3))
    
    for i in range(len(pg)):
        print(i, pg[i])
        path = pg[i]
        data = xr.open_dataset(path)
        my_SST = data.sea_surface_temperature.values[0,space_bounds[0]:space_bounds[1], space_bounds[2]:space_bounds[3]]
        my_qual = data.quality_level.values[0,space_bounds[0]:space_bounds[1], space_bounds[2]:space_bounds[3]]
        my_SST[my_qual != 5] = np.nan

        SST_block[i % block_size, :, :] = my_SST

        if i % block_size == block_size - 1:    
            tile_ind = 0
            for r in range(0, height - tile_size + 1, tile_stride):
                for c in range(0, width - tile_size + 1, tile_stride):
                    #print(r, c)
                    tile_path = data_dir + '/full_data/tile' + str(tile_ind) + '_block' + str(block_ind) + '.npy'
                    cloud_path = data_dir + '/cloud_data/tile' + str(tile_ind) + '_block' + str(block_ind) + '.npy'
                    new_tile_data = SST_block[:, r:r + tile_size, c:c + tile_size]
                    new_tile_data = avg_data(new_tile_data, avg_size)
                    new_cloud_data = ~np.isnan(new_tile_data)
                    np.save(tile_path, new_tile_data)
                    np.save(cloud_path, new_cloud_data)
                    
                    stats[tile_ind, block_ind, 0] = np.sum(~np.isnan(new_tile_data))
                    stats[tile_ind, block_ind, 1] = np.nansum(new_tile_data)
                    stats[tile_ind, block_ind, 2] = np.nansum(np.square(new_tile_data))
                    
                    
                    tile_ind += 1
            block_ind += 1
    
    np.save(data_dir + '/full_stats.npy', stats)