import tensorflow as tf
from tensorflow import keras
import numpy as np
import copy

import sys
sys.path.append("..")

from get_params import params

my_params = params()
batch_size = my_params['batchSize']
window_size = my_params['windowSize']
block_size = my_params['blockSize']
data_dir = my_params['dataDir']
tile_size = my_params['tileSize']
rand_seed = my_params['randSeed']

def get_stats():
    tile_stats = np.load(data_dir + '/masked_stats.npy')
    sum_0 = tile_stats[:, :, 0]
    sum_1 = tile_stats[:, :, 1]
    sum_2 = tile_stats[:, :, 2]
    
    final_num = np.sum(sum_0)
    if final_num == 0:
        final_mean = 0
        final_std = 1
    else:
        final_mean = np.nansum(sum_1)/final_num
        final_std = np.sqrt(np.nansum(sum_2)/final_num - (final_mean)**2)
    
    return (final_num, final_mean, final_std)

def get_one_batch(list_IDs_temp, stats):
    #print(list_IDs_temp)
    #print(len(list_IDs_temp))
    num_sst, mean_sst, std_sst = stats
        
    batch = np.zeros((2, batch_size, block_size * window_size, tile_size, tile_size))

    in_data_str = data_dir + '/masked_data/tile{tileNum}_block{blockNum}.npy'
    out_data_str = data_dir + '/full_data/tile{tileNum}_block{blockNum}.npy'
    for i, ID in enumerate(list_IDs_temp):

        x_window = np.zeros((block_size * (window_size + 1), tile_size, tile_size))
        y_window = np.zeros((block_size * (window_size + 1), tile_size, tile_size))

        for block in range(window_size + 1):
            x_window[block_size * block : block_size * (block + 1), :, :] = np.load(in_data_str.format(tileNum = int(ID[0]), blockNum = block + int(ID[1])))
            y_window[block_size * block : block_size * (block + 1), :, :] = np.load(out_data_str.format(tileNum = int(ID[0]), blockNum = block + int(ID[1])))

        x_window = x_window[int(ID[2]) : int(ID[2]) + window_size * block_size, :, :]
        y_window = y_window[int(ID[2]) : int(ID[2]) + window_size * block_size, :, :]
        #print(np.nanmin(x_window), np.nanmax(x_window), np.nanmin(y_window), np.nanmax(y_window))
        x_window[np.isnan(x_window)] = 0
        x_window[x_window < 273] = 0
        x_window[x_window != 0] = (x_window[x_window != 0] - mean_sst)/(std_sst + keras.backend.epsilon())

        batch[0, i, :, :, :] = x_window

        y_window[y_window < 273] = np.nan
        y_window[~np.isnan(y_window)] = (y_window[~np.isnan(y_window)] - mean_sst)/(std_sst + keras.backend.epsilon())
        batch[1, i, :, :, :] = y_window

    #print(np.nanmean(X[0, :, :, :]))
    #print(X[0, :, :, :])
    return batch
    

def batch_data():
    stats = get_stats()
    
    train_ids = np.load(data_dir + '/train_ids.npy')
    val_ids = np.load(data_dir + '/val_ids.npy')
    
    np.random.seed(rand_seed)
    
    train_indexes = np.arange(len(train_ids))
    val_indexes = np.arange(len(val_ids))
    
    np.random.shuffle(train_indexes)
    np.random.shuffle(val_indexes)
    
    batch_str = data_dir + '/batch_data/train_batch{batchNum}.npy'
    print("Num train batches:", str(int(np.floor(len(train_ids) / batch_size))))
    for i in range(int(np.floor(len(train_ids) / batch_size))):
        print(i)
        indexes = train_indexes[i * batch_size : (i + 1) * batch_size]
        list_IDs_temp = [train_ids[k] for k in indexes]
        batch = get_one_batch(list_IDs_temp, stats)
        np.save(batch_str.format(batchNum = i), batch)
    
    print("Num val batches:", str(int(np.floor(len(val_ids) / batch_size))))
    batch_str = data_dir + '/batch_data/val_batch{batchNum}.npy'
    for i in range(int(np.floor(len(val_ids) / batch_size))):
        print(i)
        indexes = val_indexes[i * batch_size : (i + 1) * batch_size]
        list_IDs_temp = [val_ids[k] for k in indexes]
        batch = get_one_batch(list_IDs_temp, stats)
        np.save(batch_str.format(batchNum = i), batch)
        
        
        