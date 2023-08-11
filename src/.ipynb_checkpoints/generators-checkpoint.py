import tensorflow as tf
from tensorflow import keras
import numpy as np
from pathlib import Path
import copy

import sys
sys.path.append("..")

def get_summary_stats(tile_stats):
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

def get_src_dataset(data_dir, list_ids, window_size, batch_size = 12, buffer_size = 6000):
    
    num_sst, mean_sst, std_sst = get_summary_stats(np.load(data_dir + '/masked_stats.npy'))
    
    in_data_dir = data_dir + '/masked_data'
    out_data_dir = data_dir + '/full_data'
    in_data_str = in_data_dir + '/tile{tileNum}_block{blockNum}.npy'
    out_data_str = out_data_dir + '/tile{tileNum}_block{blockNum}.npy'
    
    example = np.load(in_data_str.format(tileNum=0, blockNum=0))
    dim = example.shape
    new_dim = (dim[0] * window_size, dim[1], dim[2])
    
    
    def generate_sample():
    
        for idx in list_ids:
            x_window = np.zeros((dim[0] * (window_size + 1), dim[1], dim[2]))
            y_window = np.zeros((dim[0] * (window_size + 1), dim[1], dim[2]))
            for block in range(window_size + 1):
                x_window[dim[0] * block : dim[0] * (block + 1), :, :] = np.load(in_data_str.format(tileNum=int(idx[0]), blockNum = block + int(idx[1])))
                y_window[dim[0] * block : dim[0] * (block + 1), :, :] = np.load(out_data_str.format(tileNum=int(idx[0]), blockNum = block + int(idx[1])))     
            x_window = x_window[int(idx[2]) : int(idx[2]) + window_size * dim[0], :, :]
            y_window = y_window[int(idx[2]) : int(idx[2]) + window_size * dim[0], :, :]
            
            x_window[np.isnan(x_window)] = 0
            x_window[x_window < 273] = 0
            x_window[x_window != 0] = (x_window[x_window != 0] - mean_sst)/(std_sst + keras.backend.epsilon())
            
            y_window[y_window < 273] = np.nan
            y_window[~np.isnan(y_window)] = (y_window[~np.isnan(y_window)] - mean_sst)/(std_sst + keras.backend.epsilon())
            
            yield x_window, y_window
            
    dataset = tf.data.Dataset.from_generator(
     generate_sample,
     output_signature=(tf.TensorSpec(shape=tf.TensorShape(new_dim), dtype=tf.float32), 
                       tf.TensorSpec(shape=tf.TensorShape(new_dim), dtype=tf.float32)))
    
    return mean_sst, std_sst, dataset.batch(batch_size) #should this include shuffle?

# class DataGeneratorFast(keras.utils.Sequence):
    
#     def __init__(self, num_batches, task = 'train', shuffle = True):
#         self.num_batches = num_batches
#         self.task = task
#         self.shuffle = shuffle
        
#         self.on_epoch_end()
    
#     def __len__(self):
#         return self.num_batches
    
#     def __getitem__(self, index):
#         #print(index)
#         batch_id = self.indexes[index]
#         batch = np.load(data_dir + f'/batch_data/{self.task}_batch{batch_id}.npy')
#         return batch[0, :, :, :, :], batch[1, :, :, :, :]


#     def on_epoch_end(self):
#         self.indexes = np.arange(self.num_batches)
#         if self.shuffle:
#             np.random.seed(rand_seed)
#             np.random.shuffle(self.indexes)
       

# class DataGenerator(keras.utils.Sequence):
    
#     def __init__(self, list_IDs, stats, shuffle = True):
#         self.list_IDs = list_IDs
#         self.shuffle = shuffle
#         self.stats = stats
        
        
#         self.batch_size = batch_size
#         self.window_size = window_size
#         self.block_size = block_size
#         self.in_data_dir = data_dir + '/masked_data'
#         self.out_data_dir = data_dir + '/full_data'
#         self.dim = (block_size * window_size, tile_size, tile_size)
#         self.on_epoch_end()
    
#     def __len__(self):
#         return int(np.floor(len(self.list_IDs) / self.batch_size))
    
#     def __getitem__(self, index):
        
#         #print(index)
#         indexes = self.indexes[index * int(self.batch_size):(index + 1)*int(self.batch_size)]
        
#         list_IDs_temp = [self.list_IDs[k] for k in indexes]
        
#         X, Y = self.__data_generation(list_IDs_temp)
        
#         #print(X.shape, Y.shape)
#         return X, Y

#     def on_epoch_end(self):
#         self.indexes = np.arange(len(self.list_IDs))
#         if self.shuffle:
#             np.random.seed(rand_seed)
#             np.random.shuffle(self.indexes)
            
#     def __data_generation(self, list_IDs_temp):
        
#         num_sst, mean_sst, std_sst = self.stats
        
#         X = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
#         Y = np.zeros((self.batch_size, self.dim[0], self.dim[1], self.dim[2]))
        
#         in_data_str = self.in_data_dir + '/tile{tileNum}_block{blockNum}.npy'
#         out_data_str = self.out_data_dir + '/tile{tileNum}_block{blockNum}.npy'
#         for i, ID in enumerate(list_IDs_temp):
#             #print(ID)
            
#             x_window = np.zeros((self.block_size * (self.window_size + 1), self.dim[1], self.dim[2]))
#             y_window = np.zeros((self.block_size * (self.window_size + 1), self.dim[1], self.dim[2]))
            
#             for block in range(self.window_size + 1):
#                 x_window[self.block_size * block : self.block_size * (block + 1), :, :] = np.load(in_data_str.format(tileNum = int(ID[0]), blockNum = block + int(ID[1])))
#                 y_window[self.block_size * block : self.block_size * (block + 1), :, :] = np.load(out_data_str.format(tileNum = int(ID[0]), blockNum = block + int(ID[1])))
            

#             x_window = x_window[int(ID[2]) : int(ID[2]) + self.window_size * self.block_size, :, :]
#             y_window = y_window[int(ID[2]) : int(ID[2]) + self.window_size * self.block_size, :, :]

#             #print(np.nanmin(x_window), np.nanmax(x_window), np.nanmin(y_window), np.nanmax(y_window))
#             x_window[np.isnan(x_window)] = 0
#             x_window[x_window < 273] = 0
#             x_window[x_window != 0] = (x_window[x_window != 0] - mean_sst)/(std_sst + keras.backend.epsilon())
            
#             X[i, :, :, :] = x_window
            
#             y_window[y_window < 273] = np.nan
#             y_window[~np.isnan(y_window)] = (y_window[~np.isnan(y_window)] - mean_sst)/(std_sst + keras.backend.epsilon())
#             Y[i, :, :, :] = y_window
        
#         #print(np.nanmean(X[0, :, :, :]))
#         #print(X[0, :, :, :])
#         return X, Y
            
