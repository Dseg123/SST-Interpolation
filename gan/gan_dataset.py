import tensorflow as tf
import numpy as np
from pathlib import Path

# def get_dataset(data_dir, batch_size = 12, buffer_size = 6000):
#     p = Path(data_dir)
#     pg = list(p.glob('*.npy'))
#     #print(pg)
    

#     sample = np.load(pg[0])
#     all_data = np.zeros((len(pg), sample.shape[0], sample.shape[1], sample.shape[2], 1))
#     for i in range(len(pg)):
#         #print(i)
#         all_data[i, :, :, :, 0] = np.load(pg[i])
    
#     cutoff = int(0.8 * all_data.shape[0])
#     train_data = all_data[:cutoff, :, :, :, :]
#     test_data = all_data[cutoff:, :, :, :, :]
#     train_data_tensor = tf.convert_to_tensor(train_data, dtype=tf.float32)
#     mean = np.mean(train_data)
#     std = np.std(train_data)
#     test_data_tensor = tf.convert_to_tensor(test_data, dtype=tf.float32)
    
#     train_data_tensor = (train_data_tensor - mean)/std
#     test_data_tensor = (test_data_tensor - mean)/std
#     # Batch and shuffle the data
#     train_dataset = tf.data.Dataset.from_tensor_slices(train_data_tensor).shuffle(buffer_size).batch(batch_size)
#     test_dataset = tf.data.Dataset.from_tensor_slices(test_data_tensor).shuffle(buffer_size).batch(batch_size)
#     return mean, std, train_dataset, test_dataset


def get_gan_dataset(data_dir, list_ids, batch_size = 12, buffer_size = 6000):
    
    stats = np.load(data_dir + '/stats.npy')
    example = np.load(data_dir + '/sample_0.npy')
    
    
    def generate_sample():
    
        for idx in list_ids:
            arr = np.load(data_dir + '/sample_' + str(int(idx)) + '.npy')
            data_tensor = tf.convert_to_tensor(arr, dtype=tf.float32)
            norm_tensor = (data_tensor - stats[0])/(stats[1])

            yield norm_tensor
    
    
    dataset = tf.data.Dataset.from_generator(
     generate_sample,
     output_signature=(tf.TensorSpec(shape=example.shape, dtype=tf.float32)))
    
    return dataset.batch(batch_size) #should this include shuffle?
    
# if __name__ == "__main__":
#     print(get_gan_dataset("gan_data_size64_duration6_window2_4hourly/real_data_size64_stride1_depth1"))