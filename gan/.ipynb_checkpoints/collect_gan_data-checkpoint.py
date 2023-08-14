import numpy as np
from pathlib import Path
import os
import pandas as pd

def get_params():
    inDataDir = "training_data"
    filterSize = 32
    filterStride = 16
    filterDepth = 2
    outDataDir = "gan_training_data"
    
    return {
        'inDataDir': inDataDir,
        'outDataDir': outDataDir,
        'filterSize': filterSize,
        'filterStride': filterStride,
        'filterDepth': filterDepth
    }

def gan_data():
    params = get_params()
    in_data_dir = params['inDataDir']
    out_data_dir = params['outDataDir']
    filter_size = params['filterSize']
    filter_stride = params['filterStride']
    filter_depth = params['filterDepth']
    
    nums = []
    sums = []
    squares = []
    
    counter = 0
    counter2 = 0
    p = Path('../' + in_data_dir + '/full_data')
    pg = list(p.glob('*.npy'))
    pg.sort()
    
    print(len(pg))
    
    folder_path = out_data_dir
    print(folder_path)
    os.system(f"mkdir {folder_path}")
    os.system(f"mkdir {folder_path}/samples")
    params_df = pd.DataFrame.from_records([params])
    params_df.to_csv(f'{folder_path}/data_params.csv')
    
    out_path = folder_path + "/samples/sample_"
    for i in range(len(pg)):
        print(i)
        if i % 1000 == 0:
            print(i)
        data = np.load(pg[i])
        print(data.shape)
        tot_hours = data.shape[0]
        tot_size = data.shape[1]
        
        h = 0
        #print(data.size)
        while h + num_hours - 1 < tot_hours:
            r = 0
            while r + tile_size - 1 < tot_size:
                c = 0
                while c + tile_size - 1 < tot_size:
                    counter2 += 1
                    #print(h, r, c)
                    pic = data[h:h+filter_depth, r:r + filter_size, c:c + filter_size]
                    if np.isnan(pic).sum() == 0:
                        np.save(out_path + str(counter) + ".npy", pic)
                        nums.append(np.sum(~np.isnan(pic)))
                        sums.append(np.sum(pic))
                        squares.append(np.sum(np.square(pic)))
                        counter += 1
                    c += filter_stride
                r += filter_stride
            h += 1
            #print(h, h + num_hours - 1, tot_hours)
    
    nums = np.array(nums)
    sums = np.array(sums)
    squares = np.array(squares)
    
    mean = np.sum(sums)/(np.sum(nums))
    std = np.sqrt(np.sum(squares)/(np.sum(nums)) - np.square(mean))
    out_path = f'{folder_path}/stats.npy'
    np.save(out_path, np.array([mean, std, counter]))
    return counter, counter2

if __name__ == "__main__":
    print(gan_data())
        
                        
                    