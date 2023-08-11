import numpy as np
from pathlib import Path
from gan_params import get_params

def gan_data(tile_size, tile_stride, num_hours, data_dir):
    print(tile_size, num_hours)
    nums = []
    sums = []
    squares = []
    
    counter = 0
    counter2 = 0
    p = Path('../' + data_dir + '/full_data')
    pg = list(p.glob('*.npy'))
    pg.sort()
    
    print(len(pg))
    
    folder_path = "gan_" + data_dir
    print(folder_path)
    out_path = folder_path + f"/real_data_size{tile_size}_stride{tile_stride}_depth{num_hours}/sample_"
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
                    pic = data[h:h+num_hours, r:r + tile_size, c:c + tile_size]
                    if np.isnan(pic).sum() == 0:
                        np.save(out_path + str(counter) + ".npy", pic)
                        nums.append(np.sum(~np.isnan(pic)))
                        sums.append(np.sum(pic))
                        squares.append(np.sum(np.square(pic)))
                        counter += 1
                    c += tile_stride
                r += tile_stride
            h += 1
            #print(h, h + num_hours - 1, tot_hours)
    
    nums = np.array(nums)
    sums = np.array(sums)
    squares = np.array(squares)
    
    mean = np.sum(sums)/(np.sum(nums))
    std = np.sqrt(np.sum(squares)/(np.sum(nums)) - np.square(mean))
    out_path = folder_path + f"/real_data_size{tile_size}_stride{tile_stride}_depth{num_hours}/stats.npy"
    np.save(out_path, np.array([mean, std, counter]))
    return counter, counter2

if __name__ == "__main__":
    my_params = get_params()
    tile_size = my_params["filterSize"]
    tile_stride = my_params["filterStride"]
    data_dir = my_params["dataDir"]
    depth = my_params["filterDepth"]
    print(gan_data(tile_size, tile_stride, depth, data_dir))
        
                        
                    