from pipeline.get_raw_data import raw_data
from pipeline.get_full_data import full_data
from pipeline.get_masked_data import masked_data
from pipeline.get_ids import ids
import time
import os

import datetime
import pandas as pd

def params():
    tileSize = 64 #Should be smaller than spaceBounds size
    tileStride = 32 #Should be smaller than tileSize
    spaceBounds = (1500, 2000, 2500, 3000)
    timeBounds = (0, 1390)
    blockSize = 24
    avgSize = 4 # should be less than blockSize
    windowSize = 2
    randSeed = 100
    batchSize = 12
    trainLength = 20 #Should be greater than windowSize
    valLength = 8 #Should be greater than windowSize and less than trainLength
    threshold = 0.2
    satelliteName = "ABI_G16-STAR-L2P-v2.70"
    startTime = "2018-01-01T00:00:00Z"
    endTime = "2018-07-01T00:00:00Z"
    dataDir = "training_data"
    
    
    numBlocks = (timeBounds[1] - timeBounds[0])//blockSize
    numTiles = ((spaceBounds[1] - tileSize - spaceBounds[0]) // tileStride + 1) * ((spaceBounds[3] - tileSize - spaceBounds[2]) // tileStride + 1)
    return {
        'satelliteName': satelliteName,
        'startTime': startTime,
        'endTime': endTime,
        'tileSize': tileSize, #number of pixels in one dimension of a tile
        'tileStride': tileStride, #distance between top-left pixels of consecutive tiles
        'spaceBounds': spaceBounds, #spatial boundaries of crop
        'blockSize': blockSize, #number of hours in one time block
        'avgSize': avgSize, #number of hours to be averaged
        'windowSize': windowSize, #number of blocks in a ConvLSTM window
        'randSeed': randSeed, #random seed used
        'batchSize': batchSize, #size of an ML batch
        'trainLength': trainLength, #number of consecutive blocks in a section of train data
        'valLength': valLength, #number of consecutive blocks in a section of validation data
        'threshold': threshold, #minimum percentage of non-cloud data needed for a window to be included
        'dataDir': dataDir,
        'numBlocks': numBlocks,
        'numTiles': numTiles
    }

my_params = params()

start1 = time.time()
os.system(f"mkdir {my_params['dataDir']}")
os.system(f"mkdir {my_params['dataDir']}/raw_data")
os.system(f"mkdir {my_params['dataDir']}/full_data")
os.system(f"mkdir {my_params['dataDir']}/cloud_data")
os.system(f"mkdir {my_params['dataDir']}/masked_data")

raw_data(my_params)
print("Got raw data:", str(time.time() - start1))
start2 = time.time()
full_data(my_params)
print("Got full data:", str(time.time() - start2))
start3 = time.time()
masked_data(my_params)
print("Got masked data:", str(time.time() - start3))
start4 = time.time()
ids(my_params)
print("Got ids:", str(time.time() - start4))

data_dir = my_params['dataDir']
for x in my_params:
    my_params[x] = [my_params[x]]
params_df = pd.DataFrame.from_dict(my_params)
params_df.to_csv(f"{data_dir}/data_params.csv")
print("Total elapsed:", str(time.time() - start1))
