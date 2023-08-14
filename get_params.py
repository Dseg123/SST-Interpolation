import datetime

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
    startTime = "2019-01-01T00:00:00Z"
    endTime = "2019-06-01T00:00:00Z"
    dataDir = "data_size64_duration6_window2_4hourly_test"
    
    
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
