import numpy as np
import pandas as pd
from pathlib import Path
from src.models import *
from src.losses import *
from src.generators import *

from pykrige.ok3d import OrdinaryKriging3D
from pykrige.uk3d import UniversalKriging3D

def get_trained_ConvLSTM(experiment_dir):
    params = pd.read_csv(experiment_dir + '/model_params.csv')
    sample_size = params['src_sample_size'].iloc[0]
    tile_size = params['src_tile_size'].iloc[0]
    model = create_ConvLSTM(n_t = sample_size, tile_size = tile_size)
    p = Path(experiment_dir)
    pg = list(p.glob("*interpolator_weights*.h5"))
    pg.sort()
    
    model.load_weights(pg[-1])
    return model

def get_trained_ConvLSTM_Seeded(experiment_dir):
    params = pd.read_csv(experiment_dir + '/model_params.csv')
    
    n_t = params['src_sample_size'].iloc[0]
    tile_size = params['src_tile_size'].iloc[0]
    seed_size = params['src_seed_size'].iloc[0]
    filter_size = params['gan_size'].iloc[0]
    filter_depth = params['gan_depth'].iloc[0]
    
    model = create_ConvLSTM_Seeded(n_t = n_t, tile_size = tile_size, seed_size = seed_size, filter_size = filter_size, filter_depth = filter_depth)
    p = Path(experiment_dir)
    pg = list(p.glob("*interpolator_weights*.h5"))
    pg.sort()
    
    model.load_weights(pg[-1])
    return model

def evaluate_ConvLSTM(model, dataset):
    counter = 0
    tot = 0
    for batch in dataset:
        x, y = batch
        pred_outputs = tf.reshape(model(x, training=False), y.shape)

        interp_loss = mse_loss(y, pred_outputs)
        print(interp_loss)
        tot += interp_loss
        counter += 1
        
        if counter > 100:
            break
    return tot/counter

def evaluate_ConvLSTM_Seeded(model, dataset):
    src_seed_size = 100
    
    counter = 0
    tot = 0
    for batch in dataset:
        batch_x, batch_y = batch
        for i in range(batch_x.shape[0]):
            x = tf.reshape(batch_x[i, :, :, :], (1, batch_x.shape[1], batch_x.shape[2], batch_x.shape[3], 1))
            y = batch_y[i, :, :, :]
            
            noise = tf.random.normal([x.shape[0], src_seed_size])
            pred_outputs = tf.reshape(model([x, noise], training=False), y.shape)

            interp_loss = mse_loss(y, pred_outputs)
            print(interp_loss)
            tot += interp_loss
            counter += 1
        if counter > 100:
            break
    return tot/counter
 
def np_mse_loss(y_true, y_pred):
    mask = ~np.isnan(y_true)
    return np.mean(np.square(y_true[mask] - y_pred[mask]))

def evaluate_baseline(dataset):
    counter = 0
    tot = 0
    for batch in dataset:
        batch_x, batch_y = batch
        batch_x = batch_x.numpy()
        batch_y = batch_y.numpy()
        for ind in range(batch_x.shape[0]):
            x = batch_x[ind, :, :, :]
            y = batch_y[ind, :, :, :]
            
            locs = np.where(x != 0)

            gridx = np.arange(0.0, x.shape[1])
            gridy = np.arange(0.0, x.shape[2])
            gridz = np.arange(0.0, x.shape[0])
            arr = np.zeros((len(locs[0]), 4))

            data_arr = x[locs[0], locs[1], locs[2]]

            arr[:, 0] = locs[2]
            arr[:, 1] = locs[1]
            arr[:, 2] = locs[0]
            arr[:, 3] = data_arr
            uk3d = UniversalKriging3D(
                arr[:100, 0], arr[:100, 1], arr[:100, 2], arr[:100, 3], variogram_model="linear", verbose=False
            )
            print("Done")
            vals, ss3d = uk3d.execute("grid", gridx, gridy, gridz)
            interp_loss = np_mse_loss(y, vals)
            print(interp_loss)
            tot += interp_loss
            counter += 1
        if counter > 100:
            break
    return tot/counter
        

        
    