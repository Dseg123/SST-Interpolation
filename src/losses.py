
import numpy as np                 #for general calculations.
import tensorflow as tf
from tensorflow import keras

#both of form (120, 64, 64)

cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def mse_loss(y_true, y_pred):
    # shape = list(tf.shape(y_true).numpy())
    # shape.append(1)
    # tf.reshape(y_true, shape)
    # tf.reshape(y_pred, shape)
    invalid_data = tf.math.is_nan(y_true)
    valid_data = ~invalid_data
    
    num_valid = tf.math.reduce_sum(tf.cast(valid_data, 'float32'))
    num_invalid = tf.math.reduce_sum(tf.cast(invalid_data, 'float32'))
    num_tot = num_valid + num_invalid
    print(num_valid, num_invalid, num_tot)
    
    y_true_loss = tf.where(invalid_data, tf.zeros_like(y_true), y_true) #change all nans to 0's
    y_pred_loss = tf.where(invalid_data, tf.zeros_like(y_pred), y_pred)
    
    #sum_errors = num_tot * keras.losses.MSE(y_true_loss, y_pred_loss)
    sum_errors = tf.reduce_sum(tf.square(y_true_loss - y_pred_loss))
    mse = sum_errors/(num_valid + keras.backend.epsilon())
    return mse


def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


def interpolator_loss(y_true, y_pred, fake_output):
    lambda_mse = 0.5
    lambda_gan = 0.5
    m_l = mse_loss(y_true, y_pred)
    g_l = generator_loss(fake_output)
    tf.print("m_l", m_l, "g_l", g_l)
    return lambda_mse * mse_loss(y_true, y_pred) + lambda_gan * generator_loss(fake_output)