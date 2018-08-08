import tensorflow as tf
import numpy as np
import pandas as pd
import math

def autoencoder(dimensions=[278, 140, 70, 30]):
    """
    Build a stacked deep autoencoder with tied weights, that is w = wT.

    return a dict.
    
    Parameters
    ----------
    dimensions : list, optional
        The number of neurons for each layer of the autoencoder.
    Returns
    -------
    x : Tensor
        Input placeholder to the network
    z : Tensor
        Inner-most latent representation
    y : Tensor
        Output reconstruction of the input
    loss : Tensor
        Overall cost to use for training
    """    
    # input to the network
    x = tf.placeholder(tf.float32, [None, dimensions[0]], name='x')
    current_input = x
    print ('current_input', current_input.shape)
    #---------------------------
    # Build the encoder
    #---------------------------
    encoder = [] # for putting the weight of encoder, w1,w2,..
    for layer_i, n_output in enumerate(dimensions[1:]):
        #print ('layer_i-encoder', layer_i)
        #print ('n_output-encoder', n_output)
        n_input = int(current_input.get_shape()[1]) # [0]: batch_szie, [1]:input_dim
        #print ('n_input', n_input)
        W = tf.Variable(
            tf.random_uniform([n_input, n_output],
                              minval = -1.0 / math.sqrt(n_input),
                              maxval = 1.0 / math.sqrt(n_input)))
        b = tf.Variable(tf.zeros([n_output]))
        # saving layer of encoding for decoder
        encoder.append(W)
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        # assign current_input
        current_input = output
    #---------------------------
    # latent representation (output of encoder)
    #---------------------------
    z = current_input
    encoder.reverse() # [...,w2,w1]
    
    #---------------------------
    # Build the decoder using the same weights
    #---------------------------
    for layer_i, n_output in enumerate(dimensions[:-1][::-1]):
        #print ('layer_i-decoder', layer_i)
        #print ('n_output-decoder', n_output)
        W = tf.transpose(encoder[layer_i])
        b = tf.Variable(tf.zeros([n_output]))
        output = tf.nn.tanh(tf.matmul(current_input, W) + b)
        # assign current_input
        current_input = output

    # now have the reconstruction through the network
    y = current_input
    # Define loss and, minimize the mean squared error
    loss = tf.reduce_mean(tf.pow(x - y, 2)) 

    return {'x': x, 'z': z, 'y': y, 'loss': loss}

def raw_feature_generator(batch_size = 128, shuffle = True, num_epochs = 10000, allow_smaller_final_batch = False):
    epoch_num = 0
    while epoch_num < num_epochs:
        print ('epoch_num : {}'.format(epoch_num))
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, length, batch_size):
            batch_idx = idx[i: i + batch_size]
            if not allow_smaller_final_batch and len(batch_idx) != batch_size:
                break # terminate the loop
            yield df.values[batch_idx]
        epoch_num += 1
    

# reload again for filling
df = pd.read_hdf('../features/base_featurs.h5','base_featurs')
print (df.shape)
copy_for_the_following_merge = df[['SK_ID_CURR','TARGET']].copy()
no_need_to_comoress = ['index','TARGET', 'SK_ID_CURR']
df.drop(no_need_to_comoress, axis = 1, inplace = True)
# handling with infinity
df.replace([np.inf, -np.inf], np.nan, inplace = True)
print (df.shape)

length = len(df)
print ('length',length)
idx = np.arange(length) # 1-D array
print ('idx', idx.shape, idx)

# display_step
display_step = 1000
# learning_rate
learning_rate = 0.001
# define auto-encoder network architecture
ae = autoencoder(dimensions=[278, 140, 70, 30])
# optimizer
optimizer = tf.train.RMSPropOptimizer(learning_rate).minimize(ae['loss'])

# Initialize the variables (i.e. assign their default value)
# We create a session to use the graph
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# Training
batch_size = 128
n_epochs = 10000
i = 0
for batch_xs in raw_feature_generator(batch_size = 128, num_epochs = 10000):
    i += 1
    print ('batch_xs', batch_xs.shape)
    #-------------------------
    # feature scaling: make different feature have the same scaling
    #-------------------------
    #train = np.array([img - mean_img for img in batch_xs])
    # train
    _, loss = sess.run([optimizer, ae['loss']], feed_dict={ae['x']: batch_xs})
    if i % display_step == 0 or i == 1:
        print('Step %i: Minibatch Loss: %f' % (i, loss))


