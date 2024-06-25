
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
plt.style.use('ggplot')

import xgboost as xgb

import pickle
import gc
import os
import sys

from keras import applications
from keras import backend as K
from keras import layers
from keras import models
from keras import optimizers
import tensorflow as tf


def BaseMetrics(y_pred,y_true):
    TP = np.sum( (y_pred == 1) & (y_true == 1) )
    TN = np.sum( (y_pred == 0) & (y_true == 0) )
    FP = np.sum( (y_pred == 1) & (y_true == 0) )
    FN = np.sum( (y_pred == 0) & (y_true == 1) )
    return TP, TN, FP, FN


    
def SimpleAccuracy(y_pred,y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
    ACC = ( TP + TN ) / ( TP + TN + FP + FN )
    return ACC

def SimpleMetrics(y_pred,y_true):
    TP, TN, FP, FN = BaseMetrics(y_pred,y_true)
    ACC = ( TP + TN ) / ( TP + TN + FP + FN )
    
    # Reporting
    from IPython.display import display
    print( 'Confusion Matrix')
    display( pd.DataFrame( [[TN,FP],[FN,TP]], columns=['Pred 0','Pred 1'], index=['True 0', 'True 1'] ) )
    print( 'Accuracy : {}'.format( ACC ))

    
def get_data_batch(train, batch_size, seed=0):
    # # for random sampling - some samples will have excessively low or high sampling, but easy to implement
    # np.random.seed(seed)
    # x = train.loc[ np.random.choice(train.index, batch_size) ].values
    
    # iterating through shuffled indices, so every sample gets covered evenly
    start_i = (batch_size * seed) % len(train)
    stop_i = start_i + batch_size
    shuffle_seed = (batch_size * seed) // len(train)
    np.random.seed(shuffle_seed)
    train_ix = np.random.choice( list(train.index), replace=False, size=len(train) ) # wasteful to shuffle every time
    train_ix = list(train_ix) + list(train_ix) # duplicate to cover ranges past the end of the set
    x = train.loc[ train_ix[ start_i: stop_i ] ].values
    
    return np.reshape(x, (batch_size, -1) )


    
def CheckAccuracy( x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2 ):

    # Slightly slower code to create dataframes to feed into the xgboost dmatrix formats
 

    dtrain = np.vstack( [ x[:int(len(x)/2)], g_z[:int(len(g_z)/2)] ] ) # Use half of each real and generated set for training
    dlabels = np.hstack( [ np.zeros(int(len(x)/2)), np.ones(int(len(g_z)/2)) ] ) # synthetic labels
    dtest = np.vstack( [ x[int(len(x)/2):], g_z[int(len(g_z)/2):] ] ) # Use the other half of each set for testing
    y_true = dlabels # Labels for test samples will be the same as the labels for training samples, assuming even batch sizes
    
    dtrain = xgb.DMatrix(dtrain, dlabels, feature_names=data_cols+label_cols)
    dtest = xgb.DMatrix(dtest, feature_names=data_cols+label_cols)
    
    xgb_params = {
        # 'tree_method': 'hist', # for faster evaluation
        'max_depth': 4, # for faster evaluation
        'objective': 'binary:logistic',
        'random_state': 0,
        'eval_metric': 'auc', # allows for balanced or unbalanced classes 
        }
    xgb_test = xgb.train(xgb_params, dtrain, num_boost_round=10) # limit to ten rounds for faster evaluation

    y_pred = np.round(xgb_test.predict(dtest))


    return SimpleAccuracy(y_pred, y_true) # assumes balanced real and generated datasets
    
def PlotData( x, g_z, data_cols, label_cols=[], seed=0, with_class=False, data_dim=2, save=False, prefix='' ):
    
    real_samples = pd.DataFrame(x, columns=data_cols+label_cols)
    gen_samples = pd.DataFrame(g_z, columns=data_cols+label_cols)
    
    f, axarr = plt.subplots(1, 2, figsize=(6,2) )
    if with_class:
        axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]], c=real_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
        axarr[1].scatter( gen_samples[ data_cols[0]], gen_samples[ data_cols[1]], c=gen_samples[label_cols[0]]/2 ) #, cmap='plasma'  )
        
        
    else:
        axarr[0].scatter( real_samples[data_cols[0]], real_samples[data_cols[1]]) #, cmap='plasma'  )
        axarr[1].scatter( gen_samples[data_cols[0]], gen_samples[data_cols[1]]) #, cmap='plasma'  )
    axarr[0].set_title('real')
    axarr[1].set_title('generated')   
    axarr[0].set_ylabel(data_cols[1]) # Only add y label to left plot
    for a in axarr: a.set_xlabel(data_cols[0]) # Add x label to both plots
    axarr[1].set_xlim(axarr[0].get_xlim()), axarr[1].set_ylim(axarr[0].get_ylim()) # Use axes ranges from real data for generated data
    
    if save:
        plt.save( prefix + '.xgb_check.png' )
        
    plt.show()

    

#### The Functions to define the layers of the networks used in the 'define_models' functions below
    
def generator_network(x, data_dim, base_n_count): 
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    x = layers.Dense(data_dim)(x)    
    return x
    
def generator_network_w_label(x, labels, data_dim, label_dim, base_n_count): 
    x = layers.concatenate([x,labels])
    x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
    x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    x = layers.Dense(data_dim)(x)    
    x = layers.concatenate([x,labels])
    return x
    
def discriminator_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count, activation='relu')(x)
    x = layers.Dense(1, activation='sigmoid')(x)
    # x = layers.Dense(1)(x)
    return x
    
def critic_network(x, data_dim, base_n_count):
    x = layers.Dense(base_n_count*4, activation='relu')(x)
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count*2, activation='relu')(x) # 2
    # x = layers.Dropout(0.1)(x)
    x = layers.Dense(base_n_count*1, activation='relu')(x) # 1
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(base_n_count*4, activation='relu')(x) # extra
    # x = layers.Dense(1, activation='sigmoid')(x)
    x = layers.Dense(1)(x)
    return x

    
    
#### The functions to define the keras network models    
    
def define_models_GAN(rand_dim, data_dim, base_n_count, type=None):
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    generated_image_tensor = generator_network(generator_input_tensor, data_dim, base_n_count)

    generated_or_real_image_tensor = layers.Input(shape=(data_dim,))
    
    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim, base_n_count)
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim, base_n_count)

    generator_model = models.Model(inputs=[generator_input_tensor], outputs=[generated_image_tensor], name='generator')
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model(generator_input_tensor))
    combined_model = models.Model(inputs=[generator_input_tensor], outputs=[combined_output], name='combined')
    
    return generator_model, discriminator_model, combined_model

def define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type=None):
    generator_input_tensor = layers.Input(shape=(rand_dim, ))
    labels_tensor = layers.Input(shape=(label_dim,)) # updated for class
    generated_image_tensor = generator_network_w_label(generator_input_tensor, labels_tensor, data_dim, label_dim, base_n_count) # updated for class

    generated_or_real_image_tensor = layers.Input(shape=(data_dim + label_dim,)) # updated for class
    
    if type == 'Wasserstein':
        discriminator_output = critic_network(generated_or_real_image_tensor, data_dim + label_dim, base_n_count) # updated for class
    else:
        discriminator_output = discriminator_network(generated_or_real_image_tensor, data_dim + label_dim, base_n_count) # updated for class

    generator_model = models.Model(inputs=[generator_input_tensor, labels_tensor], outputs=[generated_image_tensor], name='generator') # updated for class
    discriminator_model = models.Model(inputs=[generated_or_real_image_tensor],
                                       outputs=[discriminator_output],
                                       name='discriminator')

    combined_output = discriminator_model(generator_model([generator_input_tensor, labels_tensor])) # updated for class
    combined_model = models.Model(inputs=[generator_input_tensor, labels_tensor], outputs=[combined_output], name='combined') # updated for class
    
    return generator_model, discriminator_model, combined_model



#### The functions specific to the WGAN architecture
#### Train discrimnator step is separated out to facilitate pre-training of the discriminator by itself

def em_loss(y_coefficients, y_pred):
    # define earth mover distance (wasserstein loss)
    # literally the weighted average of the critic network output
    # this is defined separately so it can be fed as a loss function to the optimizer in the WGANs
    return tf.reduce_mean(tf.multiply(y_coefficients, y_pred))

def train_discriminator_step(model_components, seed=0):
    
    [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path,

                        sess, _z, _x, _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
                        _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer,
                        show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses
                        ] = model_components
    
    if with_class:
        d_l_g, d_l_r, _ = sess.run([_disc_loss_generated, _disc_loss_real, disc_optimizer], feed_dict={
            _z: np.random.normal(size=(batch_size, rand_dim)),
            _x: get_data_batch(train, batch_size, seed=seed),
            _labels: get_data_batch(train, batch_size, seed=seed)[:,-label_dim:], # .reshape(-1,label_dim), # updated for class            
            epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
        })
    else:
        d_l_g, d_l_r, _ = sess.run([_disc_loss_generated, _disc_loss_real, disc_optimizer], feed_dict={
            _z: np.random.normal(size=(batch_size, rand_dim)),
            _x: get_data_batch(train, batch_size, seed=seed),
            epsilon: np.random.uniform(low=0.0, high=1.0, size=(batch_size, 1))
        })
    return d_l_g, d_l_r

def training_steps_WGAN(model_components):
    
    [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path,

                        sess, _z, _x, _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
                        _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer,
                        show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses
                        ] = model_components
    
    for i in range(starting_step, starting_step+nb_steps):
        K.set_learning_phase(1) # 1 = train

        # train the discriminator
        for j in range(k_d):
            d_l_g, d_l_r = train_discriminator_step(model_components, seed=i+j)
        disc_loss_generated.append(d_l_g)
        disc_loss_real.append(d_l_r)

        # train the generator
        for j in range(k_g):
            np.random.seed(i+j)
            z = np.random.normal(size=(batch_size, rand_dim))
            if with_class:
                labels = get_data_batch(train, batch_size, seed=i+j)[:,-label_dim:] # updated for class
                loss = combined_model.train_on_batch([z, labels], [-np.ones(batch_size)]) # updated for class
            else:
                loss = combined_model.train_on_batch(z, [-np.ones(batch_size)])
        combined_loss.append(loss)

        # To determine xgb loss each step, after training generator and discriminator
        if not i % 10: # 2x faster than testing each step...
            K.set_learning_phase(0) # 0 = test
            test_size = 492 # test using all of the actual fraud data
            x = get_data_batch(train, test_size, seed=i)
            z = np.random.normal(size=(test_size, rand_dim))
            if with_class:
                labels = x[:,-label_dim:]
                g_z = generator_model.predict([z, labels])
            else:
                g_z = generator_model.predict(z)
            xgb_loss = CheckAccuracy( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim )
            xgb_losses = np.append(xgb_losses, xgb_loss)
        
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            
            
            if show:
                PlotData( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim, 
                            save=False, prefix= data_dir + cache_prefix + '_' + str(i) )

            # save model checkpoints
            model_checkpoint_base_name = data_dir + cache_prefix + '_{}_model_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))
            pickle.dump([combined_loss, disc_loss_generated, disc_loss_real, xgb_losses], 
                open( data_dir + cache_prefix + '_losses_step_{}.pkl'.format(i) ,'wb'))
    
    return [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]

def adversarial_training_WGAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

    [rand_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ] = arguments
            
    np.random.seed(seed)     # set random seed
    
    data_dim = len(data_cols)
    print('data_dim: ', data_dim)
    print('data_cols: ', data_cols)
    
    label_dim = 0
    with_class = False
    if len(label_cols) > 0: 
        with_class = True
        label_dim = len(label_cols)
        print('label_dim: ', label_dim)
        print('label_cols: ', label_cols)
    
    # define network models
    
    K.set_learning_phase(1) # 1 = train
    
    if with_class:
        cache_prefix = 'WCGAN'
        generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count, type='Wasserstein')
    else:
        cache_prefix = 'WGAN'
        generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count, type='Wasserstein')
    
    
    # construct computation graph for calculating the gradient penalty (improved WGAN) and training the discriminator

    _z = tf.placeholder(tf.float32, shape=(batch_size, rand_dim))
    
    _labels = None
    if with_class:  
        _x = tf.placeholder(tf.float32, shape=(batch_size, data_dim + label_dim))    
        _labels = tf.placeholder(tf.float32, shape=(batch_size, label_dim)) # updated for class
        _g_z = generator_model(inputs=[_z, _labels]) # updated for class    
    else:      
        _x = tf.placeholder(tf.float32, shape=(batch_size, data_dim))
        _g_z = generator_model(_z)
    
    epsilon = tf.placeholder(tf.float32, shape=(batch_size, 1))
    
    x_hat = epsilon * _x + (1.0 - epsilon) * _g_z
    gradients = tf.gradients(discriminator_model(x_hat), [x_hat])
    _gradient_penalty = 10.0 * tf.square(tf.norm(gradients[0], ord=2) - 1.0)

    # calculate discriminator's loss
    _disc_loss_generated = em_loss(tf.ones(batch_size), discriminator_model(_g_z))
    _disc_loss_real = em_loss(tf.ones(batch_size), discriminator_model(_x))
    _disc_loss = _disc_loss_generated - _disc_loss_real + _gradient_penalty

    # update f by taking an SGD step on mini-batch loss LD(f)
    disc_optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(
        _disc_loss, var_list=discriminator_model.trainable_weights)

    sess = K.get_session()

    # compile models

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)

    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss=[em_loss])

    combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
    
    model_components = [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path,

                        sess, _z, _x, _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
                        _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer,
                        show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses
                        ]

    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    if loss_pickle_path:
        print('Loading loss pickles')
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        generator_model.load_weights(generator_model_path) #, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        discriminator_model.load_weights(discriminator_model_path) #, by_name=True)
    else:
        print('pre-training the critic...')
        K.set_learning_phase(1) # 1 = train
        for i in range(critic_pre_train_steps):
            if i%20==0:
                print('Step: {} of {} critic pre-training.'.format(i, critic_pre_train_steps))
            loss = train_discriminator_step(model_components, seed=i)
        print('Last batch of critic pre-training disc_loss: {}.'.format(loss))

    model_components = [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path,

                        sess, _z, _x, _labels, _g_z, epsilon, x_hat, gradients, _gradient_penalty,
                        _disc_loss_generated, _disc_loss_real, _disc_loss, disc_optimizer,
                        show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses
                        ]
        
    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_WGAN(model_components)
   


#### The functions specific to the vanilla GAN architecture   
        
def training_steps_GAN(model_components):
    
    [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ] = model_components  
    
    for i in range(starting_step, starting_step+nb_steps):
        K.set_learning_phase(1) # 1 = train

        # train the discriminator
        for j in range(k_d):
            np.random.seed(i+j)
            z = np.random.normal(size=(batch_size, rand_dim))
            x = get_data_batch(train, batch_size, seed=i+j)
            
            if with_class:
                labels = x[:,-label_dim:]
                g_z = generator_model.predict([z, labels])
            else:
                g_z = generator_model.predict(z)
#             x = np.vstack([x,g_z]) # code to train the discriminator on real and generated data at the same time, but you have to run network again for separate losses
#             classes = np.hstack([np.zeros(batch_size),np.ones(batch_size)])
#             d_l_r = discriminator_model.train_on_batch(x, classes)
            
            d_l_r = discriminator_model.train_on_batch(x, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
            d_l_g = discriminator_model.train_on_batch(g_z, np.random.uniform(low=0.0, high=0.001, size=batch_size)) # 0.0, 0.3 # GANs need noise to prevent loss going to zero
            # d_l_r = discriminator_model.train_on_batch(x, np.ones(batch_size)) # without noise
            # d_l_g = discriminator_model.train_on_batch(g_z, np.zeros(batch_size)) # without noise
        disc_loss_real.append(d_l_r)
        disc_loss_generated.append(d_l_g)
        
        # train the generator
        for j in range(k_g):
            np.random.seed(i+j)
            z = np.random.normal(size=(batch_size, rand_dim))
            if with_class:
                # loss = combined_model.train_on_batch([z, labels], np.ones(batch_size)) # without noise
                loss = combined_model.train_on_batch([z, labels], np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
            else:
                # loss = combined_model.train_on_batch(z, np.ones(batch_size)) # without noise
                loss = combined_model.train_on_batch(z, np.random.uniform(low=0.999, high=1.0, size=batch_size)) # 0.7, 1.2 # GANs need noise to prevent loss going to zero
        combined_loss.append(loss)
        
        # Determine xgb loss each step, after training generator and discriminator
        if not i % 10: # 2x faster than testing each step...
            K.set_learning_phase(0) # 0 = test
            test_size = 492 # test using all of the actual fraud data
            x = get_data_batch(train, test_size, seed=i)
            z = np.random.normal(size=(test_size, rand_dim))
            if with_class:
                labels = x[:,-label_dim:]
                g_z = generator_model.predict([z, labels])
            else:
                g_z = generator_model.predict(z)
            xgb_loss = CheckAccuracy( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim )
            xgb_losses = np.append(xgb_losses, xgb_loss)

        # Saving weights and plotting images
        if not i % log_interval:
            print('Step: {} of {}.'.format(i, starting_step + nb_steps))
            K.set_learning_phase(0) # 0 = test
                        
            # loss summaries      
            print( 'Losses: G, D Gen, D Real, Xgb: {:.4f}, {:.4f}, {:.4f}, {:.4f}'.format(combined_loss[-1], disc_loss_generated[-1], disc_loss_real[-1], xgb_losses[-1]) )
            print( 'D Real - D Gen: {:.4f}'.format(disc_loss_real[-1]-disc_loss_generated[-1]) )            
            # print('Generator model loss: {}.'.format(combined_loss[-1]))
            # print('Discriminator model loss gen: {}.'.format(disc_loss_generated[-1]))
            # print('Discriminator model loss real: {}.'.format(disc_loss_real[-1]))
            # print('xgboost accuracy: {}'.format(xgb_losses[-1]) )
            
            if show:
                PlotData( x, g_z, data_cols, label_cols, seed=0, with_class=with_class, data_dim=data_dim, 
                            save=False, prefix= data_dir + cache_prefix + '_' + str(i) )
            
            # save model checkpoints
            model_checkpoint_base_name = data_dir + cache_prefix + '_{}_model_weights_step_{}.h5'
            generator_model.save_weights(model_checkpoint_base_name.format('generator', i))
            discriminator_model.save_weights(model_checkpoint_base_name.format('discriminator', i))
            pickle.dump([combined_loss, disc_loss_generated, disc_loss_real, xgb_losses], 
                open( data_dir + cache_prefix + '_losses_step_{}.pkl'.format(i) ,'wb'))
    
    return [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses]
    
def adversarial_training_GAN(arguments, train, data_cols, label_cols=[], seed=0, starting_step=0):

    [rand_dim, nb_steps, batch_size, 
             k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
            data_dir, generator_model_path, discriminator_model_path, loss_pickle_path, show ] = arguments
    
    np.random.seed(seed)     # set random seed
    
    data_dim = len(data_cols)
    print('data_dim: ', data_dim)
    print('data_cols: ', data_cols)
    
    label_dim = 0
    with_class = False
    if len(label_cols) > 0: 
        with_class = True
        label_dim = len(label_cols)
        print('label_dim: ', label_dim)
        print('label_cols: ', label_cols)
    
    # define network models
    
    K.set_learning_phase(1) # 1 = train
    
    if with_class:
        cache_prefix = 'CGAN'
        generator_model, discriminator_model, combined_model = define_models_CGAN(rand_dim, data_dim, label_dim, base_n_count)
    else:
        cache_prefix = 'GAN'
        generator_model, discriminator_model, combined_model = define_models_GAN(rand_dim, data_dim, base_n_count)
    
    # compile models

    adam = optimizers.Adam(lr=learning_rate, beta_1=0.5, beta_2=0.9)

    generator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.compile(optimizer=adam, loss='binary_crossentropy')
    discriminator_model.trainable = False
    combined_model.compile(optimizer=adam, loss='binary_crossentropy')
    
    if show:
        print(generator_model.summary())
        print(discriminator_model.summary())
        print(combined_model.summary())

    combined_loss, disc_loss_generated, disc_loss_real, xgb_losses = [], [], [], []
    
    if loss_pickle_path:
        print('Loading loss pickles')
        [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = pickle.load(open(loss_pickle_path,'rb'))
    if generator_model_path:
        print('Loading generator model')
        generator_model.load_weights(generator_model_path, by_name=True)
    if discriminator_model_path:
        print('Loading discriminator model')
        discriminator_model.load_weights(discriminator_model_path, by_name=True)

    model_components = [ cache_prefix, with_class, starting_step,
                        train, data_cols, data_dim,
                        label_cols, label_dim,
                        generator_model, discriminator_model, combined_model,
                        rand_dim, nb_steps, batch_size, 
                        k_d, k_g, critic_pre_train_steps, log_interval, learning_rate, base_n_count,
                        data_dir, generator_model_path, discriminator_model_path, show,
                        combined_loss, disc_loss_generated, disc_loss_real, xgb_losses ]
        
    [combined_loss, disc_loss_generated, disc_loss_real, xgb_losses] = training_steps_GAN(model_components)
        


#### Functions specific to the DRAGAN architecture
#### Note the DRAGAN is implemented in tensorflow without Keras libraries 
#### https://github.com/kodalinaveen3/DRAGAN
        
def sample_z(m, n): # updated to normal distribution
#     return np.random.uniform(-1., 1., size=[m, n])
    return np.random.normal(size=[m, n])

def xavier_init(size): # updated to uniform distribution using standard xavier formulation
#     in_dim = size[0]
#     xavier_stddev = 1. / tf.sqrt(in_dim / 2.)
#     return tf.random_normal(shape=size, stddev=xavier_stddev, seed=global_seed)
    xavier_range = tf.sqrt( 6 / ( size[0] + size[1] ) )
    return tf.random_uniform(shape=size, minval=-xavier_range, maxval=xavier_range)

def get_perturbed_batch(minibatch):
    return minibatch + 0.5 * minibatch.std() * np.random.random(minibatch.shape)

     
def G(z, G_W, G_b): # The Generator Network
    # for i in range(len(G_layer_dims)-2):
    for i in range(len(G_W)-1):
        z = tf.nn.relu(tf.matmul(z, G_W[i]) + G_b[i])
#         print(i,G_W[i],z)
    return tf.matmul(z, G_W[-1]) + G_b[-1]     
    
def D(x, D_W, D_b): # The Discriminator Network
    # for i in range(len(D_layer_dims)-2):
    for i in range(len(D_W)-1):
        x = tf.nn.relu(tf.matmul(x, D_W[i]) + D_b[i])
    return tf.nn.sigmoid(tf.matmul(x, D_W[-1]) + D_b[-1])
     
     
def define_DRAGAN_network( X_dim=2, h_dim=128, z_dim=2, lambda0=10, learning_rate=1e-4, mb_size=128, seed=0 ):
    
    X = tf.placeholder(tf.float32, shape=[None, X_dim], name='X' )
    X_p = tf.placeholder(tf.float32, shape=[None, X_dim], name='X_p' )
    z = tf.placeholder(tf.float32, shape=[None, z_dim], name='z' )

    D_layer_dims = [X_dim, h_dim*4, h_dim*2, h_dim, 1 ]
    D_W, D_b = [], []   
    for i in range(len(D_layer_dims)-1):
        D_W.append( tf.Variable( xavier_init([D_layer_dims[i], D_layer_dims[i+1]] ), name='D_W'+str(i) ) )
    #     D_W.append( tf.Variable(  initializer=tf.contrib.layers.xavier_initializer(seed=global_seed) ) # working towards using tf's own xavier initializer
        D_b.append( tf.Variable( tf.zeros(shape=[D_layer_dims[i+1]]), name='D_b'+str(i) ) )
    theta_D = D_W + D_b

    G_layer_dims = [z_dim, h_dim, h_dim*2, h_dim*4, X_dim ]
    G_W, G_b = [], []
    for i in range(len(G_layer_dims)-1):
        G_W.append( tf.Variable( xavier_init([G_layer_dims[i], G_layer_dims[i+1]] ), name='G_W'+str(i) ) )
        G_b.append( tf.Variable( tf.zeros(shape=[G_layer_dims[i+1]]), name='g_b'+str(i) ) )
    theta_G = G_W + G_b
    # print( theta_D + theta_G )
        
    G_sample = G(z, G_W, G_b)
    D_real = D(X, D_W, D_b)
    D_fake = D(G_sample, D_W, D_b)
    D_real_perturbed = D(X_p, D_W, D_b)

    # D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real, labels=tf.ones_like(D_real)))
    # D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.zeros_like(D_fake)))
    # disc_cost = D_loss_real + D_loss_fake 
    # gen_cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake, labels=tf.ones_like(D_fake)))
    D_loss_real = tf.reduce_mean(tf.log( D_real ))
    D_loss_fake = tf.reduce_mean(tf.log( 1 - D_fake ))
    disc_cost = - D_loss_real - D_loss_fake
    gen_cost = D_loss_fake

    #Gradient penalty
    alpha = tf.random_uniform(
        shape=[mb_size,1], 
        minval=0.,
        maxval=1.) # do not set seed
        
    differences = X_p - X
    interpolates = X + (alpha*differences)
    gradients = tf.gradients(D(interpolates, D_W, D_b), [interpolates])[0]
    # slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
    # gradient_penalty = tf.reduce_mean((slopes-1.)**2)
    gradient_penalty = tf.square(tf.norm(gradients, ord=2) - 1.0 )  # corrected?

    disc_cost += lambda0 * gradient_penalty / mb_size # corrected?

    G_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(gen_cost, var_list=theta_G)
    D_solver = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.5, beta2=0.9).minimize(disc_cost, var_list=theta_D)
    
    return [ D_solver, disc_cost, D_loss_real, D_loss_fake,
                X, X_p, z,
                G_solver, gen_cost, G_sample ]
    


 
# End of function list