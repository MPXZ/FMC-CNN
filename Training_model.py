import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D,BatchNormalization, Activation, Concatenate
from tensorflow.keras.models import Model
from tensorflow import keras
from DIM_model import Encoder, GlobalDIM
import tensorflow as tf
import time
import numpy as np
from numpy import loadtxt

def MI_loss(positive, negative, N):

    '''
        Define loss function for model training
        Maximize mutual information between encoded embeddings with positive features by computing binary cross-entropy loss
        loss calculated from both positive score map and negative score map
        Loss is averaged over the parall block N
    '''

    real = tf.nn.sigmoid_cross_entropy_with_logits(logits=positive, labels=tf.ones_like(positive))
    fake = tf.nn.sigmoid_cross_entropy_with_logits(logits=negative, labels=tf.zeros_like(negative))
    loss = real + fake
    re_loss = tf.reduce_mean(loss, axis=1)/N
    return (re_loss)


def Network_DIM(FMC_shape, encoding_dim, N):

    '''
        Define a model that train encoder and discriminator together
        Input is positive dataset and negative dataset
        Output is positive and negative score map
    '''

    encoder_input_list = [Input(shape=FMC_shape) for i in range(N)]
    conv_list, encoded = Encoder(encoder_input_list, encoding_dim)
    encoder_model = Model(inputs=encoder_input_list, outputs=[conv_list, encoded], name='encoder')
    encoder_model.summary()

    xp_inputs_list = [Input(shape=(32, 1600, 1)) for i in range(32)]
    xn_inputs_list = [Input(shape=(32, 1600, 1)) for i in range(32)]
    conv_m, y_p = encoder_model(xp_inputs_list)
    conv_f, y_n = encoder_model(xn_inputs_list)

    global_p_feature = []
    global_n_feature = []
    for unitp in conv_m:
        convp = Flatten()(unitp)
        y_m = tf.concat([y_p, convp], axis=1)
        global_p_feature.append(y_m)

    for unitn in conv_f:
        convn = Flatten()(unitn)
        y_n = tf.concat([y_p, convn], axis=1)
        global_n_feature.append(y_n)

    linear_dim = global_p_feature[0].shape[1]

    M_input_list = [Input(shape=(linear_dim,)) for i in range(32)]
    glob_score = GlobalDIM(M_input_list)
    GlobalDis = Model(inputs=M_input_list, outputs=glob_score, name='Discriminator')
    GlobalDis.summary()

    g_p_score = GlobalDis(global_p_feature)
    g_n_score = GlobalDis(global_n_feature)

    DIM_model = Model(inputs=[xp_inputs_list,xn_inputs_list], outputs=[g_p_score, g_n_score])
    DIM_model.summary()

    return(DIM_model)


def Training_model(trainXp, trainXn, epochs, batch_size, encoding_dim, lr, N):

    model = Network_DIM(FMC_shape=(32,1600,1), encoding_dim=encoding_dim, N=32)
    numUpdates = int(trainXp.shape[0] / batch_size)

    def step(x_p_train, x_n_train):
        opts = keras.optimizers.Adam(lr=lr)
        # keep track of our gradients
        with tf.GradientTape() as tape:
            # make a prediction using the model and then calculate the loss
            pos, neg = model([x_p_train, x_n_train])
            loss_fn = MI_loss(pos, neg, N)

        # calculate the gradients using our tape and then update the model weights
        grads = tape.gradient(loss_fn, model.trainable_variables)
        opts.apply_gradients(zip(grads, model.trainable_variables))
        return (loss_fn)

    epoch_loss = []
    for epoch in range(epochs):
        print("\nStart of epoch %d" % (epoch,))
        epochStart = time.time()

        running_loss = 0
        for i in range(0, numUpdates):
            # determine starting and ending slice indexes for the current batch
            start = i * batch_size
            end = start + batch_size
            batch_train_p = trainXp[start:end]
            batch_train_n = trainXn[start:end]

            trainp_list = []
            trainn_list = []

            # arrange parall input list for both positive and negative samples in the batch
            for k in range(32):
                bloc = batch_train_p[:, k * 32:(k + 1) * 32, :]
                sec = batch_train_n[:, k * 32:(k + 1) * 32, :]
                bloc = np.expand_dims(bloc, axis=3)
                sec = np.expand_dims(sec, axis=3)
                trainp_list.append(bloc)
                trainn_list.append(sec)

            #calculate the batch loss and add batch loss into the epoch loss
            batch_loss = step(trainp_list, trainn_list)
            running_loss += tf.reduce_mean(batch_loss)

        #epoch loss accumulation
        total_loss = running_loss / numUpdates
        epoch_loss.append(total_loss)

        epochEnd = time.time()
        elapsed = (epochEnd - epochStart) / 60.0

        print("took {:.4} minutes".format(elapsed))
        print("training loss: %.2fs" % total_loss)

    # save only encoder in the model
    encoder = model.layers[64]
    encoder.save('encoder_model')

    return(epoch_loss)

