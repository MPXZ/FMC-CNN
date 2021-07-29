import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Flatten, Conv2D, MaxPooling2D,BatchNormalization, Activation, Concatenate, Dropout
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Layer
from tensorflow.keras import regularizers
from tensorflow import keras

import numpy as np
from numpy import loadtxt


# load dataset
def load_csv(filename):
    data=loadtxt(filename, delimiter=',')
    return(data)


#Building the parall sub-groups
def build_list(dataset, N):
    Parall_list = []
    for k in range(N):
        bloc = dataset[:, k * N:(k + 1) * N, :]
        bloc = np.expand_dims(bloc, axis=3)
        Parall_list.append(bloc)
    return (Parall_list)



#parallel the dataset and build model

inputs_list = [Input(shape=(64,1700,1)) for i in range(64)]
#regularization parameters

values = [0.6,0.7,0.8]
all_train = []
all_test = []
for param in values:
    conv_1_list = [Conv2D(filters=4, kernel_size=3, padding ='same', activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.2))(input_tensor) for input_tensor in inputs_list]
    max_1_list = [MaxPooling2D(pool_size=(2,2))(cov_tensor) for cov_tensor in conv_1_list]

    #concatenate all parallel processing unit
    parall=Concatenate()(max_1_list)

    x = Flatten()(parall)
    x = Dense(100, activation='relu')(x)
    x = Dropout(param)(x)
    outputs = Dense(3)(x)

    FMC_model = Model(inputs=inputs_list, outputs=outputs)


    FMC_model.compile(
        loss=keras.losses.CategoricalCrossentropy(from_logits=True),
        optimizer=keras.optimizers.Adam(lr=1e-3),
        metrics=['accuracy'],
    )
    #fit model
    FMC_model.fit(Train_list,y_train,batch_size=20,epochs=15, verbose=2)
    #evaluate model
    _, train_acc = FMC_model.evaluate(Train_list,y_train, verbose=0)
    _, test_acc = FMC_model.evaluate(Test_list,y_test, verbose=0)

    print('Param: %f, Train: %.3f, Test: %.3f' % (param, train_acc, test_acc))
    all_train.append(train_acc)
    all_test.append(test_acc)

