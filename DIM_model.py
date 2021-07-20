import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPooling2D,BatchNormalization, Activation, Concatenate
from tensorflow import keras


def Encoder(inputs_list, encoding_dim):

    ''' Define the network mapping FMC data to embeddings '''

    conv_1_list = [Conv2D(filters=4, kernel_size=3, padding='same', activation='relu')(input_tensor) for input_tensor in
                   inputs_list]

    max_1_list = [MaxPooling2D(pool_size=(2, 2))(cov_tensor) for cov_tensor in conv_1_list]
    parall = Concatenate()(max_1_list)

    x = Conv2D(32, (3, 3), padding="same")(parall)
    x = BatchNormalization()(x)
    x = Activation("relu")(x)

    fc = Flatten()(x)
    Encoded = Dense(encoding_dim)(fc)

    return (conv_1_list, Encoded)


def GlobalDIM(g_input_list):

    '''
        Define the network that integrate information from concatenated input list
        The parall block of features are concatenated with encoded features one by one
        the output is the score map for each parall block
    '''

    gm1 = [Dense(512)(g_tensor) for g_tensor in g_input_list]
    gm2 = [Activation("relu")(g_tensor) for g_tensor in gm1]

    gm3 = [Dense(512)(g_tensor) for g_tensor in gm2]
    f = [Activation("relu")(g_tensor) for g_tensor in gm3]
    f = [Dense(1)(f_tensor) for f_tensor in f]
    glo = Concatenate()(f)

    return (glo)

def LocalDIM(linear_dim, m_input, y_input):
    conv1 = Conv2D(filters=linear_dim, kernel_size=1, use_bias=False)(m_input)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation("relu")(conv1)

    conv2 = Conv2D(filters=linear_dim, kernel_size=1, use_bias=True)(m_input)
    conv2 = Activation("relu")(conv2)

    # global
    g1 = Dense(linear_dim)(y_input)
    g1 = BatchNormalization()(g1)
    g1 = Activation("relu")(g1)
    g1 = Dense(linear_dim)(g1)

    g2 = Dense(linear_dim)(y_input)
    g2 = Activation("relu")(g2)

    l = keras.layers.Add()([conv1, conv2])
    l = BatchNormalization()(l)

    g = keras.layers.Add()([g1, g2])

    return (g, l)


