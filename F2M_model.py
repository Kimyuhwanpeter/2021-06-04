# -*- coding:utf-8 -*-
import tensorflow as tf

l2 = tf.keras.regularizers.l2

class InstanceNormalization(tf.keras.layers.Layer):
  #"""Instance Normalization Layer (https://arxiv.org/abs/1607.08022)."""

    def __init__(self, epsilon=1e-5):
        super(InstanceNormalization, self).__init__()
        self.epsilon = epsilon
    
    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=input_shape[-1:],
            initializer=tf.random_normal_initializer(0., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=input_shape[-1:],
            initializer='zeros',
            trainable=True)
    
    def call(self, x):
        mean, variance = tf.nn.moments(x, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (x - mean) * inv
        return self.scale * normalized + self.offset

def euclidean_dist(x, y):

    x_2 = tf.reduce_sum(tf.math.pow(x, 2), -1, keepdims=True)
    y_2 = tf.reduce_sum(tf.math.pow(y, 2), -1, keepdims=True)
    dists = x_2 - 2 * tf.matmul(x, tf.transpose(y, [1, 0])) + tf.transpose(y_2, [1, 0])
    return dists

def Encoder_Decoder(input_shape=(256, 256, 3), weight_decay=0.00005, batch_size=1):

    def featmap_2_pixwise(fmap):
        fmap = tf.transpose(fmap, [3, 1, 2, 0])
        fmap = tf.reshape(fmap, [fmap.shape[0], -1])
        fmap = tf.transpose(fmap, [1, 0])
        return fmap
    def pixwise_2_featmap_cont(pwise, fmap_shape):
        pwise = tf.transpose(pwise, [1, 0])
        pwise = tf.reshape(pwise, [fmap_shape[3], fmap_shape[1], fmap_shape[2], fmap_shape[0]])
        pwise = tf.transpose(pwise, [3, 1, 2, 0])
        return pwise
    def pixwise_2_featmap_style(pwise, fmap_shape):
        pwise = tf.transpose(pwise, [1, 0])
        pwise = tf.reshape(pwise, [fmap_shape[3], fmap_shape[1], fmap_shape[2], fmap_shape[0]])
        pwise = tf.transpose(pwise, [3, 1, 2, 0])
        return pwise

    def recompose_style_feature(style1, style2):
        dist_matrix = euclidean_dist(style1, style2) # KNN
        #topk_vals, topk_idxs = tf.math.top_k(dist_matrix, 5)    # Get the top-k (5)
        topk_vals = tf.math.top_k(tf.negative(dist_matrix), 5)
        topk_vals = tf.negative(topk_vals[0])

        style1_pixwise = tf.keras.layers.Dense(192, kernel_regularizer=l2(weight_decay))(style1)
        style1_pixwise = tf.keras.layers.BatchNormalization()(style1_pixwise)
        style1_pixwise = tf.keras.layers.ReLU()(style1_pixwise)
        style1_pixwise = tf.keras.layers.Dense(192 // 2, kernel_regularizer=l2(weight_decay))(style1_pixwise)
        style1_pixwise = tf.keras.layers.BatchNormalization()(style1_pixwise)
        style1_pixwise = tf.keras.layers.ReLU()(style1_pixwise)
        style1_pixwise = tf.expand_dims(style1_pixwise, 1)

        style2_pixwise = tf.keras.layers.Dense(192, kernel_regularizer=l2(weight_decay))(style2)
        style2_pixwise = tf.keras.layers.BatchNormalization()(style2_pixwise)
        style2_pixwise = tf.keras.layers.ReLU()(style2_pixwise)
        style2_pixwise = tf.keras.layers.Dense(192 // 2, kernel_regularizer=l2(weight_decay))(style2_pixwise)
        style2_pixwise = tf.keras.layers.BatchNormalization()(style2_pixwise)
        style2_pixwise = tf.keras.layers.ReLU()(style2_pixwise)
        style2_pixwise = tf.expand_dims(style2_pixwise, 1)

        att1 = tf.keras.layers.Conv1D(filters=1,
                                      kernel_size=192 // 2,
                                      padding="same",
                                      kernel_regularizer=l2(weight_decay))(style1_pixwise)
        att2 = tf.keras.layers.Conv1D(filters=1,
                                      kernel_size=192 // 2,
                                      padding="same",
                                      kernel_regularizer=l2(weight_decay))(style2_pixwise)

        att = att1 + tf.transpose(att2, [1, 0, 2])
        att = att[:, :, 0]

        knn_filter = tf.cast(tf.greater(dist_matrix, topk_vals[:, 5-1:5]), tf.float32) * 1e-9
        att = tf.nn.softmax(tf.nn.softplus(att)+knn_filter, -1)

        out_pixwise_style = tf.matmul(att, style2)

        return out_pixwise_style

    def recompose_cont_feature(cont1, cont2):
        dist_matrix = euclidean_dist(cont1, cont2) # KNN
        #topk_vals, topk_idxs = tf.math.top_k(dist_matrix, 5)    # Get the top-k (5)
        topk_vals = tf.math.top_k(tf.negative(dist_matrix), 5)
        topk_vals = tf.negative(topk_vals[0])

        cont1_pixwise = tf.keras.layers.Dense(64 // 2, kernel_regularizer=l2(weight_decay))(cont1)
        cont1_pixwise = tf.keras.layers.BatchNormalization()(cont1_pixwise)
        cont1_pixwise = tf.keras.layers.ReLU()(cont1_pixwise)
        cont1_pixwise = tf.keras.layers.Dense(64 // 4, kernel_regularizer=l2(weight_decay))(cont1_pixwise)
        cont1_pixwise = tf.keras.layers.BatchNormalization()(cont1_pixwise)
        cont1_pixwise = tf.keras.layers.ReLU()(cont1_pixwise)
        cont1_pixwise = tf.expand_dims(cont1_pixwise, 1)

        cont2_pixwise = tf.keras.layers.Dense(64 // 2, kernel_regularizer=l2(weight_decay))(cont2)
        cont2_pixwise = tf.keras.layers.BatchNormalization()(cont2_pixwise)
        cont2_pixwise = tf.keras.layers.ReLU()(cont2_pixwise)
        cont2_pixwise = tf.keras.layers.Dense(64 // 4, kernel_regularizer=l2(weight_decay))(cont2_pixwise)
        cont2_pixwise = tf.keras.layers.BatchNormalization()(cont2_pixwise)
        cont2_pixwise = tf.keras.layers.ReLU()(cont2_pixwise)
        cont2_pixwise = tf.expand_dims(cont2_pixwise, 1)

        att1 = tf.keras.layers.Conv1D(filters=1,
                                      kernel_size=64 // 4,
                                      padding="same",
                                      kernel_regularizer=l2(weight_decay))(cont1_pixwise)
        att2 = tf.keras.layers.Conv1D(filters=1,
                                      kernel_size=64 // 4,
                                      padding="same",
                                      kernel_regularizer=l2(weight_decay))(cont2_pixwise)

        att = att1 + tf.transpose(att2, [1, 0, 2])
        att = att[:, :, 0]

        knn_filter = tf.cast(tf.greater(dist_matrix, topk_vals[:, 5-1:5]), tf.float32) * 1e-9
        att = tf.nn.softmax(tf.nn.softplus(att)+knn_filter, -1)

        out_pixwise_style = tf.matmul(att, cont2)

        return out_pixwise_style

    def ResBlock(input, dim, weight_decay):

        h = tf.keras.layers.Conv2D(filters=dim // 4,
                                   kernel_size=1,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(input)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=dim // 4,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.ReLU()(h)

        h = tf.keras.layers.Conv2D(filters=dim,
                                   kernel_size=1,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)

        return tf.keras.layers.ReLU()(h + input)

    def style_ResBlock(input, dim, weight_decay):

        h = tf.keras.layers.Conv2D(filters=dim // 4,
                                   kernel_size=1,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(input)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.PReLU()(h)

        h = tf.keras.layers.Conv2D(filters=dim // 4,
                                   kernel_size=3,
                                   strides=1,
                                   padding="same",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)
        h = tf.keras.layers.PReLU()(h)

        h = tf.keras.layers.Conv2D(filters=dim,
                                   kernel_size=1,
                                   strides=1,
                                   padding="valid",
                                   use_bias=False,
                                   kernel_regularizer=l2(weight_decay))(h)
        h = InstanceNormalization()(h)

        return tf.keras.layers.PReLU()(h + input)

    def Conv2D(filters, kernel_size, strides, padding, use_bias, weight_decay):
        return tf.keras.layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding, use_bias=use_bias,
                                      kernel_regularizer=l2(weight_decay))

    h = inputs = tf.keras.Input(input_shape, batch_size=batch_size)
    h2 = inputs2 = tf.keras.Input(input_shape, batch_size=batch_size)

    dim = 64
    shared_1 = Conv2D(dim, 7, 1, "valid", False, weight_decay)
    shared_2 = Conv2D(dim*2, 3, 2, "same", False, weight_decay)
    shared_3 = Conv2D(dim*4, 3, 2, "same", False, weight_decay)
    ############################################################
    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = shared_1(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]

    h = shared_2(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]

    h = shared_3(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [64, 64, 256]
    ############################################################

    ############################################################
    h2 = tf.keras.layers.ZeroPadding2D((3,3))(h2)
    h2 = shared_1(h2)
    h2 = InstanceNormalization()(h2)
    h2 = tf.keras.layers.ReLU()(h2) # [256, 256, 64]

    h2 = shared_2(h2)
    h2 = InstanceNormalization()(h2)
    h2 = tf.keras.layers.ReLU()(h2)   # [128, 128, 128]

    h2 = shared_3(h2)
    h2 = InstanceNormalization()(h2)
    h2 = tf.keras.layers.ReLU()(h2)   # [64, 64, 256]
    ############################################################

    for i in range(9):
        h = ResBlock(h, dim*4, weight_decay)

    for i in range(2):
        h2 = style_ResBlock(h2, dim*4, weight_decay)

    ############################################################
    h_cont, h_style = h[:, :, :, :64], h[:, :, :, 64:] # 64, 192    
    h2_cont, h2_style = h2[:, :, :, :64], h2[:, :, :, 64:]    # 64, 192

    h_cont_ = featmap_2_pixwise(h_cont)
    h_style_ = featmap_2_pixwise(h_style)
    h2_cont_ = featmap_2_pixwise(h2_cont)
    h2_style_ = featmap_2_pixwise(h2_style)

    out_style = recompose_style_feature(h_style_, h2_style_)
    out_cont = recompose_cont_feature(h_cont_, h2_cont_)

    h3 = pixwise_2_featmap_cont(out_cont, h_cont.shape)
    h4 = pixwise_2_featmap_style(out_style, h_style.shape)

    outputs = h = tf.concat([h3, h4], -1)
    ############################################################

    for i in range(9):
        h = ResBlock(h, 256, weight_decay)  # [64, 64, 256]
    #h = tf.keras.layers.Conv2D(filters=dim * 4,
    #                            kernel_size=3,
    #                            strides=1,
    #                            padding="same",
    #                            use_bias=False,
    #                            kernel_regularizer=l2(weight_decay))(h)
    #h = InstanceNormalization()(h)
    #h = tf.keras.layers.ReLU()(h)

    h_split1, h_split2 = h[:, :, :, :64], h[:, :, :, 64:]   # 64, 192
    h_split1 = tf.expand_dims(tf.reduce_mean(h_split1, -1), 3)
    h_split2 = tf.expand_dims(tf.reduce_mean(h_split2, -1), 3)
    h_cont = h_cont * h_split1  # feature map의 contrast 강화
    h_style = h2_style * h_split2
    h = tf.concat([h_cont, h_style], -1)

    h = tf.keras.layers.Conv2DTranspose(filters=128,        # U-net처럼 작용하도록 만들고싶은데...
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [128, 128, 128]

    outputs1 = tf.keras.layers.UpSampling2D((2,2))(h)   # [256, 256, 128]
    outputs1 = tf.concat([outputs1, inputs], -1)
    outputs1 = tf.keras.layers.ZeroPadding2D((3,3))(outputs1)
    outputs1 = tf.keras.layers.Conv2D(filters=3,
                                      kernel_size=7,
                                      strides=1,
                                      padding="valid")(outputs1)   # [256, 256, 3]
    outputs1 = tf.nn.tanh(outputs1)
    
    outputs2 = tf.keras.layers.UpSampling2D((2,2))(h)  # [256, 256, 128]
    outputs2 = tf.concat([outputs2, inputs2], -1)
    outputs2 = tf.keras.layers.Conv2D(filters=64,
                                      kernel_size=3,
                                      strides=1,
                                      padding="same")(outputs2)   # [256, 256, 64]
    outputs2 = InstanceNormalization()(outputs2)
    outputs2 = tf.keras.layers.ReLU()(outputs2)
    outputs2 = tf.keras.layers.ZeroPadding2D((3,3))(outputs2)
    outputs2 = tf.keras.layers.Conv2D(filters=3,
                                      kernel_size=7,
                                      strides=1,
                                      padding="valid")(outputs2)   # [256, 256, 3]
    outputs2 = tf.nn.tanh(outputs2)

    h = tf.keras.layers.Conv2DTranspose(filters=64,
                                        kernel_size=3,
                                        strides=2,
                                        padding="same",
                                        use_bias=False,
                                        kernel_regularizer=l2(weight_decay))(h)
    h = InstanceNormalization()(h)
    h = tf.keras.layers.ReLU()(h)   # [256, 256, 64]
    h = tf.concat([h, inputs, inputs2], -1)

    h = tf.keras.layers.ZeroPadding2D((3,3))(h)
    h = tf.keras.layers.Conv2D(filters=3,
                               kernel_size=7,
                               strides=1,
                               padding="valid")(h)   # [256, 256, 3]

    outputs3 = tf.nn.tanh(h)


    return tf.keras.Model(inputs=[inputs, inputs2], outputs=[outputs1, outputs2, outputs3])

model = Encoder_Decoder()
model.summary()

def Discriminator(input_shape=(256, 256, 3), weight_decay=0.00002):

    h = inputs = tf.keras.Input(shape=input_shape, batch_size=1)

    # 1

    dim_ = dim = 64
    h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same')(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    for _ in range(3 - 1):
        dim = min(dim * 2, dim_ * 8)
        h = tf.keras.layers.Conv2D(dim, 4, strides=2, padding='same', use_bias=False)(h)
        #h = InstanceNormalization(epsilon=1e-5)(h)
        h = tf.keras.layers.BatchNormalization()(h)
        h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 2
    dim = min(dim * 2, dim_ * 8)
    h = tf.keras.layers.Conv2D(dim, 4, strides=1, padding='same', use_bias=False)(h)
    #h = InstanceNormalization(epsilon=1e-5)(h)
    h = tf.keras.layers.BatchNormalization()(h)
    h = tf.keras.layers.LeakyReLU(alpha=0.2)(h)

    # 3
    h = tf.keras.layers.Conv2D(1, 4, strides=1, padding='same')(h)

    return tf.keras.Model(inputs=inputs, outputs=h)