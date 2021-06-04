# -*- coding: utf-8 -*-
from random import random, shuffle
from F2M_model import *

import matplotlib.pyplot as plt
import easydict
import numpy as np
import os

FLAGS = easydict.EasyDict({"img_size": 256,
                           
                           "batch_size": 1,
                           
                           "input_path": "D:/[1]DB/[1]second_paper_DB/AFAD_16_69_DB/backup/fix_AFAD/",

                           "input_txt": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/female_16_39_train.txt",
                           
                           "ref_path": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/Morph/All/male_40_63/",
                           
                           "ref_txt": "D:/[1]DB/[2]third_paper_DB/[4]Age_and_gender/race_age_gender_generation/Morph_AFAD_16_63/first_fold/AFAD-F_Morph-M_16_39_40_63/train/male_40_63_train.txt",
                           
                           "epochs": 200,

                           "lr": 4e-4,
                           
                           "sample_images": "C:/Users/Yuhwan/Downloads/img"})

g_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)
d_optim = tf.keras.optimizers.Adam(FLAGS.lr, beta_1=0.5)

def input_func(A_path, B_path):

    A_img = tf.io.read_file(A_path)
    B_img = tf.io.read_file(B_path)

    A_img = tf.image.decode_jpeg(A_img, 3)
    B_img = tf.image.decode_jpeg(B_img, 3)

    A_img = tf.image.resize(A_img, [FLAGS.img_size + 15, FLAGS.img_size + 15])
    B_img = tf.image.resize(B_img, [FLAGS.img_size + 15, FLAGS.img_size + 15])
    A_img = tf.image.random_crop(A_img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.
    B_img = tf.image.random_crop(B_img, [FLAGS.img_size, FLAGS.img_size, 3]) / 127.5 - 1.

    if random() > 0.5:
        A_img = tf.image.flip_left_right(A_img)
        B_img = tf.image.flip_left_right(B_img)

    return A_img, B_img

#@tf.function
def model_func(model, images, training=True):
    return model(images, training)

#@tf.function
def cal_loss(A_2_B_model, B_2_A_model, A_discrim, B_discrim, A_images, B_images):

    with tf.GradientTape() as g_tape, tf.GradientTape() as d_tape:
        bridge1_fake_B, bridge2_fake_B, final_fake_B = model_func(A_2_B_model, [A_images, B_images], True)
        bridge1_rec_A, bridge2_rec_A, final_rec_A = model_func(B_2_A_model, [final_fake_B, A_images], True)

        bridge1_fake_A, bridge2_fake_A, final_fake_A = model_func(B_2_A_model, [B_images, A_images], True)
        bridge1_rec_B, bridge2_rec_B, final_rec_B = model_func(A_2_B_model, [final_fake_A, B_images], True)

        bridge1_id_A, bridge2_id_A, final_id_A = model_func(B_2_A_model, [A_images, B_images], True)
        bridge1_id_B, bridge2_id_B, final_id_B = model_func(A_2_B_model, [B_images, A_images], True)

        realA_logits = model_func(A_discrim, A_images, True)
        fakeA_logits = model_func(A_discrim, final_fake_A, True)

        realB_logits = model_func(B_discrim, B_images, True)
        fakeB_logits = model_func(B_discrim, final_fake_B, True)

        sum_rec_A = (bridge1_rec_A + bridge2_rec_A) / 2.0
        sum_rec_B = (bridge1_rec_B + bridge2_rec_B) / 2.0

        sum_id_A = (bridge1_id_A + bridge2_id_A) / 2.0
        sum_id_B = (bridge1_id_B + bridge2_id_B) / 2.0

        cycle_loss = (tf.reduce_mean(tf.abs(final_rec_A - A_images)) * 10.0 \
            + tf.reduce_mean(tf.abs(sum_rec_A - A_images)) * 5.0 \
            + tf.reduce_mean(tf.abs(final_rec_B - B_images)) * 10.0 \
            + tf.reduce_mean(tf.abs(sum_rec_B - B_images)) * 5.0) / 4.0

        id_loss = (tf.reduce_mean(tf.abs(sum_id_A - A_images)) * 5.0 \
            + tf.reduce_mean(tf.abs(sum_id_B - B_images)) * 5.0 \
            + tf.reduce_mean(tf.abs(final_id_A - A_images)) * 10.0 \
            + tf.reduce_mean(tf.abs(final_id_B - B_images)) * 10.0) / 4.0

        gen_loss = tf.reduce_mean((fakeA_logits - tf.ones_like(fakeA_logits))**2) + tf.reduce_mean((fakeB_logits - tf.ones_like(fakeB_logits))**2)

        G_loss = cycle_loss + id_loss + gen_loss

        D_loss = (tf.reduce_mean((realA_logits - tf.ones_like(realA_logits))**2) + tf.reduce_mean((fakeA_logits - tf.zeros_like(fakeA_logits))**2) \
            + tf.reduce_mean((realB_logits - tf.ones_like(realB_logits))**2) + tf.reduce_mean((fakeB_logits - tf.zeros_like(fakeB_logits))**2)) / 2

    d_grads = d_tape.gradient(D_loss, A_discrim.trainable_variables + B_discrim.trainable_variables)
    g_grads = g_tape.gradient(G_loss, A_2_B_model.trainable_variables + B_2_A_model.trainable_variables)

    d_optim.apply_gradients(zip(d_grads, A_discrim.trainable_variables + B_discrim.trainable_variables))
    g_optim.apply_gradients(zip(g_grads, A_2_B_model.trainable_variables + B_2_A_model.trainable_variables))

    return G_loss, D_loss

def main():

    A_2_B_model = Encoder_Decoder(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), batch_size=FLAGS.batch_size)
    B_2_A_model = Encoder_Decoder(input_shape=(FLAGS.img_size, FLAGS.img_size, 3), batch_size=FLAGS.batch_size)

    A_discrim = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))
    B_discrim = Discriminator(input_shape=(FLAGS.img_size, FLAGS.img_size, 3))

    input_data = np.loadtxt(FLAGS.input_txt, dtype="<U100", usecols=0, skiprows=0)
    input_data = [FLAGS.input_path + img for img in input_data]

    ref_data = np.loadtxt(FLAGS.ref_txt, dtype="<U100", usecols=0, skiprows=0)
    ref_data = [FLAGS.ref_path + img for img in ref_data]

    count = 0
    for epoch in range(FLAGS.epochs):
        A = list(zip(input_data, ref_data))
        shuffle(A)
        input_data, ref_data = zip(*A)
    
        input_data, ref_data = np.array(input_data), np.array(ref_data)

        TR_gener = tf.data.Dataset.from_tensor_slices((input_data, ref_data))
        TR_gener = TR_gener.shuffle(len(input_data))
        TR_gener = TR_gener.map(input_func)
        TR_gener = TR_gener.batch(FLAGS.batch_size)
        TR_gener = TR_gener.prefetch(tf.data.experimental.AUTOTUNE)

        tr_iter = iter(TR_gener)
        tr_idx = len(input_data) // FLAGS.batch_size
        for step in range(tr_idx):

            A_images, B_images = next(tr_iter)

            g_loss, d_loss = cal_loss(A_2_B_model, B_2_A_model, A_discrim, B_discrim, A_images, B_images)

            print(g_loss, d_loss, count)

            if count % 100 == 0:
                bridge1_fake_B, bridge2_fake_B, final_fake_B = A_2_B_model([A_images, B_images], False)

                sum_fake_B = (bridge1_fake_B + bridge2_fake_B + final_fake_B) / 3.

                plt.imsave("C:/Users/Yuhwan/Pictures/img/{}_fakeB1.jpg".format(count), bride_fake_B[0] * 0.5 + 0.5)
                plt.imsave("C:/Users/Yuhwan/Pictures/img/{}_fakeB2.jpg".format(count), bride2_fake_B[0] * 0.5 + 0.5)
                plt.imsave("C:/Users/Yuhwan/Pictures/img/{}_fake_finaBl.jpg".format(count), final_fake_B[0] * 0.5 + 0.5)
                plt.imsave("C:/Users/Yuhwan/Pictures/img/{}_merge_fakeB.jpg".format(count), sum_fake_B[0] * 0.5 + 0.5)

                plt.imsave("C:/Users/Yuhwan/Pictures/img/{}_realA.jpg".format(count), A_images[0] * 0.5 + 0.5)
                plt.imsave("C:/Users/Yuhwan/Pictures/img/{}_targetB.jpg".format(count), B_images[0] * 0.5 + 0.5)

            count += 1

if __name__ == "__main__":
    main()