import numpy as np

import tensorflow as tf


def decode(model, condition, z=None, zaux=None):
    if len(condition.shape) == 1:
        condition = tf.expand_dims(condition, axis=0)
    batchsize = condition.shape[0]
    if z is not None:
        return model.glow.inverse(z, condition, zaux=zaux, training=False)[0]
    else:
        return model.glow.inverse(tf.random.normal((batchsize, 4, 4, 96)) * 0.75, condition, zaux=zaux, training=False)[0]


def get_img_with_noise(conditions):
    random_conditions = conditions + \
        tf.concat([tf.random.normal(conditions[:, :-1].shape)
                   * .01, conditions[:, -1:]], axis=-1)
    return decode(random_conditions)[0], random_conditions


def encode(model, condition, x):
    encoded = model.glow.forward(x, condition, training=False)
    return encoded[0], encoded[2]


def tensor_to_int8(tensor):
    return np.clip(tensor.numpy()*255, 0, 255).astype(np.uint8)


def flip_it_back(img):
    return img[:, ::-1, :]
