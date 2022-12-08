from enum import IntEnum
from PIL import Image
import io
import numpy as np
import cv2
import tensorflow as tf

from dcgazeflow_demo.demo_utils import *
from dcgazeflow_demo.models.model import GazeFlow
from kasane.fshd.FSHDIMG import FSHDJPG

import ipywidgets as widgets
from ipywidgets import interact

INPUT_SIZE = (64, 32)

class Hparams:
    images_width = INPUT_SIZE[1]
    images_height = INPUT_SIZE[0]
    images_channel = 3
    K = 18
    L = 3
    conditional = True
    width = 512
    checkpoint_path = './'
    condition_shape=(5,)
    skip_type = "whole"
    checkpoint_path_specific = "ckpt"
    
class Temp:
    def __init__(self, cond, encoded, zaux):
        self.cond = cond
        self.encoded = encoded
        self.zaux = zaux

    def set_encoded(self, new):
        self.encoded = new

    def set_zaux(self, new):
        self.zaux = new

    def set_cond(self, new):
        self.cond = new
        
def img_to_bytes(x_sample):
    """ tool funcion to code image for using ipywidgets.widgets.Image plotting function """
    imgObj = Image.fromarray((x_sample * 255).astype(np.uint8))#.convert('RGB')
    imgByteArr = io.BytesIO()
    imgObj.save(imgByteArr, format='PNG')
    imgBytes = imgByteArr.getvalue()
    return imgBytes
    

class EyeType(IntEnum):
    left = 0
    right = 1

    
def crop_eye(image, landmarks, eye_type: int, border=0.2):
    if eye_type == EyeType.left:
        eye_landmarks = list(range(42, 48))
    elif eye_type == EyeType.right:
        eye_landmarks = list(range(36, 42))
    else:
        raise ValueError('eye_type not in [0, 1]')

    x1 = int(landmarks[eye_landmarks, 0].min())
    y1 = int(landmarks[eye_landmarks, 1].min())
    x2 = int(landmarks[eye_landmarks, 0].max())
    y2 = int(landmarks[eye_landmarks, 1].max())

    # Fixme:
    eye_landmarks = list(range(36, 48))
    y1 = int(landmarks[eye_landmarks, 1].min())
    y2 = int(landmarks[eye_landmarks, 1].max())
        
    
    x_pad = int((x2 - x1) * border)
    y_pad = int((y2 - y1) * border)
    x1, y1, x2, y2 = x1 - x_pad, y1 - y_pad, x2 + x_pad, y2 + y_pad * 3
    eye = image[y1: y2, x1: x2]
    return eye.copy(), (x1, y1, x2, y2)


def get_eyes(image, landmarks, border=0.4):
    return crop_eye(image, landmarks, EyeType.left, border), crop_eye(image, landmarks, EyeType.right, border)


def get_eye_resized(image, landmarks, border=0.4, input_size=(64, 32)):
    eyes = get_eyes(image, landmarks, border)
    return (cv2.resize(eyes[0][0], input_size), cv2.resize(eyes[1][0], input_size)), (eyes[0][1], eyes[1][1])


def new_eyes(model, image, landmarks, gaze_pitch, gaze_yaw, pose_pitch, pose_yaw, delta_left, delta_right, border=0.4):
    
    # передаю подсчитанные conditions
    init_cond_lelt = np.asarray([[gaze_pitch, gaze_yaw, pose_pitch, pose_yaw, 0.]])
    init_cond_right = np.asarray([[gaze_pitch, gaze_yaw, pose_pitch, pose_yaw, 1.]])
    
    eyes = get_eye_resized(image, landmarks, border)
    
    eye_left = cv2.cvtColor(eyes[0][0], cv2.COLOR_BGR2RGB)
    eye_right = cv2.cvtColor(eyes[0][1], cv2.COLOR_BGR2RGB)
    
    encoded_left_eye, zaux_left = encode(model, init_cond_lelt, (eye_left[None] / 255.).astype(np.float32))
    encoded_right_eye, zaux_right = encode(model, init_cond_right, (eye_right[None] / 255.).astype(np.float32))
    
    delta_left = tf.convert_to_tensor(np.array([delta_left]).astype(np.float32))
    delta_right = tf.convert_to_tensor(np.array([delta_right]).astype(np.float32))

    new_eye_left = np.array(decode(model, init_cond_lelt + delta_left, encoded_left_eye, zaux_left))
    new_eye_right = np.array(decode(model, init_cond_right + delta_right, encoded_right_eye, zaux_right))
    
    eye_image = np.zeros_like(image).astype(np.float32)
    eye_mask = np.zeros_like(image)

    x1, y1, x2, y2 = eyes[1][0]
    eye_image[y1:y2, x1:x2] = cv2.resize(cv2.cvtColor(new_eye_left[0], cv2.COLOR_RGB2BGR), (x2 - x1, y2 - y1))
    eye_mask[y1:y2, x1:x2] = 1

    x1, y1, x2, y2 = eyes[1][1]
    eye_image[y1:y2, x1:x2] = cv2.resize(cv2.cvtColor(new_eye_right[0], cv2.COLOR_RGB2BGR), (x2 - x1, y2 - y1))
    eye_mask[y1:y2, x1:x2] = 1

    eye_mask = np.clip(cv2.GaussianBlur(eye_mask, (5, 5), 0), 0, 1)
    image = eye_mask * eye_image + (image / 255.) * (1 - eye_mask)
    
    return np.clip(image * 255, 0, 255).astype(np.uint8), eye_left, eye_right