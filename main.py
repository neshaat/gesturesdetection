import numpy as np
import pandas as pd
from keras.models import Sequential
import warnings
warnings.filterwarnings('ignore')
import keras
import h5py
import matplotlib.pyplot as plt # for plotting
import random
import os
import sys
import warnings
import cv2
import skimage.io
import skimage.transform
from skimage.io import imread, imshow, imread_collection, concatenate_images
from skimage.transform import resize
from skimage.morphology import label
from keras.models import Model, load_model
!pip install efficientnet
from efficientnet.tfkeras import EfficientNetB4
from keras.models import load_model
import tensorflow as tf
import time
CATEGORIES=['1','2','3','4','5','a','c','i','o','y']
from PIL import Image
import glob
from datetime import timedelta

class action_detection:
    def video(self, path):
        import os
        folder = '001'
        os.mkdir(folder)
        # use opencv to do the job
        import cv2
        print(cv2.__version__)  # my version is 3.1.0
        vidcap = cv2.VideoCapture(path)
        count = 0
        while True:
            success, image = vidcap.read()
            if not success:
                break
            cv2.imwrite(os.path.join(folder, "frame{:d}.png".format(count)), image)  # save frame as JPEG file
            count += 1
            path = folder
            for file in sorted(glob.glob(path + '/*.png')):
                np_image = cv2.imread(file)
                plt.imshow(np_image)
                plt.show()
                np_image = np.flip(np_image, 1)
                np_image = np.array(np_image).astype('float32') / 255
                np_image = transform.resize(np_image, (224, 224, 3))
                np_image = np.expand_dims(np_image, axis=0)
                pt = model.predict(np_image)
                predic = pt.argmax(axis=1)
                score = float('%0.2f ' % (max(pt[0]) * 100))
                key = predic
                return print('Gesture Class  {},'.format(CATEGORIES[predic.item()]),
                             'score estimated: {}'.format(score), 'Class id is : {}'.format(key))


def loading(path):
    for h5f in glob.iglob(path + '/*.h5'):
        with h5py.File(h5f) as data:
            model = load_model(h5f)
            # model.load_weights(directory)
            return model


if __name__ == "__main__":
    model = loading('/content/drive/MyDrive/git/')
    start_time = time.time()
    print ("Execution took: %s secs " % start_time)
    r = action_detection()
    r.video('/content/neshaat.mp4')