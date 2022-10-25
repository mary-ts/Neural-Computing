import tensorflow as tf
import keras
import pandas as pd
import numpy as np
##import cv2 as cv
import matplotlib.pyplot as plt
import glob as gb
import os
from datasets import Dataset

# random seed
np.random.seed(1671)

# Hyper Parameters
Epochs = 200
Batch_size = 1284
Verbose = 1
NB_classes = 10
N_hidden = 128
Validation_split = 0.2
Reshaped = 784

# ---------------------------------------------------------------------------------------
trainpath = "C:\\Users\\templ\\OneDrive\\college\\CS4287\\Fruit_dataset\\MY_data\\train\\"

#testpath = "C:\\Users\\templ\\OneDrive\\college\\CS4287\\Fruit_dataset\\MY_data\\test\\"

#predpath = "C:\\Users\\templ\\OneDrive\\college\\CS4287\\Fruit_dataset\\MY_data\\predict\\"

len(trainpath)
# ------------------------------------------------------------------------------------------
for folder in os.listdir(trainpath):
    files = gb.glob(pathname=str(trainpath + folder + "/*.jpeg"))
    print(f"for training data , found {len(files)} in folder {folder}")

#for folder in os.listdir(testpath):
 #   files = gb.glob(pathname=str(testpath + folder + "/*.jpeg"))
  #  print(f"for testdata , found {len(files)} in folder {folder}")

#files = gb.glob(pathname=str(predpath + "/*.jpeg"))
#print(len(files))
# ----------------------------------------------------------------------------------------------

# Network
model = tf.keras.models.Sequential()
model.add(keras.layers.Dense(N_hidden,
                             input_shape=(Reshaped,),
                             name="dense_boi",
                             activation='relu'))
model.add(keras.layers.Dense(N_hidden,
                             name="dense_boi_2",
                             activation='relu'))
model.add(keras.layers.Dense(NB_classes,
                             name="dense_boi_3",
                             activation='softmax'))

model.summary()

model.compile(optimizer='SGD',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(files,
          batch_size=Batch_size,
          epochs=Epochs,
          verbose=Verbose,
          validation_split=Validation_split)
