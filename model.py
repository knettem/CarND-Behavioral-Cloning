import os
import csv
import keras
from random import shuffle
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import sklearn

lines = []
with open('./recovery/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)

train_lines, validation_lines = train_test_split(lines, test_size=0.2)

def generator(lines, batch_size=32):
    num_lines = len(lines)
    while 1: 
        shuffle(lines)
        for offset in range(0, num_lines, batch_size):
            batch_lines = lines[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_lines:
                name = './recovery/IMG/'+batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                if batch_sample[3] != 'steering':
               	  center_angle = float(batch_sample[3])
                images.append(center_image)
                angles.append(center_angle)

            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)

train_generator = generator(train_lines, batch_size=32)
validation_generator = generator(validation_lines, batch_size=32)
ch, row, col = 3, 160, 320

#Model Architecture
model = Sequential()

model.add(Lambda(lambda x: x/127.5 - 1, input_shape=(row,col,ch)))
model.add(Cropping2D(cropping=((70,25),(0,0))))
model.add(Conv2D(24,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(36,5,5,subsample=(2,2),activation='relu'))
model.add(Conv2D(48,5,5,subsample=(2,2),activation='relu'))
#model.add(Dropout(.5))
model.add(Conv2D(64,3,3,activation='relu'))
model.add(Conv2D(64,3,3,activation='relu'))
#model.add(Dropout(.2))
model.add(Flatten())
#model.add(Dropout(.3))
model.add(Dense(100))
#model.add(Dropout(.5))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))

#Compile with optimizer = 'Adam'
model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, lines_per_epoch=len(train_lines), validation_data=validation_generator, nb_val_lines=len(validation_lines), nb_epoch=5)
model.save('model.h5')
