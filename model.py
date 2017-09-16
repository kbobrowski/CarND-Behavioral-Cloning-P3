import csv, os, cv2, sys
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import matplotlib.pyplot as plt


data_dirs_normal = ['data2_track1/data_{}'.format(i) for i in range(1,16)]
data_dirs_recovery = ['data_track1/data_recovery_{}'.format(i) for i in range(1,7)]
#data_dirs_recovery.append('data_track1/data_additional')
#data_dirs_recovery = []
data_dirs = data_dirs_normal + data_dirs_recovery
lines = []
for data_dir in data_dirs:
    with open(os.path.join(data_dir, 'driving_log.csv')) as csvfile:
        reader = csv.reader(csvfile)
        for line in reader:
            angle = float(line[3])
            if not abs(angle) < 0.05 or np.random.random() < 0.5:
                lines.append(line)
                lines[-1].append(data_dir)

train_samples, validation_samples = train_test_split(lines, test_size=0.2)

#plt.hist([float(line[3]) for line in lines])
#plt.show()


def generator(samples, batch_size=256):
    num_samples = len(samples)
    while True:
        shuffle(samples)

        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset + batch_size]
            images = []
            measurements = []

            for batch_sample in batch_samples:
                correction_factor = 0.3

                for i, correction in enumerate([0, 1, -1]):
                    filename = os.path.basename(batch_sample[i])
                    data_dir = batch_sample[-1]
                    image = cv2.imread(os.path.join(data_dir, 'IMG', filename))
                    angle = float(batch_sample[3]) + correction*correction_factor
                    flipped = np.random.randint(2)
                    if flipped:
                        image = np.fliplr(image)
                        angle = -angle
                    images.append(image)
                    measurements.append(angle)

            X_train = np.array(images)
            y_train = np.array(measurements)

            yield shuffle(X_train, y_train)



train_generator = generator(train_samples)
validation_generator = generator(validation_samples)

model_filename = 'model4_dropout.h5'

if not os.path.exists(model_filename):
    model = Sequential()
    model.add(Lambda(lambda x: x/255. - 0.5, input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((70,24),(0,0))))
    model.add(Convolution2D(24, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(36, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(48, 5, 5, subsample=(2,2), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(128, 3, 3, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.compile(loss='mse', optimizer='adam')
else:
    model = load_model(model_filename)
    model.optimizer.lr.assign(0.0001)

model.fit_generator(train_generator,
                    samples_per_epoch=3*len(train_samples),
                    validation_data=validation_generator,
                    nb_val_samples=3*len(validation_samples),
                    nb_epoch=5)

model.save('model4_dropout2_curves.h5')
