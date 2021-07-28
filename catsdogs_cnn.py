# Download do DB cat_dogs link: https://1drv.ms/u/s!AhRegqXKWQCJhLwUFiDT-r1DrsxlQw?e=c5fSLR

from tensorflow.python.client import device_lib

def get_available_devices():
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU' or x.device_type == 'CPU']

import os
import matplotlib.pyplot as plt
from keras import models, layers
from keras.preprocessing.image import ImageDataGenerator

# Preparação dos dados

base_dir = r"CatsDogs/"

train_datagen = ImageDataGenerator(rescale=1./255)
validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        os.path.join(base_dir,"catdog_train"),
        target_size=(150,150),
        batch_size=20
        #, class_mode="binary"
        )

validation_generator = validation_datagen.flow_from_directory(
        os.path.join(base_dir,"catdog_validation"),
        target_size=(150,150),
        batch_size=20
        #, class_mode="binary"
        )

# criar o modelo
model = models.Sequential()
model.add(layers.Conv2D(256, (3,3),activation='relu', input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64, (3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dense(256, activation = 'relu'))
model.add(layers.Dense(2, activation = 'softmax'))
#model.add(layers.Dense(1, activation = 'sigmoid'))
model.compile(optimizer='adam',loss='binary_crossentropy', metrics=['accuracy'])

model.summary()

# Por que não posso aumentar o numero de epocas e etapas por epicas?
history = model.fit_generator(
        train_generator, 
        steps_per_epoch = 100,
        epochs = 50,
        validation_data = validation_generator,
        validation_steps = 50
        )

# avaliar resultado do treinamento
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(acc)+1)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
