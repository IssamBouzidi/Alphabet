#coding-utf8

import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation
import os
import pickle


train_datagen = ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True, vertical_flip=True)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(directory="data/Training", target_size=(28, 28), batch_size=28, class_mode="categorical")

test_generator = test_datagen.flow_from_directory(directory="data/Testing", target_size=(28,28), batch_size=28, class_mode="categorical")


model = Sequential()

model.add(Conv2D(28, (3, 3), input_shape=(28,28,3), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))

model.add(Conv2D(28, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

# model.add(Conv2D(28, (3, 3), activation="relu"))
# model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(units=128, activation="relu"))
model.add(Dense(units=26, activation="softmax"))

model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()

entrainement = model.fit_generator(train_generator, steps_per_epoch=32, epochs=20, validation_data=test_generator, validation_steps=32)

# pickle.dump(model, open("CNN_model.sav", "wb"))
# model = pickle.load(open("CNN_model.sav","rb"))

def get_result(result):
    if result[0][0] == 1:
        return("A")
    elif result[0][1] == 1:
        return ("B")
    elif result[0][2] == 1:
        return ("C")
    elif result[0][3] == 1:
        return ("D")
    elif result[0][4] == 1:
        return ("E")
    elif result[0][5] == 1:
        return ("F")
    elif result[0][6] == 1:
        return ("G")
    elif result[0][7] == 1:
        return ("H")
    elif result[0][8] == 1:
        return ("I")
    elif result[0][9] == 1:
        return ("J")
    elif result[0][10] == 1:
        return ("K")
    elif result[0][11] == 1:
        return ("L")
    elif result[0][12] == 1:
        return ("M")
    elif result[0][13] == 1:
        return ("N")
    elif result[0][14] == 1:
        return ("O")
    elif result[0][15] == 1:
        return ("P")
    elif result[0][16] == 1:
        return ("Q")
    elif result[0][17] == 1:
        return ("R")
    elif result[0][18] == 1:
        return ("S")
    elif result[0][19] == 1:
        return ("T")
    elif result[0][20] == 1:
        return ("U")
    elif result[0][21] == 1:
        return ("V")
    elif result[0][22] == 1:
        return ("W")
    elif result[0][23] == 1:
        return ("X")
    elif result[0][24] == 1:
        return ("Y")
    elif result[0][25] == 1:
        return ("Z")


choix = r'data/Testing/a/A-200.png'
image_test = image.load_img(choix, target_size = (28,28))
plt.imshow(image_test)
image_test = image.img_to_array(image_test)
image_test = np.expand_dims(test_image, axis = 0)


result = model.predict(image_test)
result = get_result(resultat)
print(f"Il s'agit de la lettre {result}.")