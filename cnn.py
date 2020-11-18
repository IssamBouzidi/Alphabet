# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# %%
import numpy as np
# import matplotlib.pyplot as plt
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
import pickle


# %%
model = Sequential()


# %%
model.add(Conv2D(filters = 32, kernel_size = (3, 3), input_shape=(28,28,1), activation="relu"))
model.add(MaxPooling2D(pool_size = (2, 2)))


# %%
model.add(Conv2D(filters = 32, kernel_size = (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))


# %%
model.add(Flatten())


# %%
model.add(Dense(units=128, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(units=26, activation="softmax")) # valeurs de units represente le nombre de valeurs de sortie, ici, c'est 26 nombre de lettre de l'alphabet


# %%
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])


# %%
train_datagen = ImageDataGenerator(
        rescale=1./255, 
        shear_range=0.2, 
        zoom_range=0.2,
        horizontal_flip=True, 
        vertical_flip=True
    )

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    directory="data/TRAINING", 
    target_size=(28, 28), 
    batch_size=32, 
    class_mode="categorical",
    color_mode="grayscale"
    )

test_generator = test_datagen.flow_from_directory(
    directory="data/TEST", 
    target_size=(28,28), 
    batch_size=32,
    class_mode="categorical",
    color_mode="grayscale")


# %%
entrainement = model.fit_generator(
    train_generator,
    steps_per_epoch=63, # nombre d'image entrainée par lot ==> nombre d'images souhaitées / nombre de lot, ici on suppose on veut entrainé 1000 donc 2000 / 32 (batch_size)
    epochs=25, 
    validation_data=test_generator, 
    validation_steps=16 # nombre d'image tesé par lot ==> nombre d'images souhaitées / nombre de lot, ici on suppose on veut testé 400 donc 600 / 32 (batch_size)
)


# %%
# sauvegarder le model
model.save('alphabet_model.h5')


# %%
# Initialisation du tableau des alphabets
alphabet_array = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z']


# %%



# %%

from keras.preprocessing import image
from keras.models import load_model

choix = r'data/VALIDATION/B-8462.png'
test_model = load_model('alphabet_model.h5')

test_image = image.load_img(choix, target_size = (28, 28))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
train_generator.class_indices

preds = test_model.predict_classes(test_image)
prob = test_model.predict_proba(test_image)

index = preds[0]
print(f'Il s\'agit de la lettre "{alphabet_array[index]}".')


