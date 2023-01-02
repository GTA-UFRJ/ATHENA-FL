#
# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#

import tensorflow as tf


def build_model(basicNN, dataset_name):
    if dataset_name == "CIFAR-10":
        # Verify if we are training a robust model or OvA models
        if not basicNN:
#            model = tf.keras.applications.MobileNetV2((32, 32, 3), classes=10, weights=None)
#            model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
            # change the hyperparameter to test
            model = tf.keras.applications.MobileNet((32, 32, 3), classes=10, weights=None,dropout=0.3)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                   loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        
        # If it is a basic NN we train a One-versus-All models
        else:
            model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(32, 32, 3),padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
            
    elif dataset_name == "MNIST":
        # Verify if we are training a robust model or OvA models
        if not basicNN:
            model = tf.keras.applications.MobileNet((28, 28, 1), classes=10, weights=None,dropout=0.1)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        
        # If it is a basic NN we train a One-versus-All models
        else:
            model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28,1),padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 

    elif dataset_name == "FMNIST":
        # Verify if we are training a robust model or OvA models
        if not basicNN:
            model = tf.keras.applications.MobileNet((28, 28, 1), classes=10, weights=None,dropout=0.1)
            model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                    loss='sparse_categorical_crossentropy',metrics=['accuracy'])
        
        
        # If it is a basic NN we train a One-versus-All models
        else:
            model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(28, 28, 1),padding='same'),
            tf.keras.layers.MaxPooling2D(2, 2),
            tf.keras.layers.Conv2D(32, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Conv2D(64, (3,3), activation='relu',padding='same'),
            tf.keras.layers.MaxPooling2D(2,2),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')])
            model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy']) 
    
    return model
