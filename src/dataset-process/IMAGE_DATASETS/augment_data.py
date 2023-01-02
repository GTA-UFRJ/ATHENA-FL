# Author: Lucas Airam Castro de Souza
# Laboratory: Grupo de Teleinformatica e Automacao (GTA)
# University: Universidade Federal do Rio de Janeiro (UFRJ)
#
#
#
#
#
#
#

import os

import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

from numpy import expand_dims
from keras.preprocessing.image import ImageDataGenerator





def augment_dataset(dataset,label):

    if (not dataset) or (not label):
        print("The dataset is empty")

    for index, img in enumerate(dataset):

        # flip the image        
        newImage = tf.image.flip_up_down(img)
        
        if not index:
            augmented_dataset = np.array([newImage])
            augmented_label = np.array([label[index]])


        else:    
            augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
            augmented_label = np.concatenate((augmented_label,[label[index]]))

       # newImage = tf.image.flip_left_right(img)
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))

        # rotate the image 
       # newImage = tfa.image.rotate(img, tf.constant(np.pi/8))
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))
       # 
       # newImage = tfa.image.rotate(img, tf.constant(np.pi))
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))
       # 
       # newImage = tfa.image.rotate(img, tf.constant(2*np.pi/3))
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))
       # 
       # newImage = tfa.image.rotate(img, tf.constant(4*np.pi/3))
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))


       # newImage = tfa.image.rotate(img, tf.constant(7*np.pi/4))
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))

       # newImage = tfa.image.rotate(img, tf.constant(np.pi/2))
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))
       # 
       # newImage = tfa.image.rotate(img, tf.constant(np.pi/4))
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))
        

        newImage = np.asarray(tfa.image.rotate(img, tf.constant(5*np.pi/3)),dtype=np.uint8)
        augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
        augmented_label = np.concatenate((augmented_label,[label[index]]))

        ## translate the image
       # newImage = tfa.image.translate(img,[10,0])
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))

       # newImage = tfa.image.translate(img,[0,10])
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))

       # newImage = tfa.image.translate(img,[5,5])
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))
        
       # newImage = tfa.image.translate(img,[-10,5])
       # augmented_dataset = np.concatenate((augmented_dataset,[newImage]))
       # augmented_label = np.concatenate((augmented_label,[label[index]]))

        ## add noise to the image
        noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=1.0,dtype=tf.float32)
        augmented_dataset = np.concatenate((augmented_dataset,[np.asarray(img+noise,dtype=np.uint8)]))
        augmented_label = np.concatenate((augmented_label,[label[index]]))
    
    augmented_dataset = np.concatenate((augmented_dataset,dataset))
    augmented_label = np.concatenate((augmented_label,label))

    return augmented_dataset, augmented_label


