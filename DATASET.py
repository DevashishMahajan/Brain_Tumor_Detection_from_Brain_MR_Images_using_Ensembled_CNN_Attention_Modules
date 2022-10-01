
#DATASET

import os
import warnings
warnings.filterwarnings('ignore')

import numpy as np

import tensorflow as tf
import matplotlib.pyplot as plt

def load_dataset(inpDir,subDir,TEST_SIZE,RANDOM_STATE,IMG_HEIGHT,IMG_WIDTH,BATCH_SIZE):
    # Load Data
    # inpDir = 'Dataset_7022' # location where input data is stored
    data_dir = os.path.join(inpDir,subDir)
    print(data_dir)
    
    # load data and split in training and validation from a sub dir
    
    # training dataset
    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=TEST_SIZE,
        subset="training",
        seed=RANDOM_STATE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)
    
    # testing dataset
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=TEST_SIZE,
        subset="validation",
        seed=RANDOM_STATE,
        image_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE)
    
    print(type(train_ds))
    
    print(train_ds.sample_from_datasets)
    
    return train_ds,test_ds


def class_names(train_ds):
    #Is it picking the class names?
    
    class_names = train_ds.class_names
    print('Total: {:3d} Classes; namely : {:s}'.format(len(class_names), str(class_names)))
    
    return class_names


def verify_the_data(train_ds,BATCH_SIZE,class_names):
    #Verify the data
    #To verify that the dataset looks correct, let's plot a batch from the training set and display the class name below each image.
    
    
    plt.figure(figsize=(15,5))
    
    for images, labels in train_ds.take(1): # gets a batch of first BATCH_SIZE images
        
        for i in range(BATCH_SIZE):
            
            plt.subplot(2,int(BATCH_SIZE/2),i+1)
            
            plt.grid(False)
            
            plt.imshow(images[i].numpy().astype("uint8")) # image_size has converted these images to float
            
            plt.title(class_names[labels[i]])
            
            plt.axis("off")
        
        plt.tight_layout()
        
        plt.show()
    
    


def data_preprocessing(train_ds,test_ds):
    #Input shape
    #If we plan to use input layer, we need input shape. Alternatively, we use .build() on the model and let framework capture input shape from the data
    
    #print(images[i].shape)
    
    normalization_layer = tf.keras.layers.Rescaling(1./255.)
    
    img_batch, l_batch = next(iter(train_ds))
    
    img = img_batch[0]
    
    print ("np.max(img) before nomalization ",np.max(img),"np.min(img) before nomalization ", np.min(img))
    
    train_ds = train_ds.map(lambda X, y: (normalization_layer(X), y) )
    test_ds = test_ds.map(lambda X, y: (normalization_layer(X), y) )
    
    img_batch, l_batch = next(iter(train_ds))
    
    img = img_batch[0]
    print ("np.max(img) before nomalization ",np.max(img),"np.min(img) before nomalization ", np.min(img))
    
    
    
    ## Optimize for performance
    #loading data into cache for faster execution
    AUTOTUNE = tf.data.AUTOTUNE
    
    train_ds = train_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    test_ds = test_ds.cache().prefetch(buffer_size=tf.data.AUTOTUNE)
    
    return train_ds,test_ds
    

