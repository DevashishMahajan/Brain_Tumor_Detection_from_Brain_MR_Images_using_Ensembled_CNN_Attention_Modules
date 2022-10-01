
"""

Group No: 09
Group Members:
1) Devashish Mahajan (007)
2) Imroze Khan (029)
3) Manvi Poddar (024)
4) Soham Rangdal (044)
    
    
Topic: Brain tumor classification from magnetic resonace images with ensemble convolution neural network model and feature selection architecture




Versions of libraries in training enviroment

python = 3.10.4

Tensorflow = 2.9.1

numpy = 1.23.0

pandas = 1.4.3

matplotlib = 3.5.2

matplotlib-inline = 0.1.2

keras = 2.9.0

keras-preprocessing = 1.1.2

sklearnn = 1.1.1

"""


### Import Libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf


#Check for GPU available
physical_devices = tf.config.list_physical_devices('GPU') 
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))
if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
#%matplotlib inline



# Some basic parameters
global inpDir
inpDir = 'Dataset_7022' # location where input data is stored

global outDir
outDir = 'Dataset_7022' # location to store outputs

global subDir
subDir = 'Training'

global modelDir
modelDir = 'Models'

global valDir
valDir = 'Testing'

global altName 
altName = 'Brain_tumor_classification_Model_Checkpoints'


RANDOM_STATE = 24 # for initialization ----- REMEMBER: to remove at the time of promotion to production
tf.random.set_seed(RANDOM_STATE)

EPOCHS = 60 # number of cycles to run

global BATCH_SIZE
#BATCH_SIZE = 4
BATCH_SIZE = 8 # increse in batch size cause resource exhausted    MAKE BATCH_SIZE = 4
#BATCH_SIZE = 16
#BATCH_SIZE = 64
#BATCH_SIZE = 32


ALPHA = 0.001
#ALPHA = 0.01

TEST_SIZE = 0.2

IMG_HEIGHT = 224

IMG_WIDTH = 224

# MRI image shape
SHAPE = (IMG_HEIGHT,IMG_WIDTH, 3) 


# SGDRScheduler parameters
ES_PATIENCE = 50 # if performance does not improve stop

LR_PATIENCE = 5  # if performace is not improving reduce alpha

LR_FACTOR = 0.9  # rate of reduction of alpha


# Set parameters for decoration of plots
params = {'legend.fontsize' : 'large',
          'figure.figsize'  : (15,10),
          'axes.labelsize'  : 'x-large',
          'axes.titlesize'  :'x-large',
          'xtick.labelsize' :'large',
          'ytick.labelsize' :'large',
         }

CMAP = plt.cm.brg

plt.rcParams.update(params) # update rcParams


#############  Helper_Function.py #############
from Helper_Function import fn_plot_hist, fn_plot_label



#############  DATASET.py  #############

from DATASET import load_dataset,class_names, verify_the_data,data_preprocessing


train_ds,test_ds=load_dataset(inpDir,subDir,TEST_SIZE,RANDOM_STATE,IMG_HEIGHT,IMG_WIDTH,BATCH_SIZE)

class_names = class_names(train_ds)

verify_the_data(train_ds,BATCH_SIZE,class_names)


#Helper Function
fn_plot_label(train_ds, test_ds,class_names)



# data preprocessing (normalization_layer)
data_preprocessing(train_ds,test_ds)




#############  Model.py  #############

from Model import create_model
model = create_model(SHAPE)
model.summary()

from tensorflow import keras
model_plot = keras.utils.plot_model(model, show_shapes=True)
print(model_plot)


#############  SGDRScheduler  #############


# Directory where the checkpoints will be saved
checkpoint_dir = os.path.join(modelDir, subDir)
print("checkpoint_dir ", checkpoint_dir)

# Early Stopping
early_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',
                                                  restore_best_weights=True,
                                                  patience=ES_PATIENCE,
                                                  verbose=1)


# Reduction scheduler for alpha
lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', 
                                                 factor=LR_FACTOR,
                                                 patience=LR_PATIENCE)


checkpoint_prefix = os.path.join(checkpoint_dir, altName)
print("checkpoint_prefix ",checkpoint_prefix)

# Save weights
model_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_prefix,
                                                    monitor='val_loss',
                                                    mode='auto',
                                                    save_weights_only=True,
                                                    save_best_only=True,
                                                    verbose=1)


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=ALPHA),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['accuracy'])



#############  Train Model  #############

history = model.fit(train_ds,
                    validation_data=test_ds,
                    callbacks=[ early_callback, model_callback, lr_reduce],
                    epochs=EPOCHS)




# Convert history to pandas dataframe
res_df = pd.DataFrame(history.history)



# Helper Function
# Plot loss and accuracy curves for both testing and training
fn_plot_hist(res_df,CMAP)


############# Save model  #############

model.load_weights(checkpoint_dir) # checkpoint_prefix
model.save('CDAC_Project_Brain_tumor_classification_Dataset7022.h5')
model = tf.keras.models.load_model('CDAC_Project_Brain_tumor_classification_Dataset7022.h5')



#############  recall_precision_f1.py  #############
from classification_report import recall_precision_f1

y_pred, y_test = recall_precision_f1(model,test_ds)


#############  plot_confusion_matrix.py  #############

from plot_confusion_matrix import plot_confusion_matrix

plot_confusion_matrix(y_test, y_pred, class_names)






























    
