#Importing necessary libraries
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from tkinter.filedialog import askopenfile
from PIL import Image, ImageTk
import os
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K

window = tk.Tk()
window.title("Tumour Classification")

window.geometry("%dx%d" % (500, 600))

window.grid_rowconfigure(0, weight=1)
window.grid_columnconfigure(0, weight=1)
window.configure(background='#2b2b2b')

logo_img = Image.open("cdac_logo.png")    # logo image
logo_img = logo_img.resize((200,200))     
logo = ImageTk.PhotoImage(logo_img)
label = tk.Label(window, image = logo)
label.place(relx=0.5, rely=0.5, anchor=CENTER)

def classification():
    print("image in detection")
    IMG_HEIGHT = 224
    IMG_WIDTH = 224
    normalization_layer = tf.keras.layers.Rescaling(1./255.)

    # Classes
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']

    #BrainMRNet_Model_Checkpoints.h5 
    new_model = tf.keras.models.load_model('Brain_Tumor_Classification_Dataset7022_Train99.72_Test94.66.h5')

    Test_Image_dir = 'Test_Image'

    Test_Image_dir = os.path.join(Test_Image_dir)
    test_image_ds = tf.keras.utils.image_dataset_from_directory(Test_Image_dir,
                                                                image_size=(IMG_HEIGHT,
                                                                            IMG_WIDTH))
    # Normalize image
    test_image_ds = test_image_ds.map(lambda X, y: (normalization_layer(X), y))

    #Predict using model
    yhat = new_model.predict(test_image_ds)

    #argmax to find class with maximum probability
    y_pred = yhat.argmax(axis = 1)

    #Prediction label
    return(class_names[y_pred[0]])

def upload_file():
    f_types = [('Jpg Files', '*.jpg'), ('PNG Files','*.png'), ('Jpeg Files', '*.jpeg')]   # type of files to select
    filename = tk.filedialog.askopenfilename(multiple=True,filetypes=f_types)

    
    for f in filename:

        # read the image file
        img = Image.open(f)   

        # resizing for the display purpose          
        dis_img = img.resize((224,224))  

        # creating a directory named "Test_Image" where we can store the uploaded image only 
        Test_Image_dir = 'Test_Image'   # 
        if not os.path.isdir(Test_Image_dir):
            os.mkdir(Test_Image_dir)
            os.mkdir(f"{Test_Image_dir}/test")
            
        # saving uploaded image in the new directory
        img.save(f"Test_Image/test/image.jpg")  

        # function call to "classification()" for the classification of the tumour
        tum_class = classification()      

        
        # display image on screen
        img = ImageTk.PhotoImage(dis_img)

        # displaying the image on the tkinter window
        e1 =tk.Label(window)
        e1.place(relx=0.5, rely=0.5, anchor=CENTER)
        e1.image = img
        e1['image']=img # garbage collection
        lbl2 = tk.Label(window, text=tum_class.upper(), width=400, fg="white", bg="black", height=2, font=('arial', 20, 'bold'))
        lbl2.place(relx=0.5, rely=0.74, anchor=CENTER)
        

head = tk.Label(window, text='Brain Tumour Classification', width=500, fg="white", bg="black", height=5, font=('arial', 30, 'bold'))
head.place(relx=0.5, rely=0.1, anchor=CENTER)     # setting coordinates of the heading label
b1 = tk.Button(window, text='Upload File', width=500, height=4, command = lambda:upload_file(), font=('arial', 30, 'bold'))
b1.place(relx=0.5, rely=0.9, anchor=CENTER)       # setting coordinates of the upload button

window.mainloop()  # Keep the window open
