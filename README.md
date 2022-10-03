## Brain_Tumor_Detection_from_Brain_MR_Images_using_Ensembled_CNN_Attention_Modules

### Proposed a new model for brain MR image classification called DeSOMNet


### Classification of Brain MR Images into Meningioma, Glioma, No tumor and Pituitary

We propose a new convolutional neural network model named DeSOMNet. The architecture is built on attention modules and hypercolumn technique with a residual network. 

Firstly, an image is pre-processed in DeSOMNet and is transferred to attention modules using image augmentation techniques. 

Attention modules select important areas of the image and the image is then transferred to convolutional layers. One of the most important techniques that the DeSOMNet model uses in the convolutional layers is hypercolumn. 

With the help of hypercolumns, the features extracted from each layer of the DeSOMNet model are retained by the array structure in the last layer. 

Our proposed dataset consists of a collection of 7023 Brain MR images divided into 4 classes from the widely accepted dataset of figshare, SARTAJ dataset and Br35H.

### Dataset:

7023 images from the figshare, SARTAJ dataset and Br35H,

Training images : 4570, testing images : 1311  for and remaining 1142 images are for validation.

### Results

The experimental results of the proposed methodology in terms of highest accuracy, recall, precision, and error rate are 96.41%, 99%, 98%, 3.59% respectively.

