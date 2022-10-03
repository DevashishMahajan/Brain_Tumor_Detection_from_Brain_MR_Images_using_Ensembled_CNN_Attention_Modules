
#plot_confusion_matrix

import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import confusion_matrix

from sklearn.metrics import  ConfusionMatrixDisplay

import matplotlib.pyplot as plt


def plot_confusion_matrix(y_test, y_pred, class_names):


    #Confusion matrix
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                           display_labels=class_names)
    
    fig, ax = plt.subplots(figsize = (9,9))
    disp.plot(ax = ax, cmap=plt.cm.Blues);
    ax.set_xticklabels(class_names,rotation=45, ha='right');



