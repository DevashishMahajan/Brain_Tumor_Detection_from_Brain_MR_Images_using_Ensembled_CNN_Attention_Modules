
#recall, precision, f1


import numpy as np
from sklearn.metrics import  classification_report, accuracy_score



def recall_precision_f1(model,test_ds):

    #Evaluate the model
    
    test_loss, test_acc = model.evaluate(test_ds, verbose=1)
    
    yhat = model.predict(test_ds)
    
    y_test = np.concatenate([y for X, y in test_ds], axis=0).squeeze()
    
    y_pred = yhat.argmax(axis = 1)
    
    print('Accuracy score on Test Data : {:.4f}'.format(accuracy_score(y_test, y_pred)))
    
    
    #Classification_report
    
    print(classification_report(y_test, y_pred))
    
    return y_pred, y_test




