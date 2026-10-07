from sklearn.metrics import f1_score,confusion_matrix

def per_class_f1(y_true,y_pred,class_names):
    f1s = f1_score(y_true,y_pred,average=None)
    return zip(class_names,f1s)

def confusion(y_true,y_pred,num_classes):
    return confusion_matrix(y_true,y_pred,labels=list(range(num_classes)))

