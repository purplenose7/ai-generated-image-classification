from sklearn.metrics import f1_score,confusion_matrix
from config.settings import CLASS_NAMES, CLASS_TO_IDX
import torch
from collections import Counter
import pandas as pd

def per_class_f1(y_true,y_pred,class_names):
    f1s = f1_score(y_true,y_pred,average=None)
    return zip(class_names,f1s)

def confusion(y_true,y_pred,num_classes):
    return confusion_matrix(y_true,y_pred,labels=list(range(num_classes)))

def ood_report(model, ood_loader, device):
    preds, true = [], []
    model.eval()
    with torch.no_grad():
            for data in loader:
                inputs,labels = data[0].to(self.device), data[1].to(self.device)
                outputs = model(inputs)
                predclas = torch.argmax(outputs, dim=1)
                true.extend(labels.cpu().tolist())
                pred.extend(predclas.cpu().tolist())
    true_pred_df = pd.DataFrame({'true':true,'pred':pred})
    
    diff_idx = CLASS_IDX['diffusion']
    ood_accuracy_on_true_class = (true_pred_df['true']==diff_idx).sum()
    counter = Counter(pred)
    prediction_distribution = {CLASS_NAMES[cls]: count / counter.total() for cls, count in counter.items()}
    fake_reals, total_fakes = 0,0
    for tr,pr in zip(true,pred):
        if tr!=pr:
            total_fakes+=1
            if (tr==diff_idx):
                fake_reals+=1
    fooled_rate = fake_reals/total_fakes
    return {'ood_accuracy_on_true_class':ood_accuracy_on_true_class,
            'prediction_distribution':prediction_distribution, 'fooled_rate':fooled_rate}
