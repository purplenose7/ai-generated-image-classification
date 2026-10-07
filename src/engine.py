import torch
import torch.nn as nn

class ForensicsTrainer:
    def __init__(self,model,optimizer,scheduler,loss_fn, device, ckpt_path, wandb_run):
        self.model = model
        self.optimizer=optimizer
        self.scheduler = scheduler
        self.loss_fn = loss_fn
        self.device = device
        self.ckpt_path = ckpt_path
        self.wandb_run = wandb_run
    
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0
        # device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
        self.model.to(self.device)

    def train_one_epoch(self,loader):
        self.model.train()
        batch,running_loss = 0, 0.0
        for i, data in enumerate(loader,0):
            inputs, labels = data[0].to(self.device), data[1].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            
            loss = self.loss_fn(outputs,labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
            batch+=1
            
        return running_loss/batch 


    def evaluate(self,loader):
        self.model.eval()
        running_loss, true, pred = 0.0, [], []
        with torch.no_grad():
            for data in loader:
                inputs,labels = data[0].to(self.device), data[1].to(self.device)
                outputs = self.model(inputs)
                predclas = torch.argmax(outputs,dim=1)
                true.extend(labels.cpu().tolist())
                pred.extend(predclas.cpu().tolist())
                running_loss+=self.loss_fn(outputs,labels).item()
            mean_loss = running_loss/len(loader)
            accuracy = sum(t == p for t, p in zip(true, pred))/len(true)
            return {'loss':mean_loss,
                    'accuracy':accuracy,
                    'per class f1':per_class_f1(true,pred,CLASS_NAMES),
                    'confusion':confusion(true,pred,len(CLASS_NAMES)),
                    'y_true':true,
                    'y_pred':pred}

            
    def fit(self, train_loader, val_loader, max_epochs, patience):
        best_metrics = None
        best_ckpt_path = None

        for epoch in range(1,max_epochs+1):
            train_loss = self.train_one_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            self.wandb_run.log({'epoch': epoch,
                                'train_loss': train_loss,
                                'val_loss': val_metrics['loss'],
                                'val_accuracy': val_metrics['accuracy'],
                                **{f"f1_{k}": v for k, v in val_metrics['per class f1'].items()}})
            self.scheduler.step()
            
            if val_metrics['loss']<self.best_val_loss:
                self.best_val_loss=val_metrics['loss']
                self.epochs_without_improvement=0
                self.save_checkpoint(epoch,val_metrics)
                best_metrics = val_metrics
                best_ckpt_path = self.ckpt_path
            else:
                self.epochs_without_improvement+=1

            print(f"epoch number: {epoch}\tTrain Loss: {train_loss}\tValidation Loss: {val_metrics['loss']}\tValidation Accuracy: {val_metrics['accuracy']}\tLearning Rate: {self.optimizer.param_groups[0]['lr']:.4f}")
            if self.epochs_without_improvement>=patience:
                print(f"Epochs without improvement({self.epochs_without_improvement}) greater than patience({patience}). Stopping early.")
                break
        
        return (best_metrics, best_ckpt_path)

    def save_checkpoint(self,epoch,val_metrics):
        torch.save({'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics,
                    'best_val_loss':self.best_val_loss}, 
                    self.ckpt_path)

    def load_checkpoint(self, ckpt_path):
        checkpoint = torch.load(ckpt_path, map_location=self.device)
        self.best_val_loss = checkpoint['best_val_loss']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        return checkpoint