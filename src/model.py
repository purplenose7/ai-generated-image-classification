from torchvision.models import resnet50
"""
6. **`build_resnet(num_classes, mode)`** · (`mode` in `{"feature_extraction","finetune"}`) → model
  Algorithm: load ResNet-50 pretrained; find its final-layer input feature count yourself + 
  swap in a fresh head for `num_classes` (add head regularization if you choose); if `feature_extraction`, 
  freeze backbone (head trainable); if `finetune`, all trainable; return.
7. **`count_parameters(model)`** · → (trainable, total).

**The work:** build ResNet in feature-extraction mode; call `fit`; checkpoint each epoch + Save 
  Version. Pick head LR via the values-box method (fresh head tpakes a normal LR). 
  Then `evaluate` on the **test** split.

**Decisions to make:** head regularization; optimizer/LR/schedule/epochs/patience (values box).
"""

resnet = resnet50(weights='IMAGENET1K_V1')
num_feats = resnet.fc.in_features

def build_resnet(resnet, num_feats, mode, device):
  if mode=='feature_extraction':
    for param in resnet.parameters():
      param.requires_grad = False
  resnet.fc = nn.Linear(num_feats, 3) #whwther to add dropout before the linear layer???
  resnet = resnet.to(device)

  return resnet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
resnet = build_resnet(resnet,num_feats,'feature_extraction',???)

"""Then in your notebook, separately:
Call build_resnet to get the model
Create your optimizer, scheduler, loss function
Create ForensicsTrainer
Call trainer.fit(train_loader, val_lo ader, max_epochs, patience)
Call trainer.evaluate(test_loader)"""