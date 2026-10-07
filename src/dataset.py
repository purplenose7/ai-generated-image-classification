from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
from config.settings import MODEL_INPUT_SIZE

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def build_transforms(train):
    base = [
        transforms.Resize((MODEL_INPUT_SIZE, MODEL_INPUT_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    ]
    if train:
        augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
        ]
        base = augmentations + base
    return transforms.Compose(base)


class ForensicsDataset(Dataset):
    def __init__(self, manifest_csv, transform):
        self.df = pd.read_csv(manifest_csv).reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        image = Image.open(row['filepath']).convert('RGB')
        tensor = self.transform(image)
        return tensor, int(row['class_idx'])


def make_dataloader(manifest_csv, train, batch_size, num_workers=2):
    dataset = ForensicsDataset(manifest_csv, build_transforms(train))
    return DataLoader(dataset, batch_size=batch_size, shuffle=train, num_workers=num_workers, 
                      pin_memory=True)