import torch 
from PIL import Image
import os
import torch.nn as nn 
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
from glob import glob
import configs
from tqdm import tqdm

class ImgClsDataset(Dataset):
    def __init__(self, root_dir, img_height, img_width, mode, transforms = None):
        self.data_dir = os.path.join(root_dir, mode)
        
        self.img_height = img_height 
        self.img_width = img_width 

        if transforms: 
            self.transforms = transforms 
        else: 
            if mode == "train":
                self.transforms = T.Compose([T.RandomResizedCrop((configs.img_height, configs.img_width), interpolation=T.InterpolationMode.BICUBIC),
                                            T.RandomHorizontalFlip(0.5), 
                                            T.RandomVerticalFlip(0.5), 
                                            T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2),
                                            T.ToTensor(),
                                            T.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513,0.26158784))])
            elif mode == "test" or mode == "val":
                self.transforms = T.Compose([T.Resize((self.img_height, self.img_width), T.InterpolationMode.BICUBIC),
                                            T.ToTensor(),
                                            T.Normalize((0.49139968, 0.48215841, 0.44653091),(0.24703223, 0.24348513,0.26158784))])
        self.id_to_class_dict = {} 
        self.class_to_id_dict = {}
        self.images = []

        self.valid_dir = [cls for cls in os.listdir(self.data_dir) if os.path.isdir(os.path.join(self.data_dir, cls)) and not cls.startswith('.')]

        for i, cls in enumerate(self.valid_dir):
            cls_path = os.path.join(self.data_dir, cls)
            self.id_to_class_dict[i] = cls
            self.class_to_id_dict[cls] = i
        
            for image in glob(os.path.join(cls_path, "*")):
                self.images.append((image, i))

    def __getitem__(self, index):
        img_path, img_id = self.images[index]
        img = self.transforms(Image.open(img_path).convert("RGB"))
        return img, img_id

    def __len__(self):
        return len(self.images)
    
def load_dataset(root_dir=configs.root_dir, img_height = configs.img_height, img_width = configs.img_width, batch_size = configs.batch_size, shuffle = True, mode = "train"): 
    assert mode in ["train", "val", "test"], "[ERROR] Invalid dataset mode"
    ds = ImgClsDataset(root_dir, img_height, img_width, mode = mode, transforms = configs.transforms)
    configs.num_class = len(ds.id_to_class_dict)
    configs.id_to_category_dict = ds.id_to_class_dict
    configs.category_to_id_dict = ds.class_to_id_dict
    return DataLoader(dataset=ds, batch_size=batch_size, shuffle=shuffle, drop_last=True)