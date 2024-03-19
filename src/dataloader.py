import torch
from torch.utils.data import Dataset
import os
from PIL import Image
import pandas as pd
import numpy as np
from transformers import AutoTokenizer
from torchvision import transforms

class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, transform=None, max_length=50):
        self.df = df
        self.transform = transform
        self.root_dir = root_dir
        self.tokenizer= tokenizer
        self.max_length = max_length

    def __len__(self,):
        return len(self.df)
    
    def __getitem__(self,idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = os.path.join(self.root_dir , image)

        img = Image.open(img_path).convert("RGB")
        
        if self.transform is not None:
            img= self.transform(img)

        captions = self.tokenizer(caption,
                                 padding='max_length',
                                 max_length=self.max_length,
                                 truncation=True,
                                 return_tensors='pt',)
        return {
                    'image': img,
                    'input_ids': captions['input_ids'].squeeze(0),
                    'attention_mask': captions['attention_mask'].squeeze(0)
                }

# df = pd.read_csv('data/captions.txt')
# # print(df.head())
# tokenizer = AutoTokenizer.from_pretrained("gpt2")
# tokenizer.pad_token = tokenizer.eos_token
# transforms = transforms.Compose(
#     [
#         transforms.Resize((128,128)), 
#         transforms.ToTensor(),
#         transforms.Normalize(
#             mean=0.5, 
#             std=0.5
#         )
#    ]
# )

# dataset = ImgDataset(df,root_dir= "data/images", tokenizer=tokenizer, transform=transforms)
        

# print(dataset[0][0].shape)