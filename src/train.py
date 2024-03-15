# create the train loop
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from torch.nn import functional as F

from models.model import VALLM
from dataloader import ImgDataset
from torchvision import transforms

import wandb

# load the data
df = pd.read_csv('data/captions.txt')
tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token   

transforms = transforms.Compose(
    [
        transforms.Resize((224,224)), 
        transforms.ToTensor(),
        transforms.Normalize(
            mean=0.5, 
            std=0.5
        )
   ]
)


train_df, val_df = df[:int(0.8*len(df))], df[int(0.8*len(df)):]
# train_df, val_df = df[:100], df[-100:]


train_dataset = ImgDataset(train_df,root_dir= "data/images", tokenizer=tokenizer, transform=transforms)
val_dataset = ImgDataset(val_df,root_dir= "data/images", tokenizer=tokenizer, transform=transforms)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=True)

# load the model
model = VALLM()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# train the model
class Trainer:
    def __init__(self, 
                 model, 
                 criterion, 
                 optimizer, 
                 train_loader, 
                 val_loader, 
                 device, 
                 n_epochs=10, 
                 early_stopping=5, 
                 verbose=True,
                 print_every=1, 
                 ckpt_dir="./", 
                 save_model_path="./best_model.pth"):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.ckpt_dir = ckpt_dir
        self.save_model_path = save_model_path
        self.best_loss = float("inf")
        self.print_every = print_every
        self.counter = 0
        self.history = {"train_loss": [], "val_loss": []}
        self.step = 0
        
    def logger(self, log_dict):
        if self.verbose and self.step % self.print_every == 0:
            # for message in log_dict:
            print(str([message + ": " + str(log_dict[message]) for message in log_dict]))
        wandb.log(log_dict)

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for i, data in enumerate(self.train_loader, 0):
            img = data['image']
            caption = data ['input_ids'] 
            attention_mask = data['attention_mask']

            self.optimizer.zero_grad()
            outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=caption)
            loss = outputs.loss

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()

            self.step += 1
            self.logger({"Train Loss": loss.item(), "Step": self.step})

        return train_loss / len(self.train_loader)
    
    def val_epoch(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                img = data['image']
                caption = data ['input_ids'] 
                attention_mask = data['attention_mask']
                outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=caption)
                loss = outputs.loss
                val_loss += loss.item()

        return val_loss / len(self.val_loader)
    
    def train(self):
        wandb.init(project="image-captioning")

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()

            self.logger({"Epoch": epoch, "Train Loss": train_loss, "Val Loss": val_loss, "Epoch": epoch})
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), self.save_model_path)
                self.counter = 0
            else:
                self.counter += 1

            if self.counter > self.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        wandb.finish()
        
        return self.history

trainer = Trainer(model, criterion, optimizer, train_loader, val_loader, "cuda")

history = trainer.train()

print(history)