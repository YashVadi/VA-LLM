# create the train loop
import pandas as pd
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.nn import functional as F
from tqdm import tqdm
import json
from PIL import Image

from sklearn.model_selection import train_test_split
from models.model import VALLM
from dataloader import ImgDataset
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

import wandb

# train the model
class Trainer:
    def __init__(self, 
                 model,
                 criterion,  
                 train_data,
                 val_data,
                 training_config,
                 model_config,
                 logging_config,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 verbose=True,
                 print_every=1,
                 ):

        lr = training_config['learning_rate']
        num_warmup_steps = training_config['warmup_steps']
        n_epochs = training_config['epochs']
        early_stopping = training_config['early_stopping']
        weight_decay = training_config['weight_decay']
        batch_size = training_config['batch_size']

        ckpt_freq = logging_config['ckpt_freq']
        ckpt_dir = logging_config['ckpt_dir']
        project_name = logging_config['project_name']
        save_model_path = logging_config['save_model_path']

        self.model = model(model_config).to(device)
        self.criterion = criterion
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        self.val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=True)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                         num_training_steps=len(self.train_loader) * n_epochs)
        self.device = device
        self.n_epochs = n_epochs
        self.early_stopping = early_stopping
        self.verbose = verbose
        self.best_loss = float("inf")
        self.print_every = print_every
        self.counter = 0
        self.history = {"train_loss": [], "val_loss": []}
        self.step = 0
        self.prog_bar = None

        self.ckpt_freq = ckpt_freq
        self.max_checkpoints = 10
        self.save_model_path = save_model_path

        nested_config = {"training_config": training_config, "model_config": model_config, "logging_config": logging_config}
        wandb.init(project=project_name, config=nested_config)
        wandb.watch(self.model)
        self.run_id = wandb.run.id
        self.ckpt_dir = os.path.join(ckpt_dir, self.run_id)
        os.makedirs(self.ckpt_dir, exist_ok=True)

        
    def logger(self, log_dict):
        wandb.log(log_dict)

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for i, data in enumerate(self.train_loader, 0):
            img = data['image'].to(self.device)
            caption = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=caption)
            loss = outputs.loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss.item()

            self.step += 1
            self.logger({"Train Loss": loss.item(), "Step": self.step})
            if self.step % self.ckpt_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "ckpt_" + str(self.step) + ".pth"))
            ## delete old checkpoints if there are more than max_checkpoints
            self.prog_bar.set_postfix({"Train Loss": train_loss / (i+1)})
            self.prog_bar.update(1)

        return train_loss / len(self.train_loader)
    
    def val_epoch(self):
        self.model.eval()
        val_loss = 0
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                img = data['image'].to(self.device)
                caption = data['input_ids'].to(self.device)
                attention_mask = data['attention_mask'].to(self.device)
                outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=caption)
                loss = outputs.loss
                val_loss += loss.item()

        return val_loss / len(self.val_loader)
    
    def train(self):
        self.prog_bar = tqdm(range(len(self.train_loader)*self.n_epochs))

        #print total number of parameters  and trainable params in the model
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        self.logger({"Total Parameters": total_params, "Trainable Parameters": trainable_params})

        for epoch in range(self.n_epochs):
            train_loss = self.train_epoch()
            val_loss = self.val_epoch()

            self.logger({"Epoch": epoch, "Train Loss": train_loss, "Val Loss": val_loss})
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)

            if val_loss < self.best_loss:
                self.best_loss = val_loss
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, self.save_model_path))
                self.counter = 0
            else:
                self.counter += 1

            if self.counter > self.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
        wandb.finish()
        
        return self.history

config_path='src/config.json'
config = json.load(open(config_path))
training_config, model_config, data_config, logging_config = config['training_config'], config['model_config'], config['data_config'], config['logging_config']

criterion = nn.CrossEntropyLoss()

# load the data
df = pd.read_csv(data_config['caption_data_path'])
tokenizer = AutoTokenizer.from_pretrained(model_config['anchor_model'])
tokenizer.pad_token = tokenizer.eos_token

# Ref: https://github.com/openai/CLIP/blob/main/clip/clip.py#L79
def _convert_image_to_rgb(image):
    return image.convert("RGB")

transforms = Compose([
        Resize((224,224), interpolation=BICUBIC),
        CenterCrop((224,224)),
        _convert_image_to_rgb,
        ToTensor(),
        Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
    ])

train_df, val_df = train_test_split(df, test_size=data_config['test_size'])

train_dataset = ImgDataset(train_df, root_dir=data_config['image_data_root'], tokenizer=tokenizer, transform=transforms)
val_dataset = ImgDataset(val_df, root_dir=data_config['image_data_root'], tokenizer=tokenizer, transform=transforms)

trainer = Trainer(
    model=VALLM,
    criterion=criterion,
    train_data=train_dataset,
    val_data=val_dataset,
    training_config=training_config,
    model_config=model_config,
    logging_config=logging_config
)

history = trainer.train()

print(history)