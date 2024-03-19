import json
from sklearn.model_selection import train_test_split
from models.model import VALLM
from dataloader import ImgDataset
import pandas as pd
from train import Trainer
from transformers import AutoTokenizer
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

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