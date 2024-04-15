import json
from sklearn.model_selection import train_test_split
from models.model import VALLM
from dataloader import ImgDataset
import pandas as pd
import torch
from train import Trainer
from transformers import AutoTokenizer
import torch.nn as nn
from PIL import Image
from torchvision.transforms import Compose, Resize, CenterCrop, ToTensor, Normalize
import os

try:
    from torchvision.transforms import InterpolationMode
    BICUBIC = InterpolationMode.BICUBIC
except ImportError:
    BICUBIC = Image.BICUBIC

temp_dir = os.environ["SLURM_TMPDIR"]
config_path = os.environ.get('CONFIG_PATH')
config = json.load(open(config_path))
training_config, model_config, data_config, logging_config = config['training_config'], config['model_config'], config['data_config'], config['logging_config']

criterion = nn.CrossEntropyLoss()

# load the data
def parse_list(column_value):
    return eval(column_value)  # Use eval() to evaluate the string as a Python list

# Read the CSV file with the custom converter function
df = pd.read_csv(data_config['caption_data_path'], converters={'conversations': parse_list})
df = df.rename(columns={'conversations':'caption'})

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

train_df, val_df = train_test_split(df, test_size=data_config['test_size'], random_state=42)

# train_df = train_df[:16]
# val_df = val_df[:32]
train_dataset = ImgDataset(train_df, root_dir=data_config['image_data_root'], tokenizer=tokenizer, transform=transforms)
val_dataset = ImgDataset(val_df, root_dir=data_config['image_data_root'], tokenizer=tokenizer, transform=transforms)

def data_collator(examples):

    # Tokenize all input texts
    tokenized_inputs = [tokenizer.apply_chat_template(ex['text'], add_generation_prompt=False, return_tensors='pt', return_dict=True, max_length=256, truncation=True) for ex in examples]

    # Find the maximum length to pad all sequences to this length
    max_length = max(x['input_ids'].shape[1] for x in tokenized_inputs)

    # Initialize lists for input_ids, attention_masks, and images
    input_ids_list = []
    attention_mask_list = []
    images = []

    for inputs in tokenized_inputs:
        # Calculate the padding length for each sequence
        padding_length = max_length - inputs['input_ids'].shape[1]

        # Efficiently pad input_ids and attention_mask using PyTorch functions
        inputs['input_ids'] = torch.nn.functional.pad(inputs['input_ids'], (0, padding_length), value=tokenizer.eos_token_id)
        inputs['attention_mask'] = torch.nn.functional.pad(inputs['attention_mask'], (0, padding_length), value=0)

        # Append the padded tensors to the lists
        input_ids_list.append(inputs['input_ids'])
        attention_mask_list.append(inputs['attention_mask'])

    # For the images, assuming each 'ex' in 'examples' has an 'image' tensor
    images = [ex['image'] for ex in examples]

    # Stack all input_ids, attention_masks, and images to create batch tensors
    input_ids = torch.cat(input_ids_list, dim=0)
    attention_mask = torch.cat(attention_mask_list, dim=0)
    images = torch.stack(images, dim=0)

    batch = {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'image': images
    }

    return batch


trainer = Trainer(
    model=VALLM,
    criterion=criterion,
    train_data=train_dataset,
    val_data=val_dataset,
    collate_fn=data_collator,
    training_config=training_config,
    model_config=model_config,
    logging_config=logging_config,
    data_config=data_config,
)

history = trainer.train()

print(history)