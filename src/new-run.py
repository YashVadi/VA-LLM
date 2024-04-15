import json
from sklearn.model_selection import train_test_split
from models.model import VALLM
from dataloader_attn import ImgDataset
import pandas as pd
import torch
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

config_path='/home/vdhee/scratch/Nikhil/VA_LLM/configs/config-new.json'
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

train_df, val_df = train_test_split(df, test_size=data_config['test_size'], random_state=24)

# train_df = train_df[:16]
# val_df = val_df[:32]
train_dataset = ImgDataset(train_df, root_dir=data_config['image_data_root'], tokenizer=tokenizer, transform=transforms)
val_dataset = ImgDataset(val_df, root_dir=data_config['image_data_root'], tokenizer=tokenizer, transform=transforms)

# def data_collator(examples):
#     max_tokenized_length = -1
#     tokenized_input = []
#     for i, ex in enumerate(examples):
#         inputs = tokenizer.apply_chat_template(ex, add_generation_prompt=False, return_tensors='pt', return_dict=True, max_length=512)
#         max_tokenized_length = max(max_tokenized_length, inputs['input_ids'].shape[1])
#         tokenized_input.append(inputs)

#     for inputs in tokenized_input:
#         input_ids = inputs['input_ids']
#         attention_mask = inputs['attention_mask']
#         padding_length = max_tokenized_length - input_ids.shape[1]

#         # Add padding tokens
#         padding_tokens = torch.full((input_ids.shape[0], padding_length), tokenizer.eos_token_id, dtype=torch.long)
#         input_ids = torch.cat((input_ids, padding_tokens), dim=1)

#         # Add zeros in the attention mask field
#         padding_attention_mask = torch.zeros((input_ids.shape[0], padding_length), dtype=torch.long)
#         attention_mask = torch.cat((attention_mask, padding_attention_mask), dim=1)

#         # Update inputs dictionary
#         inputs['input_ids'] = input_ids
#         inputs['attention_mask'] = attention_mask

#     # Concatenate input_ids and attention_mask tensors for each example
#     input_ids = torch.cat([ex['input_ids'] for ex in tokenized_input], dim=0)
#     attention_mask = torch.cat([ex['attention_mask'] for ex in tokenized_input], dim=0)

#     batch = {'input_ids': input_ids, 'attention_mask': attention_mask}
#     batch['image'] = torch.stack([example['image'] for example in examples])

#     return batch


trainer = Trainer(
    model=VALLM,
    criterion=criterion,
    train_data=train_dataset,
    val_data=val_dataset,
    collate_fn=None,
    training_config=training_config,
    model_config=model_config,
    logging_config=logging_config,
    data_config=data_config,
)

history = trainer.train()

print(history)