# create the train loop
import numpy as np
import pickle
import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup, AutoTokenizer
from tqdm import tqdm
from huggingface_hub import HfApi
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

import wandb

# train the model
class Trainer:
    def __init__(self, 
                 model,
                 criterion,  
                 train_data,
                 val_data,
                 collate_fn,
                 training_config,
                 model_config,
                 logging_config,
                 data_config,
                 device = "cuda" if torch.cuda.is_available() else "cpu",
                 verbose=True,
                 print_every=1,
                 ):

        lr = training_config['learning_rate']
        num_warmup_steps = training_config['warmup_steps']
        n_epochs = training_config['epochs']
        early_stopping = training_config['early_stopping']
        weight_decay = training_config['weight_decay']
        train_batch_size = training_config['train_batch_size']
        validation_batch_size = training_config['validation_batch_size']

        ckpt_freq = logging_config['ckpt_freq']
        ckpt_dir = logging_config['ckpt_dir']
        project_name = logging_config['project_name']
        save_model_path = logging_config['save_model_path']
        save_model_conn_path = logging_config['save_model_conn_path']
        self.keep_last_n_checkpoints = logging_config['keep_last_n_checkpoints']
        self.push_to_hub = logging_config['push_to_hub']
        if self.push_to_hub:
            self.hf_urername = logging_config['hf_username']

        self.model = model(model_config).to(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_config['anchor_model'])
        self.criterion = criterion
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.collate_fn = collate_fn
        if collate_fn == None:
            self.train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=4)
            self.val_loader = DataLoader(val_data, batch_size=validation_batch_size, shuffle=False, num_workers=4)
        else :
            self.train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn, num_workers=1)
            self.val_loader = DataLoader(val_data, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn, num_workers=1)
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
        self.save_model_conn_path = save_model_conn_path
        self.best_model = self.model

        nested_config = {"training_config": training_config, "model_config": model_config, "logging_config": logging_config, "data_config":data_config}
        wandb.init(project=project_name, config=nested_config)
        wandb.watch(self.model)
        self.run_id = wandb.run.id
        self.ckpt_dir = os.path.join(ckpt_dir, self.run_id)

        os.makedirs(self.ckpt_dir, exist_ok=True)
        
        if os.environ.get('HF_TOKEN') is None:
            raise ValueError("Please set the HF_TOKEN environment variable to your Hugging Face API token")
        
        if self.push_to_hub:
            self.hf_api = HfApi(token=os.environ['HF_TOKEN'])
            self.hf_api.create_repo(f"{self.hf_urername}/VA-LLM-{self.run_id}")
            self.hf_repo_id = f"{self.hf_urername}/VA-LLM-{self.run_id}"

    def logger(self, log_dict):
        wandb.log(log_dict)

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        scaler = GradScaler()  # Create a GradScaler for automatic mixed precision

        for i, data in enumerate(self.train_loader, 0):
            img = data['image'].to(self.device)
            caption = data['input_ids'].to(self.device)
            attention_mask = data['attention_mask'].to(self.device)
            labels = caption.clone()

            
            self.optimizer.zero_grad()
            outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()
            train_loss += loss.item()
            # with autocast():  # Autocast for mixed precision
            #     outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=labels)
            #     loss = outputs.loss

            # scaler.scale(loss).backward()  # Scale the loss and backpropagate
            # scaler.step(self.optimizer)  # Update the optimizer with the scaled gradients
            # scaler.update()  # Update the GradScaler for the next iteration
            # self.scheduler.step()

            self.step += 1
            train_lr = self.optimizer.param_groups[0]['lr']
            self.logger({"Train Loss": loss.item(), "Learning Rate": train_lr, "Step": self.step})
            # self.logger({"Train Loss": loss.item(), "Step": self.step})
            if self.step % self.ckpt_freq == 0:
                torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "ckpt_" + str(self.step) + ".pth"))
                torch.save(self.model.conn.state_dict(), os.path.join(self.ckpt_dir, "conn_ckpt_" + str(self.step) + ".pth"))
                self.delete_old_checkpoints()
                if self.push_to_hub:
                    self.upload_ckpt(os.path.join(self.ckpt_dir, "conn_ckpt_" + str(self.step) + ".pth"), run_as_future=True)

            self.prog_bar.set_postfix({"Train Loss": train_loss / (i+1)})
            self.prog_bar.update(1)

        return train_loss / len(self.train_loader)
    
    def val_epoch(self):
        self.model.eval()
        val_loss = 0
        top_k_acc = {5:[], 10: [], 50: [], 100: [], 200: [], 500: []}
        with torch.no_grad():
            for i, data in enumerate(self.val_loader, 0):
                img = data['image'].to(self.device)
                caption = data['input_ids'].to(self.device)
                labels = caption.clone()
                attention_mask = data['attention_mask'].to(self.device)
                outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                val_loss += loss.item()

                token_ground_truth = labels[0,:]
                token_logits= outputs.logits[0,:]
                #apply mask to the logits and ground truth
                mask = attention_mask[0,:]
                token_logits = token_logits[mask]
                token_ground_truth = token_ground_truth[mask]
                sorted_logits = torch.argsort(token_logits, descending=True, dim=1)
                real_topk_pos = np.array([int(torch.where(sorted_logits[i] == token_ground_truth[i].item())[0][0])for i in range(token_ground_truth.shape[0])])

                #save the input image and caption and model outputs for first batch for debugging in pickle file
                if i%100== 0:
                    with open(os.path.join(self.ckpt_dir, f"dump_{i}.pkl"), "wb") as f:
                        pickle.dump({"image": img, "caption": caption, "outputs": outputs}, f)

                print(f"Real topk pos: {real_topk_pos}")
                # print % of token in top 10, 50, 100, 200, 500
                for k in [5, 10, 50, 100, 200, 500]:
                    top_k_acc[k].append((real_topk_pos < k).sum() / len(real_topk_pos))
        for k in [5, 10, 50, 100, 200, 500]:
            print(f"Top {k} accuracy: {np.mean(top_k_acc[k])}")

    

    
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
                torch.save(self.model.conn.state_dict(), os.path.join(self.ckpt_dir, self.save_model_conn_path))
                self.best_model = self.model
                if self.push_to_hub:
                    self.upload_ckpt(os.path.join(self.ckpt_dir, self.save_model_path), run_as_future=False)
                    self.upload_ckpt(os.path.join(self.ckpt_dir, self.save_model_conn_path), run_as_future=False)
                self.counter = 0
            else:
                self.counter += 1

            if self.counter > self.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
    
        self.prog_bar.close()
        # generate 100 from the val_df and log it to wandb with image
        table = wandb.Table(columns=["Image", "Question", "Generated"])
        self.model.load_state_dict(torch.load(os.path.join(self.ckpt_dir, self.save_model_path), map_location=self.device))

        self.model.eval()
        with torch.no_grad():
            for i, data in tqdm(enumerate(self.train_loader.dataset, 0)):
                img = data['image'].to(self.device)
                # caption = data['input_ids'].to(self.device)
                if self.collate_fn is None:
                    caption = self.tokenizer("Question: What do you see happening in the image?\nAnswer: ", return_tensors='pt').input_ids.to(self.device)
                else :
                    prompt = [
                        {
                            "role": "system",
                            "content": "You are a text-based model that has been enhanced with the unique ability to access and interpret the outputs from in-between layers of a CLIP model. This enhancement allows you to understand and analyze images through the intermediary representations produced by CLIP, despite not being able to 'see' the images directly. You use this information to provide detailed and insightful responses to questions about the content and context of images."
                        },
                        {
                            "role": "user",
                            "content": "What do you see in this image?"
                        }
                    ]
                    caption = self.tokenizer.apply_chat_template(prompt,  add_generation_prompt=True, return_tensors="pt").to(self.device)
                outputs = self.model.generate(input_ids=caption, pixel_values=img.unsqueeze(0), max_length=100)
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                log_img = wandb.Image(Image.open(data['image_path']).convert("RGB"))
                table.add_data(log_img, self.tokenizer.decode(caption[0], skip_special_tokens=True), generated)
                if i == 3:
                    break
            for i, data in tqdm(enumerate(self.val_loader.dataset, 0)):
                img = data['image'].to(self.device)
                # caption = data['input_ids'].to(self.device)
                if self.collate_fn is None:
                    caption = self.tokenizer("Question: What do you see happening in the image?\nAnswer: ", return_tensors='pt').input_ids.to(self.device)
                else :
                    prompt = [
                        {
                            "role": "system",
                            "content": "You are a text-based model that has been enhanced with the unique ability to access and interpret the outputs from in-between layers of a CLIP model. This enhancement allows you to understand and analyze images through the intermediary representations produced by CLIP, despite not being able to 'see' the images directly. You use this information to provide detailed and insightful responses to questions about the content and context of images."
                        },
                        {
                            "role": "user",
                            "content": "What do you see in this image?"
                        }
                    ]
                    caption = self.tokenizer.apply_chat_template(prompt, add_generation_prompt=True, return_tensors="pt").to(self.device)
                outputs = self.model.generate(input_ids=caption, pixel_values=img.unsqueeze(0), max_length=100)
                generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                log_img = wandb.Image(Image.open(data['image_path']).convert("RGB"))
                table.add_data(log_img, self.tokenizer.decode(caption[0], skip_special_tokens=True), generated)
                if i == 5:
                    break
                
        wandb.log({"Examples": table})


        wandb.finish()
        
        return self.history
    
    def delete_old_checkpoints(self):
        all_checkpoints = os.listdir(self.ckpt_dir)
        all_checkpoints = [os.path.join(self.ckpt_dir, ckpt) for ckpt in all_checkpoints]
        all_checkpoints = sorted(all_checkpoints, key=os.path.getctime, reverse=True)
        for ckpt in all_checkpoints[self.keep_last_n_checkpoints:]:
            os.remove(ckpt)

    def upload_ckpt(self, ckpt_path, run_as_future=False):
        model_name = os.path.basename(ckpt_path)
        self.hf_api.upload_file(
            repo_id=self.hf_repo_id,
            path_or_fileobj=ckpt_path,
            path_in_repo=f"model/{model_name}",
            run_as_future=run_as_future,
            repo_type="model",
        )

