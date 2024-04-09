# create the train loop
import shutil
import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from huggingface_hub import HfApi
import json
from safetensors.torch import load_model, save_model
import numpy as np
from accelerate import Accelerator

import math

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
        gradient_accumulation_steps = training_config['gradient_accumulation_steps']
        mix_precision = training_config['mix_precision']

        ckpt_freq = logging_config['ckpt_freq']
        ckpt_dir = logging_config['ckpt_dir']
        project_name = logging_config['project_name']
        save_model_path = logging_config['save_model_path']

        self.keep_last_n_checkpoints = logging_config['keep_last_n_checkpoints']
        self.push_to_hub = logging_config['push_to_hub']
        if self.push_to_hub:
            self.hf_urername = logging_config['hf_username']

        self.model = model(model_config).to(device)
        
        try:
            if training_config['finetune_ckpt']:
                load_model(self.model, training_config['finetune_ckpt'])
                print(f"Initializing the model from the ckpt : {training_config['finetune_ckpt']}")
            else:
                print(f"Error initializing model from the ckpt :")
                print(f"Initializing the connections from scratch.")
        except:
                print(f"Error initializing model from the ckpt ")
                print(f"Initializing the connections from scratch.")

        # self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_loader = DataLoader(train_data, batch_size=train_batch_size, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_data, batch_size=validation_batch_size, collate_fn=collate_fn)
        # self.scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=num_warmup_steps,
                                                        #  num_training_steps=len(self.train_loader) * n_epochs)
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


        nested_config = {"training_config": training_config, "model_config": model_config, "logging_config": logging_config, "data_config":data_config}
        # wandb.init(project=project_name, config=nested_config)
        # wandb.watch(self.model)
        # self.run_id = wandb.run.id

        if os.environ.get('HF_TOKEN') is None:
            raise ValueError("Please set the HF_TOKEN environment variable to your Hugging Face API token")

        self.accelerator = Accelerator(gradient_accumulation_steps=gradient_accumulation_steps, mixed_precision=mix_precision, log_with="wandb",)
        
        self.accelerator.init_trackers(project_name, {
                                                        "Training Configuration": training_config, 
                                                        "Model Configuration": model_config, 
                                                        "Logging Configuration": logging_config, 
                                                        "Data Configuration": data_config
                                                    })
        
        if self.accelerator.is_main_process:
            tracker = self.accelerator.get_tracker("wandb", unwrap=False)
 
            self.run_id = tracker.run.id
            wandb.watch(self.model)

            self.accelerator.print(f"  Wandb run id from accelerator = {self.run_id}")
            self.ckpt_dir = os.path.join(ckpt_dir, self.run_id)

            os.makedirs(self.ckpt_dir, exist_ok=True)

            if self.push_to_hub:
                self.hf_api = HfApi(token=os.environ['HF_TOKEN'])
                self.hf_repo_id = self.hf_api.create_repo(f"{self.hf_urername}/VA-LLM-{self.run_id}")
                self.hf_repo_id = f"{self.hf_urername}/VA-LLM-{self.run_id}"
                # create a config file 
                with open(os.path.join(self.ckpt_dir, "config.json"), "w") as f:
                    json.dump(nested_config, f)

                #upload the config to the hub
                self.hf_api.upload_file(
                    repo_id= self.hf_repo_id,
                    path_or_fileobj=os.path.join(self.ckpt_dir, "config.json"),
                    path_in_repo="config.json",
                    repo_type="model",
                    run_as_future=True
                )

        print("\n\n\nmain or not :", self.accelerator.is_main_process)

        self.accelerator.wait_for_everyone()

        no_decay = ["bias", "layer_norm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                "weight_decay": weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 
                "weight_decay": 0.0
            },
        ]
        self.optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
        
        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        
        self.max_train_steps = self.n_epochs * num_update_steps_per_epoch
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=self.max_train_steps,
        )

        #prepare the model for distributed training
        self.model, self.optimizer, self.train_loader, self.eval_loader, self.scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.val_loader, self.scheduler
        )

        num_update_steps_per_epoch = math.ceil(len(self.train_loader) / gradient_accumulation_steps)
        self.max_train_steps = self.n_epochs * num_update_steps_per_epoch

        total_batch_size = train_batch_size * self.accelerator.num_processes * gradient_accumulation_steps

        self.accelerator.print("***** Running training *****")
        self.accelerator.print(f"  Instantaneous batch size per device = {train_batch_size}")
        self.accelerator.print(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
        self.accelerator.print(f"  Gradient Accumulation steps = {gradient_accumulation_steps}")
        self.accelerator.print(f"  Total optimization steps = {self.max_train_steps}")

    def logger(self, log_dict):
        self.accelerator.log(log_dict)
        # wandb.log(log_dict)

    def train_epoch(self):
        self.model.train()
        train_loss = 0
        for i, data in enumerate(self.train_loader, 0):
            img = data['image'].to(self.accelerator.device)
            caption = data['input_ids'].to(self.accelerator.device)
            attention_mask = data['attention_mask'].to(self.accelerator.device)
            labels = caption.clone()

            with self.accelerator.accumulate(self.model):
            
                outputs = self.model(input_ids=caption, pixel_values=img, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                train_loss += loss.detach().float()
                self.accelerator.backward(loss)
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
            
            if self.accelerator.sync_gradients:
                self.prog_bar.set_postfix({"Train Loss": train_loss / (i+1)})
                self.prog_bar.update(1)
                self.step += 1

            self.logger({"Train Loss": loss.item(), "Step": self.step})
            if (self.step % self.ckpt_freq == 0) & (self.accelerator.is_main_process):
                self.accelerator.print("saving the checkpoints")
                output_dir = f"step_{self.step}"
                output_dir = os.path.join(self.ckpt_dir,output_dir)
                self.accelerator.save_state(output_dir)
                
                self.delete_old_checkpoints()
                if self.push_to_hub:
                    self.upload_ckpt(output_dir, run_as_future=True)
            
            # self.prog_bar.update(1)

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
             
                self.accelerator.print(f"Real topk pos: {real_topk_pos}")
                # print % of token in top 10, 50, 100, 200, 500
                for k in [5, 10, 50, 100, 200, 500]:
                    top_k_acc[k].append((real_topk_pos < k).sum() / len(real_topk_pos))
        for k in [5, 10, 50, 100, 200, 500]:
            self.accelerator.print(f"Top {k} accuracy: {np.mean(top_k_acc[k])}")


    
    def train(self):
        self.prog_bar = tqdm(range(self.max_train_steps))

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

            self.accelerator.wait_for_everyone()
            if (val_loss <= self.best_loss) & self.accelerator.is_main_process:
                print("saving best model at", os.path.join(self.ckpt_dir, 'best_model'))
                self.best_loss = val_loss
                self.accelerator.save_model(self.model, os.path.join(self.ckpt_dir, 'best_model'))

                if self.push_to_hub:
                    self.upload_ckpt(os.path.join(self.ckpt_dir, 'best_model'), run_as_future=True)

                self.counter = 0
            else:
                self.counter += 1

            if self.counter > self.early_stopping:
                print(f"Early stopping at epoch {epoch}")
                break
    
        self.prog_bar.close()
        # generate 100 from the val_df and log it to wandb with image
        # table = wandb.Table(columns=["Image", "Question", "Generated"])
        # self.model.eval()
        # with torch.no_grad():
        #     for i, data in enumerate(self.val_loader.dataset, 0):
        #         img = data['image'].to(self.device)
        #         caption = data['input_ids'].to(self.device)
        #         attention_mask = data['attention_mask'].to(self.device)
        #         outputs = self.model.generate(input_ids=caption, pixel_values=img, attention_mask=attention_mask, max_length=100)
        #         generated = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        #         table.add_data(img, self.tokenizer.decode(caption[0], skip_special_tokens=True), generated)
        #         if i == 100:
        #             break


        # wandb.finish()
        self.accelerator.end_training()
        
        return self.history
    
    def delete_old_checkpoints(self):
        all_checkpoints = [file for file in os.listdir(self.ckpt_dir) if file.startswith("step")]
        if len(all_checkpoints) > self.max_checkpoints:
            all_checkpoints = sorted(all_checkpoints, key=lambda x: int(x.split("_")[1]))
            for ckpt in all_checkpoints[:-self.keep_last_n_checkpoints]:
                shutil.rmtree(os.path.join(self.ckpt_dir, ckpt))


    def upload_ckpt(self, ckpt_path, run_as_future=False):
        model_name = os.path.basename(ckpt_path)
        self.hf_api.upload_folder(
            repo_id=self.hf_repo_id,
            folder_path=ckpt_path,
            path_in_repo=f"{model_name}",
            run_as_future=run_as_future,
            repo_type="model",
        )

