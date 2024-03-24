# create the train loop
import os
import torch
from torch.utils.data import DataLoader
from transformers import AdamW, get_linear_schedule_with_warmup
from tqdm import tqdm
from huggingface_hub import HfApi

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
        self.criterion = criterion
        self.optimizer = AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, collate_fn=collate_fn)
        self.val_loader = DataLoader(val_data, batch_size=validation_batch_size, shuffle=False, collate_fn=collate_fn)
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
            self.hf_repo_id = self.hf_api.create_repo(f"{self.hf_urername}/VA-LLM-{self.run_id}")

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
                # torch.save(self.model.state_dict(), os.path.join(self.ckpt_dir, "ckpt_" + str(self.step) + ".pth"))
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
                torch.save(self.model.conn.state_dict(), os.path.join(self.ckpt_dir, self.save_model_conn_path))
                if self.push_to_hub:
                    self.upload_ckpt(os.path.join(self.ckpt_dir, self.save_model_path), run_as_future=True)
                    self.upload_ckpt(os.path.join(self.ckpt_dir, self.save_model_conn_path), run_as_future=True)
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

