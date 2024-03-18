import torch
import torch.nn as nn
from transformers import  AutoModelForCausalLM, AutoModel
from transformers import CLIPProcessor, CLIPVisionModel
from torch.nn import functional as F
from transformers.modeling_outputs import CausalLMOutput

from models.attention import MultiHeadCrossAttentionLayer
from models.gpt_utils import gpt_ln_lmhead, gpt_modules, gpt_embed

from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask, AttentionMaskConverter
from typing import List, Tuple, Optional
from torch.nn import CrossEntropyLoss 

class VALLM(nn.Module):
    def __init__(self,
                 config,
                 llm_enc = gpt_embed,
                 llm_blocks = gpt_modules,
                 llm_ln_lmhead = gpt_ln_lmhead
                 ):
        """
        Args:
            llm_enc: function, a function to get the encoder of the language model
            llm_blocks: function, a function to get the blocks of the language model
            llm_ln_lmhead: function, a function to apply the layer normalization and the lm_head given the language model and last hidden state
        """
        super(VALLM, self).__init__()
        self.llm = AutoModelForCausalLM.from_pretrained(config['anchor_model'])
        self.vit = CLIPVisionModel.from_pretrained(config['augmenting_model'])
        self.vit_processor = CLIPProcessor.from_pretrained(config['augmenting_model'])
        
        self.llm_enc = llm_enc
        self.llm_blocks = llm_blocks
        self.llm_ln_lmhead = llm_ln_lmhead

        # freeze the parameters
        if config['freeze_anchor_params']:
            for param in self.llm.parameters():
                param.requires_grad = False

        if config['freeze_augment_params']:
            for param in self.vit.parameters():
                param.requires_grad = False

        self.llm_dim = self.llm.config.hidden_size
        self.vit_dim = self.vit.config.hidden_size
        self.layer_connections = config['layer_connections']

        self.anchor_output_weight = config['anchor_output_weight']
        self.augment_output_weight = config['augment_output_weight']


        self.conn = nn.ModuleList([MultiHeadCrossAttentionLayer(self.llm_dim, self.vit_dim, config['num_attn_heads']) for i in range(len(self.layer_connections))])

        # make conn trainable
        for conn in self.conn:
            for param in conn.parameters():
                param.requires_grad = True

        self.llm_conn = {elem[0]: elem[1] for elem in self.layer_connections}
        self.vit_conn = {elem[1]: elem[0] for elem in self.layer_connections}
        self.connections = {elem[0]: i for i, elem in enumerate(self.layer_connections)}

    def forward(self, input_ids, pixel_values, attention_mask=None, labels=None):
        device = input_ids.device

        cached_vit_hs = {}

        # vit_hs = self.vit_processor(images=pixel_values)
        vit_hs = self.vit.vision_model.embeddings(pixel_values)
        vit_hs = self.vit.vision_model.pre_layrnorm(vit_hs)

        for i, layer_module in enumerate(self.vit.vision_model.encoder.layers):
            vit_hs = layer_module(vit_hs, attention_mask=None, causal_attention_mask=None)[0]
            if i in self.vit_conn.keys():
                cached_vit_hs[i] = vit_hs
            # break if the layer is the last layer we need to cache
            if i == max(self.vit_conn.keys()):
                break   
        
        llm_hs = self.llm_enc(self.llm, input_ids)
        for i, layer_module in enumerate(self.llm_blocks(self.llm)):
            llm_hs = layer_module(llm_hs)[0]
            if i in self.llm_conn.keys():
                llm_hs = self.anchor_output_weight * llm_hs + self.augment_output_weight * self.conn[self.connections[i]](llm_hs , cached_vit_hs[self.llm_conn[i]])
        
        logits = self.llm_ln_lmhead(self.llm, llm_hs)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))


        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )


# model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
# processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# model = VALLM()

# print(model)
# print(torch.tensor([[1,2,3]]).shape, torch.zeros(1, 3, 224, 224).shape)
# model_pre = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
# out = model(torch.ones(2,5).long(), torch.zeros(2, 3, 224, 224).long())
    
# out = AttentionMaskConverter(is_causal=True).to_causal_4d(1,3,3, torch.float32)
# print(out.shape)
