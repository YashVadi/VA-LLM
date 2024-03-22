import torch.nn as nn
import torch
from transformers import  AutoModelForCausalLM, AutoModel
from transformers import CLIPProcessor, CLIPVisionModel
from transformers.modeling_outputs import CausalLMOutput

from models.attention import MultiHeadCrossAttentionLayer
from models.gpt_utils import gpt_ln_lmhead, gpt_modules, gpt_embed
from models.stableLM_utils import stableLM_ln_lmhead, stableLM_modules, stableLM_embed
# from attention import MultiHeadCrossAttentionLayer
# from gpt_utils import gpt_ln_lmhead, gpt_modules, gpt_embed
# from stableLM_utils import stableLM_ln_lmhead, stableLM_modules, stableLM_embed
from transformers.modeling_attn_mask_utils import _prepare_4d_causal_attention_mask

from torch.nn import CrossEntropyLoss 

class VALLM(nn.Module):
    def __init__(self,
                 config,
                 llm_enc = stableLM_embed,
                 llm_blocks = stableLM_modules,
                 llm_ln_lmhead = stableLM_ln_lmhead
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
        
        batch_size, seq_length = input_ids.shape

        llm_hs = self.llm_enc(self.llm, input_ids)

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), device=device)

        position_ids = torch.arange(seq_length, dtype=torch.long, device=device).unsqueeze(0).expand(batch_size, -1)

        attention_mask_4d = _prepare_4d_causal_attention_mask(
            attention_mask, (batch_size, seq_length), llm_hs, 0
        )
        
        for i, layer_module in enumerate(self.llm_blocks(self.llm)):
            llm_hs = layer_module(llm_hs,attention_mask=attention_mask_4d, position_ids=position_ids)[0]
            if i in self.llm_conn.keys():
                llm_hs = self.anchor_output_weight * llm_hs + self.augment_output_weight * self.conn[self.connections[i]](llm_hs , cached_vit_hs[self.llm_conn[i]])
        
        logits = self.llm_ln_lmhead(self.llm, llm_hs)

        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_masks = attention_mask[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction="none")
            loss_batch = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            loss = torch.sum(loss_batch * shift_masks.view(-1)) / torch.sum(shift_masks) 


        return CausalLMOutput(
            loss=loss,
            logits=logits,
            hidden_states=None,
            attentions=None,
        )