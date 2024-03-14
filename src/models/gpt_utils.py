import torch
import torch.nn as nn
from torch.nn import functional as F

def gpt_ln_lmhead(model, hs):
    hs = model.transformer.ln_f(hs)
    hs = model.lm_head(hs)
    return hs

def gpt_modules(model):
    return model.transformer.h

def gpt_embed(model, ids):
    inputs_embeds = model.transformer.wte(ids)
    position_embeds = model.transformer.wpe(torch.arange(0, ids.size()[-1], dtype=torch.long, device=ids.device))
    return inputs_embeds + position_embeds
