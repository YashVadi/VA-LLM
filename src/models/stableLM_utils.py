def stableLM_modules(model):
    return model.model.layers

def stableLM_embed(model, input_ids):
    return model.model.embed_tokens(input_ids)

def stableLM_ln_lmhead(model, hs):
    return model.lm_head(model.model.norm(hs))