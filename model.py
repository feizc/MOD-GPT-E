import torch 
from torch import nn 
from torch.nn import CrossEntropyLoss, MSELoss 
import os 
from transformers import * 

class MemeDialoGPT(GPT2PreTrainedModel): 
    def __init__(self, config): 
        super(MemeDialoGPT, self).__init__(config) 
        self.transformer = GPT2Model(config) 
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False) 
        self.img_ff = nn.Linear(512, config.n_embd) 
        self.img_inverse_ff = nn.Linear(config.n_embd, 512) 

    def tie_weights(self): 
        self._tie_or_clone_weights(self.lm_head, self.transformer.wte) 
    
    def forward(self, input_embeds, token_type_ids, labels, img_feature): 
        transformer_outputs = self.transformer(input_embeds=input_embeds, token_type_ids=token_type_ids) 
        hidden_states = transformer_outputs[0] 
        txt_hidden_states, img_hidden_states = hidden_states[:-1, :], hidden_states[-1, :].unsqueeze(0) 

        lm_logits = self.lm_head(txt_hidden_states) 
        txt_loss_fct = CrossEntropyLoss(ignore_index=-100) 
        loss = txt_loss_fct(lm_logits, labels) 

        if img_feature[0][0]! = 0.: 
            img_regs = self.img_inverse_ff(img_hidden_states) 
            img_loss_fct = MSELoss() 
            loss += img_loss_fct(img_regs, img_feature) 
        return loss, lm_logits, img_hidden_states  
