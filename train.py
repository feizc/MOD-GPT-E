from transformers import * 
import os 
import torch 
import json
import numpy as np 

from model import MemeDialoGPT 
from dataset import MODDataset, get_data 
from utils import accuracy_compute, AverageMeter 


SPECIAL_TOKENS = ['[BOS]', '[EOS]', '[speaker1]', '[speaker2]', '[IMG]', '[TAG]', '[PAD]']
SPECIAL_TOKENS_DICT = {'bos_token':'[BOS]', 'eos_token':'[EOS]', 'additional_special_tokens':['[speaker1]', '[speaker2]', '[IMG]', '[TAG]'], 'pad_token':'[PAD]'}

# data parameters
train_data_path = 'data/dialog/toy_data.json'
#train_data_path = 'data/dialog/toy_data.json' 
val_data_path = 'data/dialog/toy_data.json' 
feature_path = 'data/meme/id2feature.json'
#feature_path = 'data/meme/id2feature.json'


# model parameters
use_cuda = torch.cuda.is_available() 
device = torch.device('cuda' if use_cuda else 'cpu') 
model_path = 'ckpt/mod_gpt' 
gpt_path = 'ckpt/origin_gpt'
ckpt_usage = False 
lr = 6e-5
epochs = 1 
gradient_accumulation_steps = 1
print_freq = 1 


def main(): 
    
    # model initialize 
    if ckpt_usage == True: 
        ckpt_path = 'ckpt/mod_gpt/model.bin' 
         

    else:
        tokenizer = GPT2Tokenizer.from_pretrained(gpt_path, do_lower_case=True)
        model = MemeDialoGPT.from_pretrained(gpt_path)
        tokenizer.add_special_tokens(SPECIAL_TOKENS_DICT) 
        model.resize_token_embeddings(len(tokenizer))
    model = model.to(device) 
    optimizer = AdamW(model.parameters(), lr=lr) 

    # data read 
    train_dialogs, id2feature = get_data(tokenizer, train_data_path, feature_path) 
    print(len(train_dialogs))
    val_dialogs, _ = get_data(tokenizer, val_data_path, feature_path) 

    train_dataset = MODDataset(train_dialogs, id2feature, tokenizer) 
    val_dataset = MODDataset(val_dialogs, id2feature, tokenizer) 

    for epoch in range(epochs): 
        
        # one epoch's training
        val_loss = train(model=model, tokenizer=tokenizer, optimizer=optimizer, dataset=train_dataset, epoch=epoch) 
        
        # one epoch's validation 
        

        # save checkpoint 
        torch.save({'model':model.state_dict(), 'optimizer': optimizer.state_dict()},\
            '%s/epoch_%d_loss_%.3f'%(model_path, epoch, val_loss))
        model.config.to_json_file(os.path.join(model_path, 'config.json'))
        tokenizer.save_vocabulary(model_path)


def train(model, tokenizer, optimizer, dataset, epoch): 
    model.train() 
    
    avg_loss = AverageMeter() 
    avg_acc = AverageMeter() 
    iteration = 1

    for instance in dataset: 
        history_txt, history_img, token_type_ids, labels = instance 
        history_txt, history_img, token_type_ids, labels = history_txt.to(device), history_img.to(device), token_type_ids.to(device), labels.to(device) 
        history_txt_embs = model.transformer.wte(history_txt) 
        #print(history_txt_embs.size()) 
        history_img_embs = model.img_ff(history_img) 
        #print(history_img_embs.size()) 
        #print(token_type_ids) 
        #print(history_txt)
        input_embs = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
        input_embs = input_embs.to(device) 
        img_feature = history_img[-1, :].unsqueeze(0)
        # print(input_embs.size()) 
        # print(img_feature.size()) 
        loss, lm_logits, _ = model(input_embs, token_type_ids, labels, img_feature) 
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0) 

        if iteration % gradient_accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
            
        acc = accuracy_compute(lm_logits, labels, 5)
        avg_acc.update(acc)
        avg_loss.update(loss.item())
        
        # print status 
        if iteration % print_freq == 0:
            print('Epoch:[{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Acc {acc.val:.3f} ({acc.avg:.3f})'.format(epoch, iteration, len(dataset),loss=avg_loss, acc=avg_acc))
        
        iteration += 1 

        # print(loss)
        break 
    return avg_loss.avg  


# concatenate the input 
def input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer): 
    bos, eos, speaker1, speaker2, img, tag = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[:-1]) 
    emb_length = token_type_ids.size(-1) 
    emb_dim = history_txt_embs.size(-1)
    img_num = history_img_embs.size(0) 

    input_embs = torch.zeros((emb_length, emb_dim)) 

    txt_idx = 0 
    img_idx = 0 
    left_idx  = 0 
    right_idx = 0 
    while right_idx < emb_length: 
        #if right_idx == emb_length-1 and token_type_ids[right_idx] == img: 
        #    break 
        if right_idx < emb_length-1 and token_type_ids[right_idx] == img:
            txt_length = right_idx - left_idx 
            input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:txt_idx+txt_length, :] 
            txt_idx += txt_length 
            input_embs[right_idx,:] = history_img_embs[img_idx, :] 
            img_idx += 1
            left_idx = right_idx + 1 
        right_idx += 1
    txt_length = right_idx - left_idx 
    if txt_length > 0: 
        input_embs[left_idx:right_idx, :] = history_txt_embs[txt_idx:, :]
    # img_feature = history_img_embs[img_idx,:] 
    return input_embs


def validate(model, tokenizer, dataset, epoch): 
    model.eval() 
    avg_loss = AverageMeter() 
    avg_acc = AverageMeter() 
    avg_bleu = AverageMeter() 

    with torch.no_grad(): 
        for instance in dataset: 
            history_txt, history_img, token_type_ids, labels = instance 
            history_txt, history_img, token_type_ids, labels = history_txt.to(device), history_img.to(device), token_type_ids.to(device), labels.to(device) 
            history_txt_embs = model.transformer.wte(history_txt) 
            history_img_embs = model.img_ff(history_img) 
            
            input_embs = input_construct(history_txt_embs, history_img_embs, token_type_ids, tokenizer) 
            input_embs = input_embs.to(device) 
            img_feature = history_img[-1, :].unsqueeze(0) 
            loss, lm_logits, cur_img_feature = model(input_embs, token_type_ids, labels, img_feature) 




if __name__ == '__main__': 
    main()
