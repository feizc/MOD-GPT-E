import torch 
import os 
import clip 
from PIL import Image 

# load clip model 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model, preprocess = clip.load('ViT-B/32', device=device)

# clip feature generation 
def clip_feature_generate(meme_path): 
    meme_name_list = os.listdir(meme_path) 
    print(len(meme_name_list)) 
    


if __name__ == '__main__': 
    meme_path = 'data/meme/image' 
    clip_feature_generate(meme_path)
