import torch 
import os 
import json 
import clip 
from PIL import Image 

# load clip model 
device = 'cuda' if torch.cuda.is_available() else 'cpu' 
model, preprocess = clip.load('ViT-B/32', device=device)

# clip feature generation 
def clip_feature_generate(meme_path, img2id_path): 
    # read meme list
    meme_name_list = os.listdir(meme_path) 

    # project meme name to meme id 
    with open(img2id_path, 'r', encoding='utf-8') as f: 
        img2id_dict = json.load(f) 
    # print(img2id_dict.keys())
    
    id2feature = {} 

    for meme_name in meme_name_list: 
        if '.DS_Store' in meme_name:
            continue
        meme_abs_path = os.path.join(meme_path, meme_name) 
        image = preprocess(Image.open(meme_abs_path)).unsqueeze(0).to(device) 
        image_features = model.encode_image(image).tolist()[0] 
        meme_id = img2id_dict[meme_name] 
        id2feature[meme_id] = image_features 
        
    # save the image feature 
    id2feature_path = 'data/meme/id2feature.json'
    with open(id2feature_path, 'w', encoding='utf-8') as f: 
        #json.dump(id2feature, f, indent=4)
        json.dump(id2feature, f)
    




if __name__ == '__main__': 
    meme_path = 'data/meme/image' 
    img2id_path = 'data/meme/img2id.json'
    # clip_feature_generate(meme_path, img2id_path) 
    with open('data/meme/id2feature.json', 'r', encoding='utf-8') as f: 
        id2feature_dict = json.load(f) 
    print(len(id2feature_dict.keys()))
