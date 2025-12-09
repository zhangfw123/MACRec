from tqdm import tqdm
import glob
import os
from clip import clip
import torch
from PIL import Image
import numpy as np
import argparse
import json

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def get_id2item_dict(item2id_file):
    
    with open(item2id_file, 'r') as fp:
        all_item2id = fp.readlines()
        
    id2item = {}
    for line in all_item2id:
        item, item_id = line.strip().split('\t')
        
        id2item[item_id] = item
        
    return id2item

def get_feature(args):
    print(args.dataset, args.backbone)

    image_file_path = f'{args.image_root}/{args.dataset}'
    save_path = f'{args.save_root}/{args.dataset}'
    
    item2id_file = os.path.join(save_path, f'{args.dataset}.item2id')
    id2item = get_id2item_dict(item2id_file)
    
    # print(id2item[str(0)])
    
    images_info = load_json(os.path.join(args.image_root, f'{args.dataset}_images_info.json'))
    
    # os.makedirs(save_path, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    print('Load model.')
    model, preprocess = clip.load(args.backbone, device=device, download_root=args.model_cache_dir)

    embeddings = []

    with torch.no_grad():
        for i in tqdm(range(len(id2item))):
            item = id2item[str(i)]
            image_name = images_info[item][0]
            image_file = os.path.join(image_file_path, image_name)

            try:
                image = Image.open(image_file)
                image = preprocess(image).unsqueeze(0).to(device)
            except Exception as e:
                print("Error type:", type(e))
                print(e)
                print(image_file)

            image_features = model.encode_image(image)
            image_feature = image_features[0].cpu()
            
            embeddings.append(image_feature)
            
            # break
            
    embeddings = torch.stack(embeddings, dim=0).numpy()
    print('Embeddings shape: ', embeddings.shape)

    backbone_name = args.backbone.replace('/', '-')
    
    file = os.path.join(save_path, args.dataset + '.emb-' + backbone_name + ".npy")
    
    np.save(file, embeddings)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Instruments', help='Instruments / Arts / Games')
    parser.add_argument('--image_root', type=str, default="/userhome/dataset/amazon18/Images")
    parser.add_argument('--save_root', type=str, default="/userhome/dataset/MQL4GRec")
    parser.add_argument('--gpu_id', type=int, default=0, help='ID of running GPU')
    parser.add_argument('--backbone', type=str, default='ViT-L/14')
    parser.add_argument('--model_cache_dir', type=str, default='/userhome/cache_models/clip')
    return parser.parse_args()
    
if __name__ == "__main__":
    args = parse_args()
    get_feature(args)