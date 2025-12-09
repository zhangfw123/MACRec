import collections
import json
import logging

import numpy as np
import torch
import copy
from tqdm import tqdm
import argparse
from collections import defaultdict

from torch.utils.data import DataLoader

from datasets import EmbDataset, DualEmbDataset
from models.rqvae import CrossRQVAE

import os

def parse_args():
    parser = argparse.ArgumentParser(description="Index")
    parser.add_argument('--text_data_path', type=str, default=None)
    parser.add_argument('--image_data_path', type=str, default=None)
    parser.add_argument('--dataset', type=str, default=None)
    parser.add_argument('--ckpt_path', type=str, default=None)
    parser.add_argument('--output_dir', type=str, default=None)
    parser.add_argument('--output_file', type=str, default=None)
    parser.add_argument('--content', type=str, default=None)
    parser.add_argument('--device', type=str, default="cuda:0")
    return parser.parse_args()

def check_collision(all_indices_str):
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    return tot_item==tot_indice

def get_indices_count(all_indices_str):
    indices_count = collections.defaultdict(int)
    for index in all_indices_str:
        indices_count[index] += 1
    return indices_count

def get_collision_item(all_indices_str):
    index2id = {}
    for i, index in enumerate(all_indices_str):
        if index not in index2id:
            index2id[index] = []
        index2id[index].append(i)

    collision_item_groups = []

    for index in index2id:
        if len(index2id[index]) > 1:
            collision_item_groups.append(index2id[index])

    return collision_item_groups

args = parse_args()

dataset = args.dataset
ckpt_path = args.ckpt_path
output_dir = args.output_dir
output_file = args.output_file
output_file = os.path.join(output_dir, output_file)
device = torch.device(args.device)

if args.content == 'image':
    prefix = ["<A_{}>","<B_{}>","<C_{}>","<D_{}>","<E_{}>"]
else:
    prefix = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>","<e_{}>"]

ckpt = torch.load(ckpt_path, map_location=torch.device('cpu'))
args = ckpt["args"]
state_dict = ckpt["state_dict"]

cmd_args = parse_args()
args.content = cmd_args.content

data = DualEmbDataset(args.text_data_path,args.image_data_path)

model = CrossRQVAE(text_in_dim=data.text_dim,
                  image_in_dim=data.img_dim,
                  num_emb_list=args.num_emb_list,
                  e_dim=args.e_dim,
                  layers=args.layers,
                  dropout_prob=args.dropout_prob,
                  bn=args.bn,
                  loss_type=args.loss_type,
                  quant_loss_weight=args.quant_loss_weight,
                  kmeans_init=args.kmeans_init,
                  kmeans_iters=args.kmeans_iters,
                  sk_epsilons=args.sk_epsilons,
                  sk_iters=args.sk_iters,
                  use_cross_rq=args.use_cross_rq,
                  begin_cross_layer=args.begin_cross_layer,
                  )

model.load_state_dict(state_dict)
model = model.to(device)
model.eval()
print(model)

data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=64, shuffle=False,
                             pin_memory=True)

all_indices = []
all_indices_str = []
all_distances = [[] for i in range(len(args.num_emb_list))]
all_indices_str_set = set()

for batch_idx, (text_d,img_d,_) in tqdm(enumerate(data_loader)):
    text_d = text_d.to(device)
    img_d = img_d.to(device)
    text_indices, image_indices, text_distances, image_distances = model.get_indices(text_d, img_d, use_sk=False)
    if args.content == 'image':
        indices = image_indices
        distances = image_distances
    else:
        indices = text_indices
        distances = text_distances
    indices = indices.view(-1, indices.shape[-1]).cpu().numpy()
    for index in indices:
        code = []
        for i, ind in enumerate(index):

            code.append(int(ind))

        all_indices.append(code)
        all_indices_str.append(str(code))
        all_indices_str_set.add(str(code))

    for k in range(len(all_distances)):
        all_distances[k].extend(distances[k])





for i in all_indices_str_set:
    print(i)
    break

for k in range(len(all_distances)):
    print(len(all_distances[k]))


sort_distances_index = [np.argsort(all_distances[k]) for k in range(len(all_distances))]

item_min_dis = defaultdict(list)

for k in range(len(all_distances)):
    for item, distances in tqdm(enumerate(all_distances[k]), desc='cal distances'):
        item_min_dis[item].append(np.min(distances))



    
collision_item_groups = get_collision_item(all_indices_str)
all_collision_items = set()
for collision_items in collision_item_groups:
    for item in collision_items:
        all_collision_items.add(item)
        
print('collision items num: ', len(all_collision_items))



tt = 0
level = len(args.num_emb_list) - 1
max_num = args.num_emb_list[0]

while True:
    tot_item = len(all_indices_str)
    tot_indice = len(set(all_indices_str))
    print(f'tot_item: {tot_item}, tot_indice: {tot_indice}')
    print("Collision Rate",(tot_item-tot_indice)/tot_item)
    
    if check_collision(all_indices_str) or tt == 2:
        print('tt', tt)
        break

    collision_item_groups = get_collision_item(all_indices_str)
    print(collision_item_groups)
    print(len(collision_item_groups))
    
    
    for collision_items in collision_item_groups:
        
        min_distances = []
        for i, item in enumerate(collision_items):
            min_distances.append(item_min_dis[item][level])


        min_index = np.argsort(np.array(min_distances))
        
        for i, m_index in enumerate(min_index):
            
            if i == 0:
                continue
            
            item = collision_items[m_index]
            # print(item)
            
            ori_code = copy.deepcopy(all_indices[item])
            # print(ori_code)
            
            num = i
            while str(ori_code) in all_indices_str_set and num < max_num:

                ori_code[level] = sort_distances_index[level][item][num]
                num += 1
            for i in range(1, max_num):
                if str(ori_code) in all_indices_str_set:
                    ori_code = copy.deepcopy(all_indices[item])
                    ori_code[level-1] = sort_distances_index[level-1][item][i]
                    
                num = 0
                while str(ori_code) in all_indices_str_set and num < max_num:

                    ori_code[level] = sort_distances_index[level][item][num]
                    num += 1
                    
                if str(ori_code) not in all_indices_str_set:
                    break
                
            all_indices[item] = ori_code
            all_indices_str[item] = str(ori_code)

            all_indices_str_set.add(str(ori_code))

            # print(str(ori_code))
        
        
    # if level == 2:
    #     break
    tt += 1


print("All indices number: ",len(all_indices))
all_indices_str = [str(indice) for indice in all_indices]
print("Max number of conflicts: ", max(get_indices_count(all_indices_str).values()))

tot_item = len(all_indices_str)
tot_indice = len(set(all_indices_str))
print("Collision Rate",(tot_item-tot_indice)/tot_item)

all_indices_dict = {}
for item, indices in enumerate(all_indices):
    code = []
    for i, ind in enumerate(indices):
        code.append(prefix[i].format(int(ind)))
        
    all_indices_dict[item] = code
    
print('check.')
code2item = {}
for item, code in all_indices_dict.items():
    code2item[str(code)] = item
    
print('check: ', len(code2item) == len(all_indices_dict))

with open(output_file, 'w') as fp:
    json.dump(all_indices_dict, fp, indent=4)
