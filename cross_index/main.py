import argparse
import random
import torch
import numpy as np
from time import time
import logging
import json
from torch.utils.data import DataLoader

from datasets import EmbDataset, DualEmbDataset
from models.rqvae import CrossRQVAE
from trainer import  CrossTrainer

def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument('--lr', type=float, default=1e-3, help='learning rate')
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--num_workers', type=int, default=4, )
    parser.add_argument('--eval_step', type=int, default=1, help='eval step')
    parser.add_argument('--learner', type=str, default="AdamW", help='optimizer')
    parser.add_argument("--data_path", type=str,
                        default="/datasets/datasets/LC-Rec_all/Instruments/Instruments.emb-llama-td.npy",
                        help="Input data path.")
    parser.add_argument("--text_data_path", type=str,
                        default="/datasets/datasets/LC-Rec_all/Instruments/Instruments.emb-llama-td.npy",
                        help="Input text data path.")
    parser.add_argument("--image_data_path", type=str,
                        default="/datasets/datasets/LC-Rec_all/Instruments/Instruments.emb-llama-td.npy",
                        help="Input image data path.")

    parser.add_argument('--weight_decay', type=float, default=1e-4, help='l2 regularization weight')
    parser.add_argument("--dropout_prob", type=float, default=0.0, help="dropout ratio")
    parser.add_argument("--bn", type=bool, default=False, help="use bn or not")
    parser.add_argument("--loss_type", type=str, default="mse", help="loss_type")
    parser.add_argument("--kmeans_init", type=bool, default=True, help="use kmeans_init or not")
    parser.add_argument("--kmeans_iters", type=int, default=100, help="max kmeans iters")
    parser.add_argument('--sk_epsilons', type=float, nargs='+', default=[0.0, 0.0, 0.0], help="sinkhorn epsilons")
    parser.add_argument("--sk_iters", type=int, default=50, help="max sinkhorn iters")
    parser.add_argument("--use_cross_rq", type=bool, default=False, help="use cross rq or not")
    parser.add_argument("--begin_cross_layer", type=int, default=4, help="begin cross layer")

    parser.add_argument("--device", type=str, default="cuda:0", help="gpu or cpu")

    parser.add_argument('--num_emb_list', type=int, nargs='+', default=[256,256,256], help='emb num of every vq')
    parser.add_argument('--e_dim', type=int, default=32, help='vq codebook embedding size')
    parser.add_argument('--quant_loss_weight', type=float, default=1.0, help='vq quantion loss weight')
    parser.add_argument('--layers', type=int, nargs='+', default=[2048,1024,512,256,128,64], help='hidden sizes of every layer')

    parser.add_argument("--ckpt_dir", type=str, default="./log", help="output directory for model")

    parser.add_argument("--text_class_info", type=str, default="", help="text class info")
    parser.add_argument("--image_class_info", type=str, default="", help="image class info")
    parser.add_argument("--text_contrast_weight", type=float, default=1.0, help="text contrast weight")
    parser.add_argument("--image_contrast_weight", type=float, default=1.0, help="image contrast weight")
    parser.add_argument("--recon_contrast_weight", type=float, default=0.001, help="recon contrast weight")

    return parser.parse_args()


def process_class_info(class_info):
    
    with open(class_info, "r") as f:
        class_info = json.load(f)
    class2item_list = {}
    class_size = []
    for item in class_info:
        class_name = class_info[item][0]
        if class_name not in class2item_list:
            class2item_list[class_name] = []
        class2item_list[class_name].append(item)
    item2item_list = {}
    for item in class_info:
        item2item_list[int(item)] = class2item_list[class_info[item][0]]
        class_size.append(len(item2item_list[int(item)]))
    avg_class_size = sum(class_size) / len(class_size)
    print(f"avg class size: {avg_class_size}")
    return item2item_list

if __name__ == '__main__':
    """fix the random seed"""
    seed = 2023
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    args = parse_args()
    print(args)
    if args.text_class_info != "":
        text_class_info = process_class_info(args.text_class_info)
    else:
        text_class_info = None
    if args.image_class_info != "":
        image_class_info = process_class_info(args.image_class_info)
    else:
        image_class_info = None
    logging.basicConfig(level=logging.DEBUG)

    """build dataset"""
    print("use cross rq", args.use_cross_rq)
    print("begin cross layer", args.begin_cross_layer)
    data = DualEmbDataset(args.text_data_path, args.image_data_path)
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
                  text_class_info=text_class_info,
                  image_class_info=image_class_info
                  )
    print(model)
    data_loader = DataLoader(data,num_workers=args.num_workers,
                             batch_size=args.batch_size, shuffle=True,
                             pin_memory=True)
    trainer = CrossTrainer(args,model)
    best_loss, best_text_collision_rate, best_image_collision_rate = trainer.fit(data_loader)

    print("Best Loss",best_loss)
    print("Best Text Collision Rate", best_text_collision_rate)
    print("Best Image Collision Rate", best_image_collision_rate) 

