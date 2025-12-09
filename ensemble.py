import json
import os
import argparse
from collections import defaultdict

from evaluate import get_topk_results, get_metrics_results

def get_sort_results(predictions, scores, targets_ids, users, k, index2id):
    
    B = len(users)
    predictions = [_.strip().replace(" ","") for _ in predictions]

    
    infos = []
    
    for b in range(B):
        batch_seqs = predictions[b * k: (b + 1) * k]
        batch_scores = scores[b * k: (b + 1) * k]
        
        batch_items = []
        for index in batch_seqs:
            if index in index2id:
                batch_items.append(index2id[index])
            else:
                batch_items.append([-1])
             
        
        target = targets_ids[b]
        user = users[b]
        
        infos.append((user, target, batch_items, batch_scores))
        
    sorted_infos = sorted(infos, key=lambda x: x[0])
    
    return sorted_infos

def get_topk_results_ensemble(text_info, image_info):
    
    assert len(text_info) == len(image_info)
    
    B = len(text_info)

    results = []
    
    for b in range(B):
        text_user, text_target, text_batch_items, text_batch_scores = text_info[b]
        image_user, image_target, image_batch_items, image_batch_scores = image_info[b]
        
        assert text_user == image_user
        if text_target != image_target:
            print(text_target, image_target)
        target_item = text_target

        item_id2score = {}
        
        for i in range(len(text_batch_items)):

            item_id = text_batch_items[i][0]
            score = text_batch_scores[i]
            
            if item_id == -1:
                score = -1000
            
            if item_id in item_id2score and item_id != -1:

                item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
            else:
                item_id2score[item_id] = score
            

            item_id = image_batch_items[i][0]
            score = image_batch_scores[i]
            
            if item_id == -1:
                score = -1000
            
            if item_id in item_id2score and item_id != -1:

                item_id2score[item_id] = (score + item_id2score[item_id]) / 2 + 1
            else:
                item_id2score[item_id] = score
                
                
        pairs = []
        for item_id, score in item_id2score.items():
            pairs.append((item_id, score))
        

        sorted_pairs = sorted(pairs, key=lambda x: x[1], reverse=True)

        one_results = []
        for sorted_pred in sorted_pairs:
            if sorted_pred[0] in target_item:
                one_results.append(1)
            else:
                one_results.append(0)

        results.append(one_results)

    return results

def main(args):
    metrics = args.metrics.split(",")

    text_save_file = os.path.join(args.output_dir, f'save_seqrec_{args.num_beams}.json')
    text_info = json.load(open(text_save_file, 'r'))

    text_outputs = text_info['all_outputs'] 
    text_scores = text_info['all_scores'] 
    text_targets = text_info['all_targets']
    text_users = text_info['all_users'] 

    topk_res = get_topk_results(text_outputs, text_scores, text_targets, args.num_beams)
    metrics_results = get_metrics_results(topk_res, metrics)

    total = len(text_targets)

    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total
        
    print(metrics_results)

    #
    image_save_file = os.path.join(args.output_dir, f'save_seqimage_{args.num_beams}.json')
    image_info = json.load(open(image_save_file, 'r'))

    image_outputs = image_info['all_outputs'] 
    image_scores = image_info['all_scores'] 
    image_targets = image_info['all_targets'] 
    image_users = image_info['all_users'] 

    topk_res = get_topk_results(image_outputs, image_scores, image_targets, args.num_beams)
    metrics_results = get_metrics_results(topk_res, metrics)

    total = len(image_targets)

    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total
        
    print(metrics_results)




    index_text = os.path.join(args.data_path, args.dataset, f'{args.dataset}{args.index_file}')
    index_image = os.path.join(args.data_path, args.dataset, f'{args.dataset}{args.image_index_file}')

    item_id2text_index = json.load(open(index_text, 'r'))
    text_index2item_id = defaultdict(list)
    for item_id, text_index in item_id2text_index.items():
        text_index = ''.join(text_index)
        text_index2item_id[text_index].append(int(item_id)) 
        
    print(len(text_index2item_id))
        
    text_targets_ids = []
    for text_target in text_targets:

        text_targets_ids.append(text_index2item_id[text_target])

    ### 
    item_id2image_index = json.load(open(index_image, 'r'))
    image_index2item_id = defaultdict(list)
    for item_id, image_index in item_id2image_index.items():
        image_index = ''.join(image_index)
        image_index2item_id[image_index].append(int(item_id)) 
        
    print(len(image_index2item_id))
        
    image_targets_ids = []
    for image_target in image_targets:
        image_targets_ids.append(image_index2item_id[image_target])
        



    text_info = get_sort_results(text_outputs, text_scores, text_targets_ids, text_users, 20, text_index2item_id)
    image_info = get_sort_results(image_outputs, image_scores, image_targets_ids, image_users, 20, image_index2item_id)

    print('text info: ', len(text_info))
    print('image info: ', len(image_info))

    topk_res = get_topk_results_ensemble(text_info, image_info)
    metrics_results = get_metrics_results(topk_res, metrics)

    total = len(image_targets)

    for m in metrics_results:
        metrics_results[m] = metrics_results[m] / total
        
    print(metrics_results)

    print(total)
    
    save_file = os.path.join(args.output_dir, f'results_ensemble_{args.num_beams}.json')
    json.dump(metrics_results, open(save_file, 'w'), indent=4)


def parse_args():
    parser = argparse.ArgumentParser(description="Index")

    parser.add_argument("--data_path", type=str, default="/userhome/dataset/LC-Rec_images", help="Input data path.")
    parser.add_argument("--dataset", type=str, default="Instruments", help="Input data path.")
    
    parser.add_argument("--output_dir", type=str, default="/userhome/projects/TIGER_image/log/encoder-decoder/Instruments/ckpt_b256_lr0.0005_wd0.01_dm0_e200_index_lemb_256_dis_seqrec,seqimage,item2image,image2item,fusionseqrec", help="Input data path.")
    
    parser.add_argument("--index_file", type=str, default=".index_lemb_256_dis.json", help="Input data path.")
    parser.add_argument("--image_index_file", type=str, default=".index_vitemb_256_dis.json", help="Input data path.")
    
    parser.add_argument("--metrics", type=str, default="hit@1,hit@5,hit@10,ndcg@5,ndcg@10", help="Input data path.")
    parser.add_argument("--num_beams", type=int, default=20, help="Input data path.")

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)



