import json
from collections import defaultdict
import gzip
from tqdm import tqdm
import argparse
import os

from utils import amazon18_dataset2fullname
import requests

def download_image(url, save_path):
    try:
        response = requests.get(url, stream=True)
        response.raise_for_status()
        
        with open(save_path, 'wb') as file:
            for chunk in response.iter_content(chunk_size=8192):
                file.write(chunk)
        
        # print(f"图片已成功下载到: {save_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"下载失败: {e}")
        return False
    
def is_valid_jpg(jpg_file):

    with open(jpg_file, 'rb') as f:
        file_size = os.path.getsize(jpg_file)
        
        if file_size < 2:
            return False
        
        f.seek(file_size - 2)
        return f.read() == b'\xff\xd9'
        
        
def load_meta_items(file):
    items = {}
    with gzip.open(file, "r") as fp:
        for line in tqdm(fp, desc="Load metas"):
            data = json.loads(line)
            item = data["asin"]
            
            if 'imageURLHighRes' in data:
                imageURLHighRes = data['imageURLHighRes']
            else:
                imageURLHighRes = []
                
            # if len(imageURLHighRes) == 0:
            #     print(data)
            #     break

            items[item] = {'imageURLHighRes': imageURLHighRes}
            # break
            # print(items[item])
    return items


def load_meta_data(args):
    
    print('Process data: ')
    print(' Dataset: ', args.dataset)
    
    meta_data_path = args.meta_data_path

    dataset_full_name = amazon18_dataset2fullname[args.dataset]

    # load item IDs with meta data
    meta_file_path = os.path.join(meta_data_path, f'meta_{dataset_full_name}.json.gz')
    meta_items = load_meta_items(meta_file_path)
    
    return meta_items

def load_ratings_items(args, meta_items):
    # load ratings

    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    rating_file_path = os.path.join(args.rating_data_path, dataset_full_name + '.csv')

    filter_items = {}
    
    with open(rating_file_path, 'r') as fp:
        for line in tqdm(fp, desc='Load ratings'):
            try:
                item, user, rating, time = line.strip().split(',')
                
                if item in meta_items and item not in filter_items:
                    filter_items[item] = meta_items[item]
                    
            except ValueError:
                print(line)
                
    return filter_items

def load_5_core_review(args, meta_items):
    
    dataset_full_name = amazon18_dataset2fullname[args.dataset]
    review_file_path = os.path.join(args.review_data_path, dataset_full_name + '_5.json.gz')

    filter_items = {}
    
    gin = gzip.open(review_file_path, 'rb')

    for line in tqdm(gin):

        line = json.loads(line)

        user_id = line['reviewerID']
        item_id = line['asin']
        time = line['unixReviewTime']

        if item_id in meta_items and item_id not in filter_items:
            filter_items[item_id] = meta_items[item_id]
                
    return filter_items

def load_json(file):
    with open(file, 'r') as f:
        data = json.load(f)
    return data

def main(args, meta_items):
    
    dataset = args.dataset
    save_path = f'{args.save_path}/{dataset}'

    item_images_file = f'{args.save_path}/{dataset}_images_info.json'
    os.makedirs(save_path, exist_ok=True)

    if os.path.exists(item_images_file):
        item_images = load_json(item_images_file)
    else:
        item_images = defaultdict(list)

    miss_num = 0
    
    for asin, info in tqdm(meta_items.items()):
        
        if asin in item_images and len(item_images[asin]) != 0:
            continue
        
        image_urls = info['imageURLHighRes']
        
        if len(image_urls) == 0:
            # print('no image')
            item_images[asin] = []
            miss_num += 1
        
        flag = False
        for idx, image_url in enumerate(image_urls):
            name = os.path.basename(image_url)
            
            save_file = os.path.join(save_path, name)
            if not os.path.exists(save_file):
                flag = download_image(image_url, save_file)
                    
            else:
                flag = True
                
            if flag and is_valid_jpg(save_file):
                item_images[asin].append(name)
                break
            
        
        if not flag:
            item_images[asin] = []
            
        # break

    print('miss num: ', miss_num)
    print("cover rate: ", (len(meta_items) - miss_num) / len(meta_items))
    item_images_f = open(item_images_file, 'w', encoding='utf8')
    json.dump(item_images, item_images_f, indent=4)
    item_images_f.close()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Arts', help='Instruments / Arts / Games')
    parser.add_argument('--meta_data_path', type=str, default='/datasets/datasets/amazon18/Metadata')
    parser.add_argument('--rating_data_path', type=str, default='/datasets/datasets/amazon18/Ratings')
    parser.add_argument('--review_data_path', type=str, default='/datasets/datasets/amazon18/Review')
    parser.add_argument('--save_path', type=str, default='/datasets/datasets/amazon18/Images')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()

    meta_items = load_meta_data(args)
    
    print('meta items: ', len(meta_items))
    
    # filter_items = load_ratings_items(args, meta_items)
    
    filter_items = load_5_core_review(args, meta_items)
    print('filter items: ', len(filter_items))

    for i in range(1):
        main(args, filter_items)