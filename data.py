import copy
import random
import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import json
import numpy as np
from collections import defaultdict

class BaseDataset(Dataset):

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.dataset = args.dataset
        self.data_path = os.path.join(args.data_path, self.dataset)

        self.max_his_len = args.max_his_len
        self.his_sep = args.his_sep
        self.index_file = args.index_file
        self.image_index_file = args.image_index_file
        self.add_prompt = args.add_prompt

        self.new_tokens = None
        self.allowed_tokens = None
        self.all_items = None
        
        self.image_indices = None
        
        self.fg_image_indices = None
        self.base_tokens = set()
        tasks = self.args.tasks.split(",")
        for task in tasks:
            self.base_tokens.add(f"<{task}_1>")
            self.base_tokens.add(f"<{task}_2>")
            self.base_tokens.add(f"<{task}_3>")


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
            
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
            image_indices = json.load(f)
            
        self.image_indices = {}
        for k, v in image_indices.items():
            if random.random() > 0.5:
                self.image_indices[k] = v

    def get_new_tokens(self):
        


        if self.new_tokens is not None:
            return self.new_tokens

        self.new_tokens = set()
        
            
        for index in self.indices.values():
            for token in index:
                self.new_tokens.add(token)
        
        if self.image_indices is not None:
            for index in self.image_indices.values():
                for token in index:
                    self.new_tokens.add(token)
                    
        if self.fg_image_indices is not None:
            for index in self.fg_image_indices.values():
                for token in index:
                    self.new_tokens.add(token)
                
        self.new_tokens = sorted(list(self.new_tokens))

        return self.new_tokens
    
    def get_all_tokens(self):

        if self.new_tokens is not None:
            return self.new_tokens
        
        if self.args.tasks == 'seqrec':
            prefix_list = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>"]
            
        elif self.args.tasks == 'seqimage':
            prefix_list = ["<A_{}>","<B_{}>","<C_{}>","<D_{}>"]
            
        else:
            prefix_list = ["<a_{}>","<b_{}>","<c_{}>","<d_{}>", "<A_{}>","<B_{}>","<C_{}>","<D_{}>"]
            
        new_tokens = set()
        
        for prefix in prefix_list:
            for i in range(self.args.code_num):
                token = prefix.format(i)
                new_tokens.add(token)
                
        self.new_tokens = sorted(list(new_tokens))
        

        return self.new_tokens

    def get_all_items(self):

        if self.all_items is not None:
            return self.all_items

        self.all_items = set()
        for index in self.indices.values():
            self.all_items.add("".join(index))

        return self.all_items

    def get_prefix_allowed_tokens_fn(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = {}
            for index in self.indices.values():
                for i, token in enumerate(index):
                    token_id = tokenizer(token)["input_ids"][1]
                    if i not in self.allowed_tokens.keys():
                        self.allowed_tokens[i] = set()
                    self.allowed_tokens[i].add(token_id)
            self.allowed_tokens[len(self.allowed_tokens.keys())] = set([tokenizer.eos_token_id])
        sep = [0]


        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()
            reversed_sent = sentence[::-1]
            for i in range(len(reversed_sent)):
                if reversed_sent[i:i + len(sep)] == sep[::-1]:
                    # print(list(self.allowed_tokens[i]))
                    return list(self.allowed_tokens[i])

        return prefix_allowed_tokens_fn
    
    def get_prefix_allowed_tokens_fn_new(self, tokenizer):


        if self.allowed_tokens is None:
            self.allowed_tokens = defaultdict(set)
            first_ids = set()
            end_ids = set()
            for index in self.indices.values():
                token_id_list = [tokenizer(token)["input_ids"][1] for token in index]
                for i, token_id in enumerate(token_id_list):
                    if i == 0:
                        first_ids.add(token_id)
                    elif i == len(token_id_list) - 1:
                        end_ids.add(token_id)
                        
                    if i < len(token_id_list) - 1: 
                        self.allowed_tokens[token_id].add(token_id_list[i+1])
                        
            for ids in end_ids:
                self.allowed_tokens[ids] = first_ids

        def prefix_allowed_tokens_fn(batch_id, sentence):
            sentence = sentence.tolist()

            return list(self.allowed_tokens[sentence[-1]])

        return prefix_allowed_tokens_fn

    def _process_data(self):

        raise NotImplementedError



class SeqRecDataset(BaseDataset):
        
    def __init__(self, args, task='seqrec', mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.train_data_mode = args.train_data_mode
        self.add_prompt = args.add_prompt
        self.task = task
        self._load_data()
        self._remap_items()
        
        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        elif self.mode == 'valid':
            self.inter_data = self._process_valid_data()
        elif self.mode == 'test':
            self.inter_data = self._process_test_data()
        else:
            raise NotImplementedError



    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        if self.task == 'seqrec':
            with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
                self.indices = json.load(f)
        elif self.task == 'seqimage':
            with open(os.path.join(self.data_path, self.dataset + self.args.image_index_file), 'r') as f:
                self.indices = json.load(f)

    def _remap_items(self):

        self.remapped_inters = dict()
        for uid, items in self.inters.items():
            new_items = ["".join(self.indices[str(i)]) for i in items]
            self.remapped_inters[uid] = new_items


    def _process_train_data(self):

        inter_data = []
        
        if self.train_data_mode == 0: ## origin, for encoder-decoder
            for uid  in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]
                for i in range(1, len(items)):
                    one_data = dict()
                    # one_data["user"] = uid
                    one_data["item"] = items[i]
                    history = items[:i]
                    if self.max_his_len > 0:
                        history = history[-self.max_his_len:]

                    one_data["inters"] = history
                    inter_data.append(one_data)
                    
        elif self.train_data_mode == 1:  ## all max length
            for uid  in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]

                if len(items) <= self.max_his_len + 1:
                    one_data = dict()
                    one_data["item"] = items[-1]
                    history = items[:-1]
                    one_data["inters"] = history
                    inter_data.append(one_data)
                else:
                    for i in range(self.max_his_len + 1, len(items)):
                        one_data = dict()
                        one_data["item"] = items[i]
                        history = items[:i]
                        history = history[-self.max_his_len:]
                        one_data["inters"] = history
                        inter_data.append(one_data)
        else:
            for uid  in self.remapped_inters:
                items = self.remapped_inters[uid][:-2]
                inter_data.append(items)

        return inter_data
    
    def _process_valid_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-2]
            history = items[:-2]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]

            one_data["inters"] = history
            inter_data.append(one_data)

        return inter_data

    def _process_test_data(self):

        inter_data = []
        for uid in self.remapped_inters:
            items = self.remapped_inters[uid]
            one_data = dict()
            # one_data["user"] = uid
            one_data["item"] = items[-1]
            history = items[:-1]
            if self.max_his_len > 0:
                history = history[-self.max_his_len:]

            one_data["inters"] = history
            inter_data.append(one_data)

        if self.sample_num > 0:
            all_inter_idx = range(len(inter_data))
            sample_idx = np.random.choice(all_inter_idx, self.sample_num, replace=False)
            # print(sample_idx[:10])##################
            inter_data = np.array(inter_data)[sample_idx].tolist()

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):

        return len(self.inter_data)
    
    def _get_text_data(self, data, prompt):

        instruction = prompt["instruction"].format(**data)
        response = prompt["response"].format(**data)

        return instruction, response

    def __getitem__(self, index):
        
        if self.mode != 'train' or self.train_data_mode <= 1:
            d = self.inter_data[index]
            
        elif self.train_data_mode == 2: 
            d = dict()
            items = self.inter_data[index]
            
            if len(items) <= self.max_his_len + 1:
                d["item"] = items[-1]
                d["inters"] = items[:-1]
            else:
                end = random.randint(self.max_his_len + 1, len(items) - 1)
                start = end - self.max_his_len
                d["item"] = items[end]
                d["inters"] = items[start : end]
                
        elif self.train_data_mode == 3: 
            d = dict()
            items = self.inter_data[index]
            
            end = random.randint(1, len(items) - 1)
            d["item"] = items[end]
            history = items[:end]
            history = history[-self.max_his_len:]
            d["inters"] = history
        
        else:  
            d = dict()
            items = self.inter_data[index]
            
            start = random.randint(0, len(items) - 2)
            end = random.randint(start + 1, len(items) - 1)
            
            d["item"] = items[end]
            history = items[start:end]
            history = history[-self.max_his_len:]
            d["inters"] = history
            
        if self.add_prompt:
            if self.mode == 'train':
                prompt_id = random.randint(0, len(self.prompts) - 1)
            else:
                prompt_id = self.prompt_id
              
            prompt = self.prompts[prompt_id]

            data = {}
            data["item"] = d["item"]
            data["inters"] = self.his_sep.join(d["inters"])
            
            input, output = self._get_text_data(data, prompt)
            
        else:
            input = ''.join(d["inters"])
            output = d["item"]

        return dict(input_ids=input, labels=output, label=index, task_flag=0)

    
class ItemImageDataset(BaseDataset):

    def __init__(self, args, task="item2image", prompt_sample_num=1, sample_num=-1):
        super().__init__(args)

        self.task = task.lower()
        self.prompt_sample_num = prompt_sample_num
        self.sample_num = sample_num

        self.soft_prompt = args.soft_prompts[self.task]
        
        self.image_index_file = args.image_index_file

        self._load_data()
        self.data_pair = self._process_data()


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
            self.image_indices = json.load(f)


    def _process_data(self):

        data_pair = []
        for item_id, item_token in self.indices.items():
            item_index = "".join(self.indices[item_id])
            image_index = "".join(self.image_indices[item_id])
            
            if self.task == 'item2image':
                data_pair.append([item_index, image_index])
            else:
                data_pair.append([image_index, item_index])

        return data_pair


    def __len__(self):
        return len(self.data_pair)

    def __getitem__(self, index):

        d = self.data_pair[index]

        input = self.soft_prompt + d[0]
        output = d[1]
        return dict(input_ids=input, labels=output, task_flag=1)
    
class FusionSeqRecDataset(BaseDataset):

    def __init__(self, args, task='seqitem2image', mode="train",
                 prompt_sample_num=1, prompt_id=0, sample_num=-1):
        super().__init__(args)

        self.mode = mode
        self.prompt_sample_num = prompt_sample_num
        self.prompt_id = prompt_id
        self.sample_num = sample_num
        self.task = task
        
        self.image_index_file = args.image_index_file

        self.soft_prompt = args.soft_prompts[self.task]

        # load data
        self._load_data()

        # load data
        if self.mode == 'train':
            self.inter_data = self._process_train_data()
        else:
            raise NotImplementedError


    def _load_data(self):

        with open(os.path.join(self.data_path, self.dataset + ".inter.json"), 'r') as f:
            self.inters = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.index_file), 'r') as f:
            self.indices = json.load(f)
        with open(os.path.join(self.data_path, self.dataset + self.image_index_file), 'r') as f:
            self.image_indices = json.load(f)

    def _process_train_data(self):


        inter_data = []
        for uid in self.inters:
            items = self.inters[uid][:-2]
            for i in range(1, len(items)):
                one_data = dict()
                
                item = items[i]
                history = items[:i]
                if self.max_his_len > 0:
                    history = history[-self.max_his_len:]
                
                if self.task == 'seqitem2image':
                    history = ["".join(self.indices[str(i)]) for i in history]
                    one_data["inters"] = ''.join(history)
                    one_data["item"] = ''.join(self.image_indices[str(item)])
                    
                elif self.task == 'seqimage2item':
                    history = ["".join(self.image_indices[str(i)]) for i in history]
                    one_data["inters"] = ''.join(history)
                    one_data["item"] = ''.join(self.indices[str(item)])
                else:
                    one_data["inters"] = history
                    one_data["item"] = item
                
                inter_data.append(one_data)

        return inter_data

    def set_prompt(self, prompt_id):

        self.prompt_id = prompt_id

    def __len__(self):
        if self.mode == 'train':
            return len(self.inter_data)
        else:
            raise NotImplementedError

    def __getitem__(self, index):

        d = self.inter_data[index]

        if self.task == 'seqitem2image' or self.task == 'seqimage2item' or self.task == 'mix2mix' or self.task == 'itemimage2itemimage':
            input = self.soft_prompt + d['inters']
            
            output = d['item']
        
        else:
            
            if random.random() > 0.5:
                history = ["".join(self.indices[str(i)]) for i in d['inters']]
                input = self.soft_prompt + ''.join(history)
                output = ''.join(self.image_indices[str(d['item'])])
            else:
                history = ["".join(self.image_indices[str(i)]) for i in d['inters']]
                input = self.soft_prompt + ''.join(history)
                output = ''.join(self.indices[str(d['item'])])

        return dict(input_ids=input, labels=output, task_flag=0)
    
