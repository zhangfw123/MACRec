import numpy as np
import torch
import torch.utils.data as data
import os

class EmbDataset(data.Dataset):

    def __init__(self,data_path):

        self.data_path = data_path
        self.embeddings = np.load(data_path)
        self.dim = self.embeddings.shape[-1]

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetAll(data.Dataset):

    def __init__(self, args):

        self.datasets = args.datasets.split(',')
        embeddings = []
        self.dataset_count = []
        for dataset in self.datasets:
            print(dataset)
            embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
            embedding = np.load(embedding_path)
            embeddings.append(embedding)
            self.dataset_count.append(embedding.shape[0])
            
        self.embeddings = np.concatenate(embeddings)
        self.dim = self.embeddings.shape[-1]
        
        print(self.dataset_count)
        print(self.embeddings.shape[0])

    def __getitem__(self, index):
        emb = self.embeddings[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embeddings)
    
class EmbDatasetOne(data.Dataset):

    def __init__(self, args, dataset):


        print(dataset)
        embedding_path = os.path.join(args.data_root, dataset, f'{dataset}{args.embedding_file}')
        self.embedding = np.load(embedding_path)

        self.dim = self.embedding.shape[-1]
        
        self.data_count = self.embedding.shape[0]

        print(self.embedding.shape)

    def __getitem__(self, index):
        emb = self.embedding[index]
        tensor_emb=torch.FloatTensor(emb)
        return tensor_emb

    def __len__(self):
        return len(self.embedding)


class DualEmbDataset(data.Dataset):
    
    def __init__(self, text_data_path, img_data_path):
        self.text_data_path = text_data_path
        self.img_data_path = img_data_path
        
        self.text_embeddings = np.load(text_data_path)
        self._text_dim = self.text_embeddings.shape[-1]
        
        self.img_embeddings = np.load(img_data_path)
        self._img_dim = self.img_embeddings.shape[-1]
        
        assert len(self.text_embeddings) == len(self.img_embeddings), \
            f"Text and image data must have the same length. Text: {len(self.text_embeddings)}, Image: {len(self.img_embeddings)}"
        
        print(f"Loaded dual-modal dataset:")
        print(f"  Text embeddings: {self.text_embeddings.shape} (dim: {self._text_dim})")
        print(f"  Image embeddings: {self.img_embeddings.shape} (dim: {self._img_dim})")
        print(f"  Total samples: {len(self.text_embeddings)}")

    def __getitem__(self, index):
        text_emb = self.text_embeddings[index]
        img_emb = self.img_embeddings[index]
        
        text_tensor = torch.FloatTensor(text_emb)   
        img_tensor = torch.FloatTensor(img_emb)
        
        return text_tensor, img_tensor, index

    def __len__(self):
        return len(self.text_embeddings)
    
    @property
    def text_dim(self):
        return self._text_dim
    
    @property
    def img_dim(self):
        return self._img_dim