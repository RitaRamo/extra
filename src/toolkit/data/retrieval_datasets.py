
"""PyTorch dataset classes for the retrieval datasets"""

import os
import json
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from toolkit.data.datasets import CaptionDataset
import gc
import h5py


from toolkit.utils import (
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    DATASET_SPLITS_FILENAME,
    CAPTIONS_FILENAME,
    IMAGES_NAMES_FILENAME,

)

import torch

def report_gpu():
   gc.collect()
   torch.cuda.empty_cache()

# ============================================================================ #
#                           RESNET                                             #
# ============================================================================ #

class ClipResnetRetrievalDataset():
    #This is the dataset used to store the clip-resnet captions representations in Faiss index
    # See toolkit->retrievals

    def __init__(self, dataset_splits_dir, feature_extractor, clip_model):
        super().__init__()

        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
            self.split = json.load(f)
        self.split = self.split[TRAIN_SPLIT]
        self.clip_model = clip_model

        self.feature_extractor = feature_extractor
        with open(os.path.join(dataset_splits_dir, CAPTIONS_FILENAME)) as f:
            self.captions_text = json.load(f)
    
    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]
        captions = self.captions_text[coco_id] 
        inputs = self.feature_extractor.tokenize(captions, truncate="true")
        text_features = self.clip_model.encode_text(inputs)        
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
        coco_ids=[(int(coco_id)*10 + cap_id) for cap_id in range(5)]

        return text_features.detach(), np.array(coco_ids, dtype=np.int64)

    def __len__(self):
        return len(self.split)


class ClipResnetRetrievalQueryDataset(ClipResnetRetrievalDataset):
    # This is the dataset used to query the clip-resnet Faiss index 
    # i.e., fetch the nearest caps by proving the current image as a query to the index


    def __init__(self, dataset_splits_dir, feature_extractor, clip_model, split="val", images_dir="../remote-sensing-images-caption/src/data/COCO/raw_dataset/images/"):
        super().__init__(dataset_splits_dir, feature_extractor, clip_model)
        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
                self.split = json.load(f)
        if split =="val":
            self.split = self.split[VALID_SPLIT]
        elif split =="test":
            self.split = self.split[TEST_SPLIT]
        else:
            self.split = self.split[TRAIN_SPLIT]

       
        with open(os.path.join(dataset_splits_dir, IMAGES_NAMES_FILENAME)) as f:
            self.images_names = json.load(f)
        
        self.images_dir= images_dir
        #"../remote-sensing-images-caption/src/data/COCO/raw_dataset/images/"
        self.feature_extractor = feature_extractor

    
    def __getitem__(self, i):
        coco_id = self.split[i]

        image_name=self.images_names[coco_id]
        image_filename= self.images_dir+self.images_names[coco_id]

        #image = Image.open(image_filename)
        img_open = Image.open(image_filename).copy()
        img = np.array(img_open)
        if len(img.shape) ==2 or img.shape[-1]!=3: #convert grey or CMYK to RGB
            img_open = img_open.convert('RGB')
        inputs_features = self.feature_extractor(img_open).unsqueeze(0)

        image_features = self.clip_model.encode_image(inputs_features)  
        image_features /= image_features.norm(dim=-1, keepdim=True)
     
        return image_features.detach(), coco_id

    def __len__(self):
        return len(self.split)



# ============================================================================ #
#                             VIT                                              #
# ============================================================================ #


class ClipVitRetrievalDataset():
    #This is the dataset used to store the clip-vit captions representations in Faiss index
    # See toolkit->retrievals

    def __init__(self, dataset_splits_dir, feature_extractor, clip_model, model):
        super().__init__()
        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
            self.split = json.load(f)
        self.split = self.split[TRAIN_SPLIT]
        self.clip_model = clip_model
        self.model = model

        self.feature_extractor = feature_extractor
        with open(os.path.join(dataset_splits_dir, CAPTIONS_FILENAME)) as f:
            self.captions_text = json.load(f)
    
    def __getitem__(self, i):
        coco_id = self.split[i]
        captions = self.captions_text[coco_id] 
        captions = [caption for caption in captions]
        inputs = self.feature_extractor(text=captions, return_tensors="pt", padding=True)
        outputs = self.model(**inputs)        
        text_embeds=self.clip_model.text_projection(outputs.pooler_output)
        text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
        coco_ids=[(int(coco_id)*10 + cap_id) for cap_id in range(5)]

        return text_embeds.detach(), np.array(coco_ids, dtype=np.int64)

    def __len__(self):
        return len(self.split)


class ClipVitRetrievalQueryDataset(ClipVitRetrievalDataset):
    # This is the dataset used to query the clip-vit Faiss index 
    # i.e., fetch the nearest caps by proving the current image as a query to the index

    def __init__(self, dataset_splits_dir, feature_extractor, clip_model, model, split="val"):
        super().__init__(dataset_splits_dir, feature_extractor, clip_model, model)
        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
                self.split = json.load(f)
        if split =="val":
            self.split = self.split[VALID_SPLIT]
        elif split =="test":
            self.split = self.split[TEST_SPLIT]
        else:
            self.split = self.split[TRAIN_SPLIT]

       
        with open(os.path.join(dataset_splits_dir, IMAGES_NAMES_FILENAME)) as f:
            self.images_names = json.load(f)
        
        self.images_dir="../remote-sensing-images-caption/src/data/COCO/raw_dataset/images/"

    
    def __getitem__(self, i):
        coco_id = self.split[i]

        image_name=self.images_names[coco_id]
        image_filename= self.images_dir+self.images_names[coco_id]
        #image = Image.open(image_filename)
        img_open = Image.open(image_filename).copy()
        img = np.array(img_open)
        if len(img.shape) ==2 or img.shape[-1]!=3: #convert grey or CMYK to RGB
            img_open = img_open.convert('RGB')
        inputs_features = self.feature_extractor(images=img_open, return_tensors="pt")

        outputs = self.model(pixel_values=inputs_features.pixel_values)        
        image_embeds=self.clip_model.visual_projection(outputs.pooler_output)
        image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)

        return image_embeds.detach(), coco_id

    def __len__(self):
        return len(self.split)


# ============================================================================ #
#                           Faster R-CNN                                       #
# ============================================================================ #


class FasterRCNNRetrievalDataset(CaptionDataset):
    # This is the dataset used to store the faster r-cnn images representations in Faiss index
    # This is also the dataset used to query the  faster r-cnn Faiss index 
    # i.e., both to store and fetch the nearest images (and associated caps) 
    # by proving the current image as to store/query to the index


    def __init__(self, dataset_splits_dir, features_fn, split="train"):
        super().__init__(dataset_splits_dir, features_fn)
        if split=="train":
            self.split = self.split[TRAIN_SPLIT]
        elif split == "val":
            self.split = self.split[VALID_SPLIT]
        else:
            self.split = self.split[TEST_SPLIT]

    def __getitem__(self, i):
        coco_id = self.split[i]

        image = self.get_image_features(coco_id)
        return image, coco_id

    def __len__(self):
        return len(self.split)