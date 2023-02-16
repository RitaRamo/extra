"""PyTorch dataset classes for the image captioning training and testing datasets"""

import os
import h5py
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

from toolkit.utils import (
    DATA_CAPTIONS,
    DATA_CAPTION_LENGTHS,
    TRAIN_SPLIT,
    VALID_SPLIT,
    TEST_SPLIT,
    TAGS,
    CLIP_VIT_CAPS,
    CLIP_VIT_THRESHOLD,
    CLIP_VIT_INFO,
    CLIP_RESNET_CAPS,
    CLIP_RESNET_CAPS_EXTERNAL,
    CLIP_RESNET_CAPS_CURRENT_AND_EXTERNAL,
    MODEL_PREDICTED_CAPS,
    FASTERRCNN_L2_CAPS,
    CLIP_VIT_IMG2IMG, 
    ENCODED_METAS_FILENAME,
    DATASET_SPLITS_FILENAME,
    CAPTIONS_FILENAME,
    IMAGES_NAMES_FILENAME,
    BOXES_TRAIN_AND_VAL,
    WIDHTS_HEIGHTS,
    MODEL_LXMERT_GPT_CLS_VL,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class CaptionDataset(Dataset):
    """
    PyTorch Dataset that provides batches of images of a given split
    """

    def __init__(self, dataset_splits_dir, features_fn):
        """
        :param data_dir: folder where data files are stored
        :param features_fn: Filename of the image features file
        :param split: split, indices of images that should be included
        :param normalize: PyTorch normalization transformation
        :param features_scale_factor: Additional scale factor, applied before normalization
        """
        self.image_features = h5py.File(features_fn, "r")

        # Load image meta data, including captions
        with open(os.path.join(dataset_splits_dir, ENCODED_METAS_FILENAME)) as f:
            self.image_metas = json.load(f)

        #load captions with text for retrieval
        with open(os.path.join(dataset_splits_dir, CAPTIONS_FILENAME)) as f:
            self.captions_text = json.load(f)

        self.captions_per_image = len(next(iter(self.image_metas.values()))[DATA_CAPTIONS])

        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
            self.split = json.load(f)


        with open(os.path.join(dataset_splits_dir, IMAGES_NAMES_FILENAME)) as f:
            self.images_names = json.load(f)

        self.dataset_splits_dir=dataset_splits_dir

        self.img_visualize_script = {}

    def get_image_features(self, coco_id):
        image_data = self.image_features[coco_id][()]
        image = torch.FloatTensor(image_data)
        return image

    def __getitem__(self, i):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError



class CaptionTrainVLAblationDataset(CaptionDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """
    CAPTION_LEN =20

    def __init__(self, dataset_splits_dir, features_fn, tokenizer, enc_tokenizer, args=None):
        super().__init__(dataset_splits_dir, features_fn)
        
        self.split = self.split[TRAIN_SPLIT]
        self.tokenizer = tokenizer
        self.enc_tokenizer = enc_tokenizer
        self.train_boxes = h5py.File(os.path.join(dataset_splits_dir,BOXES_TRAIN_AND_VAL), "r")

        self.language_inputs = args.language_inputs

        elif self.language_inputs == "clip_resnet" or self.language_inputs =="clip_resnet_4_with_ground_truth" or self.language_inputs == "clip_resnet_three" or self.language_inputs == "clip_resnet_five":
            with open(os.path.join(dataset_splits_dir, CLIP_RESNET_CAPS)) as f:
                self.clip_resnet_caps = json.load(f)

        elif self.language_inputs == "clip_resnet_five_mix_current_and_external_data":
            with open(os.path.join(dataset_splits_dir, CLIP_RESNET_CAPS_CURRENT_AND_EXTERNAL)) as f:
                self.clip_resnet_caps = json.load(f)
            self.language_inputs = "clip_resnet_five"

        elif self.language_inputs == "clip_vit_five":
            with open(os.path.join(dataset_splits_dir, CLIP_VIT_CAPS)) as f:
                self.clip_vit_pairs = json.load(f)

        elif self.language_inputs == "fasterrcnn_l2_five":
            with open(os.path.join(dataset_splits_dir, FASTERRCNN_L2_CAPS)) as f:
                self.fasterrcnn_l2_caps = json.load(f)


    def get_language_inputs(self, coco_id):

        if self.language_inputs == "clip_resnet":
            captions = self.clip_resnet_caps[coco_id]
            input_cap = self.enc_tokenizer(captions[0], add_special_tokens=True, truncation=True, padding='max_length', max_length=50, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        elif self.language_inputs == "clip_resnet_three":
            captions = self.clip_resnet_caps[coco_id]
            input_cap = self.enc_tokenizer(captions[0]+ self.enc_tokenizer.sep_token+captions[1]+self.enc_tokenizer.sep_token+captions[2], add_special_tokens=True, truncation=True, padding='max_length', max_length=70, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        elif self.language_inputs == "clip_resnet_five":
            captions = self.clip_resnet_caps[coco_id]
            input_cap = self.enc_tokenizer(captions[0]+ self.enc_tokenizer.sep_token+captions[1]+self.enc_tokenizer.sep_token+captions[2]+self.enc_tokenizer.sep_token+captions[3]+self.enc_tokenizer.sep_token+captions[4], add_special_tokens=True, truncation=True, padding='max_length', max_length=100, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        elif self.language_inputs == "clip_resnet_4_with_ground_truth":
            captions = self.clip_resnet_caps[coco_id]
            captions_ground= self.captions_text[coco_id][np.random.randint(5)] 
            input_cap = self.enc_tokenizer(captions_ground+ self.enc_tokenizer.sep_token+captions[1]+self.enc_tokenizer.sep_token+captions[2]+self.enc_tokenizer.sep_token+captions[3]+self.enc_tokenizer.sep_token+captions[4], add_special_tokens=True, truncation=True, padding='max_length', max_length=100, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        elif self.language_inputs == "ground_truth_5":
            captions = self.captions_text[coco_id] 
            input_cap = self.enc_tokenizer(captions[0]+ self.enc_tokenizer.sep_token+captions[1]+self.enc_tokenizer.sep_token+captions[2]+self.enc_tokenizer.sep_token+captions[3]+self.enc_tokenizer.sep_token+captions[4], add_special_tokens=True, truncation=True, padding='max_length', max_length=100, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        elif self.language_inputs == "fasterrcnn_l2_five":
            captions = self.fasterrcnn_l2_caps[coco_id]
            input_cap = self.enc_tokenizer(captions[0]+ self.enc_tokenizer.sep_token+captions[1]+self.enc_tokenizer.sep_token+captions[2]+self.enc_tokenizer.sep_token+captions[3]+self.enc_tokenizer.sep_token+captions[4], add_special_tokens=True, truncation=True, padding='max_length', max_length=100, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        elif self.language_inputs == "clip_vit_five":
            captions = self.clip_vit_pairs[coco_id]
            input_cap = self.enc_tokenizer(captions[0]+ self.enc_tokenizer.sep_token+captions[1]+self.enc_tokenizer.sep_token+captions[2]+self.enc_tokenizer.sep_token+captions[3]+self.enc_tokenizer.sep_token+captions[4], add_special_tokens=True, truncation=True, padding='max_length', max_length=100, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        elif self.language_inputs == "cls_end":
            input_tokens = torch.tensor([self.enc_tokenizer.cls_token_id, self.enc_tokenizer.sep_token_id])
            attention_mask = torch.ones(len(input_tokens))

        elif self.language_inputs == "any": #random cap
            all_coco_ids = list(self.images_names.keys())
            random_coco_id= self.split[np.random.randint(len(self.split))]
            caption = self.captions_text[random_coco_id][np.random.randint(5)] #a random caption
            input_cap = self.enc_tokenizer(caption, add_special_tokens=True, truncation=True, padding='max_length', max_length=50, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]
    
        else: 
            raise Exception("Define languge input")
        return input_tokens, attention_mask

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i // self.captions_per_image]
        caption_index = i % self.captions_per_image

        image = self.get_image_features(coco_id)
        bounding_boxes = self.train_boxes[coco_id][()]

        caption = self.captions_text[coco_id][caption_index]
        target_cap = self.tokenizer(caption, add_special_tokens=True, truncation=True, padding='max_length', max_length=50, return_tensors="pt")
        caption_length = torch.LongTensor([self.image_metas[coco_id][DATA_CAPTION_LENGTHS][caption_index]])
        
        input_tokens, attention_mask= self.get_language_inputs(coco_id)
            
        return input_tokens, attention_mask, image, bounding_boxes, target_cap.input_ids[0], target_cap.attention_mask[0], caption_length

    def __len__(self):
        return len(self.split) * self.captions_per_image

class CaptionEvalVLAblationDataset(CaptionTrainVLAblationDataset):
    """
    PyTorch training dataset that provides batches of images with a corresponding caption each.
    """
    CAPTION_LEN =20

    def __init__(self, dataset_splits_dir, features_fn, tokenizer, enc_tokenizer, eval_split="val", args=None):
        super().__init__(dataset_splits_dir, features_fn, tokenizer, enc_tokenizer, args=args)

        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
            self.split = json.load(f)
        self.train_split= self.split[TRAIN_SPLIT] #for ablation of any

        if eval_split == "val":
            self.split = self.split[VALID_SPLIT]
        elif eval_split == "critical":
            self.split = self.split[TRAIN_SPLIT]
        else:
            self.split = self.split[TEST_SPLIT]

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]


        image = self.get_image_features(coco_id)
        bounding_boxes = self.train_boxes[coco_id][()]

        all_captions_for_image = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTIONS])
        caption_lengths = torch.LongTensor(self.image_metas[coco_id][DATA_CAPTION_LENGTHS])

        input_tokens, attention_mask= self.get_language_inputs(coco_id)

        if self.language_inputs == "any":
            
            all_coco_ids = list(self.images_names.keys())
            random_coco_id= self.train_split[np.random.randint(len(self.train_split))]
            caption = self.captions_text[random_coco_id][np.random.randint(5)] #a random caption
            input_cap = self.enc_tokenizer(caption, add_special_tokens=True, truncation=True, padding='max_length', max_length=50, return_tensors="pt")
            input_tokens= input_cap.input_ids[0]
            attention_mask = input_cap.attention_mask[0]

        return input_tokens, attention_mask, image, bounding_boxes, all_captions_for_image, caption_lengths, coco_id

    def __len__(self):
        return len(self.split)


def collate_fn(batch):
    #Filter the None (corrputed images) values in the collate_fn()
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)


def get_data_loader(split, batch_size, dataset_splits_dir, image_features_fn, workers, text_tokenizer=None, model_name=None, enc_tokenizer=None, args=None):

    if split == "train":

        data_loader = torch.utils.data.DataLoader(
            CaptionTrainVLAblationDataset(dataset_splits_dir, image_features_fn, text_tokenizer,enc_tokenizer,  args),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate_fn
        )


    elif split in {"val", "test", "critical"}:
       
        data_loader = torch.utils.data.DataLoader(
            CaptionEvalVLAblationDataset(dataset_splits_dir, image_features_fn, text_tokenizer, enc_tokenizer, split, args),
            batch_size=batch_size, shuffle=True, num_workers=workers, pin_memory=True, collate_fn=collate_fn
        )

    else:
        raise ValueError("Invalid data_loader split. Options: train, val, test")

    return data_loader



