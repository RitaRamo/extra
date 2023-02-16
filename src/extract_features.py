
#from transformers import CLIPVisionModel, CLIPModel, CLIPProcessor
import json
import torch
#import clip
import sys
import argparse
import h5py
import os
from PIL import Image
import numpy as np
import clip
import requests
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
import torch.nn as nn
import open_clip

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from toolkit.utils import (
    IMAGES_NAMES_FILENAME,
    DATASET_SPLITS_FILENAME,
    TRAIN_SPLIT, 
    VALID_SPLIT,
    TEST_SPLIT
)




#device=torch.device("cpu")

class ClipHFFeaturesDataset():

    def __init__(self, dataset_splits_dir, images_dir, clip_encoder, clip_processor, split):
        super().__init__()

        with open(os.path.join(dataset_splits_dir, IMAGES_NAMES_FILENAME)) as f:
            self.images_names = json.load(f)

        with open(os.path.join(dataset_splits_dir, DATASET_SPLITS_FILENAME)) as f:
            self.split = json.load(f)     

        if split == "val":
            self.split = self.split[VALID_SPLIT]
        elif split == "train":
            self.split = self.split[TRAIN_SPLIT]
        else:
            self.split = self.split[TEST_SPLIT]
   
        self.clip_processor = clip_processor
        self.clip_encoder = clip_encoder

        self.images_dir = images_dir

        print("len images_names", len(self.images_names))

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]
        image_name=self.images_names[coco_id]
        image_filename= self.images_dir+self.images_names[coco_id]
        img_open = Image.open(image_filename).copy()
        img = np.array(img_open)
        if len(img.shape) ==2 or img.shape[-1]!=3: #convert grey or CMYK to RGB
            img_open = img_open.convert('RGB')
        inputs_features = self.clip_processor(images=img_open, return_tensors="pt")

        return coco_id, inputs_features.pixel_values[0]

    def __len__(self):
        return len(self.split)

class ClipOpenAIFeaturesDataset(ClipHFFeaturesDataset):

    def __init__(self, dataset_splits_dir, images_dir, clip_encoder, clip_processor, split):
        super().__init__(dataset_splits_dir, images_dir, clip_encoder, clip_processor, split)

    def __getitem__(self, i):
        # Convert index depending on the dataset split
        coco_id = self.split[i]
        image_name=self.images_names[coco_id]
        image_filename= self.images_dir+self.images_names[coco_id]
        img_open = Image.open(image_filename).copy()
        img = np.array(img_open)
        if len(img.shape) ==2 or img.shape[-1]!=3: #convert grey or CMYK to RGB
            img_open = img_open.convert('RGB')
        inputs_features=self.clip_processor(img_open)

        return coco_id, inputs_features

    def __len__(self):
        return len(self.split)

def main(args):
   

    data=args.dataset_splits_dir.split("/")[-2]

    if args.clip_embeddings=="clip_resnet":
        output_name="clip_resnet_50_4.hdf5"

        clip_encoder, clip_processor = clip.load("RN50x4", device=device)
        encoder_dim=640


    clip_encoder.eval()

    with torch.no_grad():
        with h5py.File(os.path.join(args.image_features_dir,output_name), 'w') as h5py_file:
            #features_h5 = h.create_dataset('features', (123 287, 50, 768))   

            for split in ["train", "val", "test"]:
                print("currently in split:", split)
                data_loader = torch.utils.data.DataLoader(
                            ClipOpenAIFeaturesDataset(args.dataset_splits_dir,args.images_dir,clip_encoder, clip_processor, split),
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True
                )

                for i, (coco_id,inputs_features) in enumerate(data_loader):
                    if i%1000==0:
                        print("i",i)
                    inputs_features=inputs_features.to(device)
                    clip_features=clip_encoder.encode_image(inputs_features)

                    # print("clip_features",clip_features.size())
                    # print(stop)
                    h5py_file.create_dataset(coco_id[0], (1, encoder_dim), data=clip_features.cpu().numpy())


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--images_dir",
                        help="folder for the images")
    parser.add_argument("--image_features_dir",
                        help="Folder where the preprocessed image will be located")
    parser.add_argument("--dataset_splits_dir",
                        help="Pickled file containing the datasets")
    parser.add_argument("--create_retrieval", default=False, action="store_true",
                        help="")
    parser.add_argument("--clip_embeddings", default="clip_resnet", choices=["clip_resnet"])

    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)






