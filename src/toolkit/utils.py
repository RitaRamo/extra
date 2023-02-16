"""General utility functions and variables"""

import os
import logging
import shutil
import pdb

import torch
import torch.nn as nn
import os.path

#from scipy.misc import imread, imresize
#import matplotlib.pyplot as plt
import numpy as np
from torchvision.transforms import transforms
from tqdm import tqdm


# Special tokens
TOKEN_UNKNOWN = "<unk>"
TOKEN_START = "<start>"
TOKEN_END = "<end>"
TOKEN_PAD = "<pad>"

# COCO attributes and filenames
DATA_CAPTIONS = "captions"
DATA_CAPTION_LENGTHS = "caption_lengths"
DATA_COCO_SPLIT = "coco_split"
CAPTIONS_FILENAME = "captions.json"
IMAGES_NAMES_FILENAME = "images_names.json"
TAGS = "tags.json"
NEAREST_CAPS = "retrieved_nearest_caps_first.json"
BEST_CAPS = "retrieved_nearest_caps_best_clip.json"
#CLIP_VIT_CAPS ="retrieved_nearest_vit.json"
CLIP_VIT_THRESHOLD = "retrieved_nearest_vit_threshold_median.json"
CLIP_VIT_INFO = "retrieved_nearest_vit_info.json"
FASTERRCNN_L2_CAPS ="retrieved_nearest_l2_caps.json"
CLIP_RESNET_CAPS ="retrieved_nearest_resnet_caps.json"
CLIP_VIT_CAPS = "retrieved_nearest_vit_caps.json"
CLIP_VIT_IMG2IMG= "retrieved_nearest_vit_img2img_caps.json"
CLIP_RESNET_CAPS_EXTERNAL= "retrieved_nearest_resnet_caps_external.json"
CLIP_RESNET_CAPS_CURRENT_AND_EXTERNAL = "retrieved_nearest_resnet_caps_mix_current_and_external_data.json"
MODEL_PREDICTED_CAPS = "model_predicted_caps.json"




IMAGES_FILENAME = "images.hdf5"
BU_FEATURES_FILENAME = "image_features.hdf5"

# Syntax attributes and filenames
#TOKEN_IDLE = "IDLE"
#STANFORDNLP_DIR = os.path.expanduser('~/bin/stanfordnlp_resources')
# STANFORDNLP_ANNOTATIONS_FILENAME = "stanfordnlp_annotataions.json"
# STANFORDNLP_FIELD2IDX = {'id': 0, 'word': 1, 'lemma': 2, 'upos': 3, 'xpos': 4, 
#                          'feats': 5, 'head': 6, 'deprel': 7, 'deps': 8, 
#                          'misc': 9}
# CAPTIONS_META = "captions_meta"
# TAGGED_CAPTIONS = "tagged_captions"
# CAPTIONS_META_FILENAME = "captions_meta.json"

# Dataset splits attributes
DATASET_SPLITS_FILENAME = "dataset_splits.json"
TRAIN_SPLIT = "train_images"
VALID_SPLIT = "val_images"
TEST_SPLIT = "test_images"
HELDOUT_PAIRS = "heldout_pairs"
WORD_MAP_FILENAME = "word_map.json"
ENCODED_METAS_FILENAME = "encoded_captions.json"
BOXES_TRAIN_AND_VAL= "boxes_train_and_val.h5"
WIDHTS_HEIGHTS= "boxes_train_and_val_width_height.h5"

# Models
MODEL_LXMERT_GPT_CLS_VL = "LXMERT_GPT_CLS_VL"

# ============================================================================ #
#                                     DATA    -used in syncap repo             #
# ============================================================================ #
def create_word_map(words):
    """
    Create a dictionary of word -> index.
    """
    word_map = {w: i + 1 for i, w in enumerate(words)}
    # Mapping for special characters
    word_map[TOKEN_UNKNOWN] = len(word_map) + 1
    word_map[TOKEN_START] = len(word_map) + 1
    word_map[TOKEN_END] = len(word_map) + 1
    word_map[TOKEN_PAD] = 0
    return word_map


def encode_caption(caption, word_map, max_caption_len):
    """
    Map words in caption into corresponding indices
    after adding <start>, <stop> and <pad> tokens.
    """
    return (
        [word_map[TOKEN_START]]
        + [word_map.get(word, word_map[TOKEN_UNKNOWN]) for word in caption]
        + [word_map[TOKEN_END]]
        + [word_map[TOKEN_PAD]] * (max_caption_len - len(caption))
    )


def decode_caption(encoded_caption, word_map):
    rev_word_map = {v: k for k, v in word_map.items()}
    return " ".join(rev_word_map[ind].lower() for ind in encoded_caption)


def rm_caption_special_tokens(caption, word_map):
    """Remove start, end and padding tokens from encoded caption."""
    rev_word_map = {v: k for k, v in word_map.items()}
    return [tok for tok in caption
            if not (tok in {word_map[TOKEN_START], word_map[TOKEN_END], word_map[TOKEN_PAD]}
                    or rev_word_map[tok].startswith("_"))]



# ============================================================================ #

# ============================================================================ #
def adjust_learning_rate(optimizer, shrink_factor):
    """
    Shrink the learning rate by a specified factor.

    :param optimizer: optimizer whose learning rate should be shrunk.
    :param shrink_factor: factor to multiply learning rate with.
    """

    logging.info("\nAdjusting learning rate.")
    for param_group in optimizer.param_groups:
        param_group["lr"] = param_group["lr"] * shrink_factor
    logging.info(
        "The new learning rate is {}\n".format(optimizer.param_groups[0]["lr"])
    )


def clip_gradients(optimizer, grad_clip):
    """
    Clips gradients computed during backpropagation to avoid explosion of gradients.

    :param optimizer: optimizer with the gradients to be clipped
    :param grad_clip: clip value
    """
    for group in optimizer.param_groups:
        for param in group["params"]:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def get_log_file_path(logging_dir, split="train"):
    if not os.path.exists(logging_dir):
        os.makedirs(logging_dir)
    return os.path.join(logging_dir, split + ".log")


def save_checkpoint(checkpoints_dir, model_name, model,
                    epoch, epochs_since_last_improvement,
                    encoder_optimizer, decoder_optimizer,
                    generation_metric_score, is_best, critical_checkpoint=None, **kwargs):
    """
    Save a model checkpoint.

    :param epoch: epoch number
    :param epochs_since_improvement: number of epochs since last improvement
    :param encoder: encoder model
    :param decoder: decoder model
    :param encoder_optimizer: optimizer to update the encoder's weights
    :param decoder_optimizer: optimizer to update the decoder's weights
    :param validation_metric_score: validation set score for this epoch
    :param is_best: True, if this is the best checkpoint so far (will save the model to a dedicated file)
    """
    
    state = {
        "model_name": model_name,
        "epoch": epoch,
        "epochs_since_improvement": epochs_since_last_improvement,
        "gen_metric_score": generation_metric_score,
        "model": model,
        "encoder_optimizer": encoder_optimizer,
        "decoder_optimizer": decoder_optimizer,
    }
    for k, v in kwargs.items():
        state[k] = v

    if critical_checkpoint:
        if not os.path.exists(critical_checkpoint):
            os.makedirs(critical_checkpoint)
        filename = os.path.join(critical_checkpoint, "_checkpoint.last.pth.tar")
        filename_best= os.path.join(critical_checkpoint, "_checkpoint.best.pth.tar")
    else:
        filename = os.path.join(checkpoints_dir, "checkpoint.last.pth.tar")
        filename_best= os.path.join(checkpoints_dir, "checkpoint.best.pth.tar")

    torch.save(state, filename)
    #shutil.copyfile(filename, os.path.join(checkpoints_dir, "checkpoint.last.pth.tar"))
    if is_best:
        shutil.copyfile(filename, filename_best)


class AverageMeter(object):
    """Class to keep track of most recent, average, sum, and count of a metric."""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def backprop(decoder_optimizer, encoder_optimizer, loss, grad_clip, enc_lr_scheduler, dec_lr_scheduler):
    # Backward propagation
    decoder_optimizer.zero_grad()
    if encoder_optimizer:
        encoder_optimizer.zero_grad()
    loss.backward()

    # Clip gradients
    if grad_clip:
        clip_gradients(decoder_optimizer, grad_clip)
        if encoder_optimizer:
            clip_gradients(encoder_optimizer, grad_clip)

    # Update weights
    decoder_optimizer.step()
    if encoder_optimizer:
        encoder_optimizer.step()
        if enc_lr_scheduler:
            enc_lr_scheduler.step()
    if dec_lr_scheduler:
            dec_lr_scheduler.step()



