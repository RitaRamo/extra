"""Metrics for the image captioning task"""

import os
import json
from tqdm import tqdm
from collections import Counter
from PIL import Image
import numpy as np

from pycocotools.coco import COCO
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.eval import COCOEvalCap

from toolkit.utils import (
    decode_caption,
    rm_caption_special_tokens,
    IMAGES_NAMES_FILENAME
)

base_dir = os.path.dirname(os.path.abspath(__file__))


def coco_metrics(generated_captions_fn, annotations_dir, split):

    with open(os.path.join(annotations_dir+'/datasets/', IMAGES_NAMES_FILENAME)) as f:
        images_names = json.load(f)

    with open(generated_captions_fn) as f:
        caps_generated = json.load(f)

    caps_generated_by_id={}
    images_ordered =[]
    ids_ordered=[]
    text_ordered=[]
    count_exception=0


    print("generated_captions_fn", generated_captions_fn)

    ann_fn = "{}/annotations/captions_{}.json".format(annotations_dir, split)

    print("ann_fn", ann_fn)
    # ann_fn['type'] = 'captions'

    with open(ann_fn, 'r') as f:
        data = json.load(f)
        data['type'] = 'captions'
    with open(annotations_dir+'/annotations/captions_'+split+'_new.json', 'w') as f:
        json.dump(data, f)

    ann_fn = annotations_dir+'/annotations/captions_'+split+'_new.json'
    print("new ann_fn", ann_fn)

    coco = COCO(ann_fn)
    cocoRes = coco.loadRes(generated_captions_fn)

    # create cocoEval object by taking coco and cocoRes
    cocoEval = COCOEvalCap(coco, cocoRes)

    # evaluate on a subset of images by setting
    # cocoEval.params['image_id'] = cocoRes.getImgIds()
    # please remove this line when evaluating the full validation set
    cocoEval.params['image_id'] = cocoRes.getImgIds()

    # evaluate results
    # SPICE will take a few minutes the first time, but speeds up due to caching
    cocoEval.evaluate()

    # print output evaluation scores
    for metric, score in cocoEval.eval.items():
        print('%s: %.3f' % (metric, 100 * score))

    each_image_score = {}
    individual_scores = [eva for eva in cocoEval.evalImgs]
    count_exception=0
    for i in range(len(individual_scores)):
        coco_id = individual_scores[i]["image_id"]
        each_image_score[coco_id] = individual_scores[i]

   
    each_image_score["avg_metrics"] = cocoEval.eval

    return each_image_score["avg_metrics"], each_image_score


def calc_bleu(generated_captions_fn, target_captions_fn):

    with open(generated_captions_fn) as f:
        generated_captions = json.load(f)
    with open(target_captions_fn) as f:
        target_captions = json.load(f)
    id2caption = {meta['image_id']: [meta['caption']] for meta in generated_captions}
    id2targets = {meta['image_id']: meta['captions'] for meta in target_captions}

    bleu4 = Bleu(n=4)
    bleu_scores, _ = bleu4.compute_score(id2targets, id2caption)
    bleu_scores = [float("%.2f" % elem) for elem in bleu_scores]
    print("BLEU scores:", bleu_scores)
    return bleu_scores