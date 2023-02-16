"""Evaluate an image captioning model on the specified evaluation set using the specified set of evaluation metrics"""

import sys
import json
import os.path
import logging
import argparse
from tqdm import tqdm
import numpy as np
#import torch.utils.data
import torch
import torch.backends.cudnn as cudnn
from options import add_model_args
from toolkit.data.datasets import get_data_loader
from toolkit.evaluation.sequence_generator import ( 
    beam_search_transformers_gpt_vl
)
from toolkit.utils import (
    MODEL_LXMERT_GPT_CLS_VL,
)

LIST_OF_MODELS = [
    MODEL_LXMERT_GPT_CLS_VL
]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True 
torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

def evaluate(args):
    # Load model
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model_name = checkpoint["model_name"]
    model = checkpoint["model"]
    model = model.to(device)
    model.eval()
    
    word_map = model.word_map
    text_tokenizer = model.text_tokenizer

    logging.info("Model: {}".format(model_name))
    logging.info("Model params: {}".format(vars(model)))

    # DataLoader
    if model_name not in LIST_OF_MODELS:
        raise Exception("Model not allowed")

    if model_name == MODEL_LXMERT_GPT_CLS_VL:
        data_loader = get_data_loader(args.split, 1, args.dataset_splits_dir, args.image_features_filename, workers=1, model_name=model_name, enc_tokenizer=model.enc_text_tokenizer, args=args)
    

    # Lists for target captions and generated captions for each image
    generated_captions = {}
    generated_beams = {}

    layers_v={0:[],1:[],2:[], 3:[]}
    layers_t={0:[],1:[],2:[], 3:[]}
    counter_i=0

    for text_input_ids, text_attention_mask, image_features, bounding_boxes, all_captions_for_image, caption_lengths, coco_id in tqdm(
            data_loader, desc="Evaluate with beam size " + str(args.beam_size)):

        
        #print("coco id", coco_id)
        if coco_id == -1:
            continue
        coco_id = coco_id[0]


        # Generate captions
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)
        image_features = image_features.to(device)
        bounding_boxes = bounding_boxes.to(device)

        store_beam = args.store_beam 

        top_k_generated_captions, alphas, beam = beam_search_transformers_gpt_vl(
                    model, text_input_ids, text_attention_mask, image_features, bounding_boxes,args.beam_size,
                    max_caption_len=args.max_caption_len,
                    store_beam=store_beam,
                    coco=coco_id,
                    captions_text=data_loader.dataset.captions_text
        )

        generated_captions[coco_id] = top_k_generated_captions[:args.eval_beam_size]


        if store_beam:
            generated_beams[coco_id] = beam
    
        sorted_seq_tokens=[model.text_tokenizer.decode(complete_seq_id,skip_special_tokens=True) for complete_seq_id in top_k_generated_captions]  


        if args.show_attention:
            best_cap=0 #best cap correspond to the first, since it's already ordered the attentions
            sorted_att = alphas

            att_for_best_sentence=[]

            img_layers_v=[]
            img_layers_t=[]
            for dec_layer in range(model.decoder.config.n_layer):
                ordered_by_text_first=torch.cat((sorted_att[dec_layer][best_cap][:,:,:,36:],  sorted_att[dec_layer][0][:,:,:,:36]), dim=-1)
                all_ts=[]
                all_vs=[]

                for a in range(12):
                    #sum attentions over V or L -> avaraged across the time-steps
                    all_vs.append(sorted_att[dec_layer][best_cap][:,a,:,:36].sum(2).mean().item())
                    all_ts.append(sorted_att[dec_layer][best_cap][:,a,:,36:].sum(2).mean().item())

                #avarage then across the attention heads
                layers_v[dec_layer].append(np.mean(all_vs))
                layers_t[dec_layer].append(np.mean(all_ts))

            # if counter_i>5:
            #     break
            # counter_i+=1

    if args.show_attention:
        for dec_layer in range(model.decoder.config.n_layer):
            print("layer",dec_layer)
            print("final v", np.mean(layers_v[dec_layer]))
            print("final t", np.mean(layers_t[dec_layer]))


    # Save results
    name = args.split
    name += ".beam_" + str(args.beam_size)
    outputs_dir = os.path.join(args.output_path, "outputs")
    if not os.path.exists(outputs_dir):
        os.makedirs(outputs_dir)

    torch.save(layers_v, os.path.join(outputs_dir, "v_attention"))
    torch.save(layers_t, os.path.join(outputs_dir, "l_attention"))
    

    # Save results file with top caption for each image :
    # (JSON file of image -> caption output by the model)
    results = []
    for coco_id, top_k_captions in generated_captions.items():
        caption = text_tokenizer.decode(top_k_captions[0],skip_special_tokens=True)
        results.append({"image_id": int(coco_id), "caption": caption})
    
    results_output_file_name = os.path.join(outputs_dir, name + ".json")
    json.dump(results, open(results_output_file_name, "w"))

    # Save results file with all generated captions for each image:
    # JSON file of image -> top-k captions output by the model. Used for recall.
    results = []
    for coco_id, top_k_captions in generated_captions.items():
        captions = [text_tokenizer.decode(capt,skip_special_tokens=True) for capt in top_k_captions]
        results.append({"image_id": int(coco_id), "captions": captions})

    results_output_file_name = os.path.join(outputs_dir, name + ".top_%d" % args.eval_beam_size + ".json")
    json.dump(results, open(results_output_file_name, "w"))


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--image-features-filename",
                        help="Folder where the preprocessed image data is located")
    parser.add_argument("--dataset-splits-dir",
                        help="Pickled file containing the dataset splits")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to checkpoint of trained model")
    parser.add_argument("--split", default="val", choices=["val", "test"])
    parser.add_argument("--show_attention", default=False, action="store_true",
                        help="")
    parser.add_argument("--store_beam", default=False, action="store_true",
                        help="")
    parser.add_argument("--output-path",
                        help="Folder where to store outputs")
    parser.add_argument("--max-caption-len", type=int, default=20)
    parser.add_argument("--beam-size", type=int, default=5,
                        help="Size of the decoding beam")
    parser.add_argument("--eval-beam-size", type=int, default=5,
                        help="Number of sequences from the beam that should be used for evaluation")
    add_model_args(parser)

    parsed_args = parser.parse_args(args)
    return parsed_args


if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    logging.info(args)
    evaluate(args)

    
