from toolkit.data.retrieval_datasets import ClipResnetRetrievalImg2ImgDataset, ClipResnetRetrievalDataset,ClipResnetRetrievalQueryDataset, ClipVitRetrievalDataset, ClipVitRetrievalQueryDataset, FasterRCNNRetrievalDataset
from toolkit.retrievals.clip import ClipRetrieval
from toolkit.retrievals.faster_rcnn import ImageRetrieval
from transformers import CLIPTextModel, CLIPVisionModel, CLIPModel, CLIPProcessor
import json
import torch
import clip
import sys
import argparse

from toolkit.utils import (
    CLIP_RESNET_CAPS,
    CLIP_RESNET_CAPS_EXTERNAL,
    CLIP_RESNET_CAPS_CURRENT_AND_EXTERNAL,
    FASTERRCNN_L2_CAPS,
    CLIP_VIT_CAPS
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_nearest_caps(img_id,nearest_coco_ids, D, eval_data_lader):
    #With clip, we need to be careful to not have captions retrieved from the actual image
    list_of_similar_caps = []
    a=0
    for nearest_coco_id in nearest_coco_ids:
        best_retrieved_imgid= str(str(nearest_coco_id)[:-1]) # this is the id of the img
        best_retrieved_capid= int(str(nearest_coco_id)[-1]) #there are 5 reference caps, hence the need to know the corresponding index (first, second,..., fifth cap)

        if int(best_retrieved_imgid) == img_id:
            #we don't want the caption retrieved to be a caption from the actual image 
            a+=1
            continue
        else:
            nearest_cap = eval_data_lader.dataset.captions_text[best_retrieved_imgid][best_retrieved_capid]
            distance_cap=D[a]
            list_of_similar_caps.append((nearest_cap,distance_cap))
        a+=1
    return list_of_similar_caps


def main(args):

    retrieval_embeddings=args.retrieval_embeddings

    if args.create_retrieval:
        
        if retrieval_embeddings == "faster_rcnn":

            data_loader = torch.utils.data.DataLoader(
                            FasterRCNNRetrievalDataset(args.dataset_splits_dir,args.image_features_filename),
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True
                        )

            encoder_output_dim= 2048
            image_retrieval = ImageRetrieval(args.retrieval_dir, encoder_output_dim, data_loader, device, is_to_add=True, index_name="image_retrieval")
            

        elif retrieval_embeddings == "clip_vit":
            #Clip vit available on huggingface

            clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
            text_model= CLIPTextModel.from_pretrained("openai/clip-vit-base-patch32")

            clip_model.eval()
            text_model.eval()
            feature_extractor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

            data_loader = torch.utils.data.DataLoader(
                            ClipVitRetrievalDataset(args.dataset_splits_dir,feature_extractor=feature_extractor, clip_model=clip_model, model=text_model),
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True
                        )

            encoder_output_dim = 512 #faster r-cnn features
            image_retrieval = ClipRetrieval(args.retrieval_dir, encoder_output_dim, data_loader, device, is_to_add=True, index_name="clip_retrieval_vit")


        elif retrieval_embeddings == "clip_resnet":
            #Clip resnet not available on huggingface, thus we use code from the original repo

            clip_model, feature_extractor = clip.load("RN50x4", device=device)
            clip_model.eval()

            data_loader = torch.utils.data.DataLoader(
                            ClipResnetRetrievalDataset(args.dataset_splits_dir,feature_extractor=clip, clip_model=clip_model),
                            batch_size=1, shuffle=False, num_workers=1, pin_memory=True
            )

            encoder_output_dim = 640 

            image_retrieval = ClipRetrieval(args.retrieval_dir, encoder_output_dim, data_loader, device, is_to_add=True, index_name="clip_retrieval_resnet")
                            
    else: 
        #getting nearest caps from created retrieval
        img2nearest_caps={}
        img2nearest_distances={}


        if retrieval_embeddings == "faster_rcnn":

            path_for_retrieved_caps = FASTERRCNN_L2_CAPS

            data_loader = torch.utils.data.DataLoader(
                FasterRCNNRetrievalDataset(args.dataset_splits_dir,args.image_features_filename, args.split),
                batch_size=1, shuffle=False, num_workers=1, pin_memory=True
            )

            encoder_output_dim= 2048
            image_retrieval = ImageRetrieval(args.retrieval_dir, encoder_output_dim, data_loader, device, is_to_add=False, index_name="image_retrieval")
            

            for i, (encoder_output, imgs_indexes) in enumerate(data_loader):
                if i%100==0:
                    print("i and img index of ImageRetrival",i)
                
                    input_img = encoder_output.mean(dim=1)
                    D, nearest_coco_ids=image_retrieval.retrieve_nearest_for_train_query_with_D(encoder_output.mean(dim=1).numpy(),k=7)
                    
                    nearest_caps= [data_loader.dataset.captions_text[str(nearest_coco_id)][0] for nearest_coco_id in nearest_coco_ids[0]]
                    nearest_distances= [str(distance) for distance in list(D[0])]

                    img2nearest_distances[str(imgs_indexes[0])]=nearest_distances
                    img2nearest_caps[str(imgs_indexes[0])] = nearest_caps

        else:
            #CLIP models (both vit and resnet)

            if retrieval_embeddings == "clip_vit":
                path_for_retrieved_caps = CLIP_VIT_CAPS
                clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
                vision_model =CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32")
                clip_model.eval()
                vision_model.eval()
                feature_extractor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

                data_loader = torch.utils.data.DataLoader(
                        ClipVitRetrievalQueryDataset(args.dataset_splits_dir,feature_extractor=feature_extractor, clip_model=clip_model, model=vision_model, split=args.split),
                                batch_size=args.batch_size, shuffle=False, num_workers=1, pin_memory=True
                )

                encoder_output_dim = 512 #faster r-cnn features
                image_retrieval = ClipRetrieval(args.retrieval_dir, encoder_output_dim, data_loader, device, is_to_add=False, index_name="clip_retrieval_vit")

            
            elif retrieval_embeddings == "clip_resnet":
                path_for_retrieved_caps = CLIP_RESNET_CAPS

                clip_model, feature_extractor = clip.load("RN50x4", device=device)
                clip_model.eval()
                data_loader = torch.utils.data.DataLoader(
                                ClipResnetRetrievalQueryDataset(args.dataset_splits_dir,feature_extractor=feature_extractor, clip_model=clip_model, split=args.split),
                                batch_size=1, shuffle=False, num_workers=1, pin_memory=True
                )
                encoder_output_dim = 640 
                image_retrieval = ClipRetrieval(args.retrieval_dir, encoder_output_dim, data_loader, device, is_to_add=False, index_name="clip_retrieval_resnet")


            for i, (vision_embedding, img_id) in enumerate(data_loader):
                if i%100==0:
                    print("i and img index of ImageRetrival",i)
                
                D, nearest_coco_ids=image_retrieval.retrieve_nearestk_for_train_query_with_D(vision_embedding[0].numpy())

                # first batch hence [0]
                img_id = int(img_id[0])
                nearest_coco_ids = nearest_coco_ids[0]
                D = D[0]

                #all nearest caps (including the caps from the actual image)
                nearest_caps= [data_loader.dataset.captions_text[str(nearest_coco_id)[:-1]][int(str(nearest_coco_id)[-1])] for nearest_coco_id in nearest_coco_ids]
                
                list_of_similar_caps=get_nearest_caps(img_id,nearest_coco_ids, D, data_loader)
                most_sim_caps,sorted_distances=zip(*list_of_similar_caps)
                img2nearest_caps[str(img_id)] = most_sim_caps
                sorted_distances= [str(distance) for distance in sorted_distances]
                img2nearest_distances[str(img_id)] =sorted_distances


        with open(args.split+"_"+path_for_retrieved_caps, 'w+') as f:
            json.dump(img2nearest_caps, f, indent=2)

        with open(args.split+"_distances_"+ path_for_retrieved_caps, 'w+') as f:
            json.dump(img2nearest_distances, f, indent=2)


def check_args(args):
    parser = argparse.ArgumentParser()
    parser.add_argument("--retrieval_dir",
                        help="Folder where the retrieval index is located")
    parser.add_argument("--image-features-filename",
                        help="Folder where the preprocessed image data is located")
    parser.add_argument("--dataset-splits-dir",
                        help="Pickled file containing the datasets")
    parser.add_argument("--create_retrieval", default=False, action="store_true",
                        help="")
    parser.add_argument("--retrieval_embeddings", default="clip_resnet", choices=["clip_resnet", "clip_vit", "faster_rcnn", "clip_resnet_img2img"])
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--batch-size", type=int, default=100,
                        help="Batch size")
    parsed_args = parser.parse_args(args)
    return parsed_args



if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    main(args)























