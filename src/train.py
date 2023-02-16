"""Training script for the implemented image captioning models"""
import os
import sys
import logging
import numpy as np
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
#import wandb
import transformers
from critical_train import CriticalTraining
from transformers import CLIPProcessor, GPT2Tokenizer, LxmertTokenizer
import json
import torch
import torch.nn as nn
import torch.optim
import torch.utils.data
import torch.backends.cudnn as cudnn
from torch.nn.utils.rnn import pack_padded_sequence

from options import check_args
from toolkit.models.lxmert_gpt_cls_vl import LxMertGPTCLSVLModel
from toolkit.data.datasets import get_data_loader
from toolkit.optim import create_optimizer, create_adamw_optimizer, create_scheduler, backprop, create_criterion
from toolkit.utils import (
    AverageMeter,
    clip_gradients,
    backprop,
    decode_caption,
    save_checkpoint,
    get_log_file_path,
    rm_caption_special_tokens,
    MODEL_LXMERT_GPT_CLS_VL, 
    #MODEL_PREDICTION_CAPS,
    TOKEN_PAD
)

from datetime import datetime 


abbr2name = {
    "lxmert_gpt_cls_vl": MODEL_LXMERT_GPT_CLS_VL
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
cudnn.benchmark = True  # improve performance if inputs to model are fixed size
all_training_losses = []
all_validation_bleu = []

# ==================================================================================================================== #
#                                                        HELPERS                                                       #
# ==================================================================================================================== #
def build_model(args, model_name, text_tokenizer=None):
    if model_name == MODEL_LXMERT_GPT_CLS_VL: 
        model = LxMertGPTCLSVLModel(args, text_tokenizer)
    return model


def build_optimizers(args, model):
    encoder_optimizer = None
    
    if abbr2name[args.model] in {
        MODEL_LXMERT_GPT_CLS_VL
    } and args.enc_finetune:
        print("creating enc optimizer")
        if args.optimizer_type == "adam":
            encoder_optimizer = create_optimizer(model.encoder, args.encoder_learning_rate)
        elif args.optimizer_type =="radam":
            encoder_optimizer = create_radam_optimizer(model.encoder, args.encoder_learning_rate)
        else:
            encoder_optimizer = create_adamw_optimizer(model.encoder, args.encoder_learning_rate)

    #other use features of Faster R-CNN
    if args.optimizer_type == "adam":
        decoder_optimizer = create_optimizer(model.decoder, args.decoder_learning_rate)
    elif args.optimizer_type =="radam":
        decoder_optimizer = create_radam_optimizer(model.decoder, args.decoder_learning_rate)
    else:
        decoder_optimizer = create_adamw_optimizer(model.decoder, args.decoder_learning_rate)
    return encoder_optimizer, decoder_optimizer


# ==================================================================================================================== #
#                                                      TRAIN & VAL                                                     #
# ==================================================================================================================== #

def train(model, data_loader,
          encoder_optimizer, decoder_optimizer,
          criterion, grad_clip,
          epoch, print_freq, debug, encoder_lr_scheduler, decoder_lr_scheduler):
    """
    Perform one training epoch.

    """

    model.train()
    losses = AverageMeter()
    epoch_training_losses = []

    # Loop over training batches
    for i, (batch_data) in enumerate(data_loader):
        
        loss, decode_lengths = model(batch_data, data_loader)
        

        backprop(decoder_optimizer, encoder_optimizer, loss, grad_clip, encoder_lr_scheduler, decoder_lr_scheduler)
        
        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths).item())

        # Log status
        if i % print_freq == 0:
            logging.info("Epoch: {0}[Batch {1}/{2}]\t"
                         "Loss: {loss.val:.4f} (Average: {loss.avg:.4f})\t".format(
                            epoch, i, len(data_loader), loss=losses))

        epoch_training_losses.append(loss.item())

        if debug:
            break
    
    logging.info("\n * LOSS - {loss.avg:.3f}".format(loss=losses))

    return epoch_training_losses


def train_critical(self, critical_training, model, data_loader, encoder_optimizer, decoder_optimizer, max_caption_len, print_freq, epoch, debug, grad_clip, encoder_lr_scheduler, decoder_lr_scheduler, greedy, reward_with_bleu):

    word_map = self.word_map
    model.train()
    losses = AverageMeter()

    epoch_training_rewards = []

    for i, batch_data in enumerate(data_loader):
        model.train()

        target_captions, generated_captions = [], []

        (
            captions_tokens_ids, 
            decode_lengths, 
            samples_scores, 
            batch_size, 
            all_captions_for_image, 
            coco_id
         ) = model.forward_critical_sample(batch_data, data_loader, max_caption_len)
        
        # Generated captions
        captions_tokens =[]
        for caps_index in range(batch_size):
            captions_tokens.append(self.text_tokenizer.decode(captions_tokens_ids[caps_index,:decode_lengths[caps_index].item()]))
        generated_captions.extend(captions_tokens)

        # Target captions
        for j in range(all_captions_for_image.shape[0]):
            #remove <end_token> and use end_token of the model
            img_captions = [decode_caption(rm_caption_special_tokens(caption, word_map), word_map)
                            for caption in all_captions_for_image[j].tolist()]
            target_captions.append(img_captions)

        assert len(target_captions) == len(generated_captions)

        id2targets = {coco_id[ix]: target_captions[ix] for ix in range(len(coco_id))}
        id2caption = {coco_id[ix]: [generated_captions[ix]] for ix in range(len(coco_id))}

        if greedy:
            model.eval()

            with torch.no_grad():
                generated_greedy_captions = []

                (
                    captions_tokens_ids, 
                    decode_lengths, 
                    _, 
                    batch_size, 
                    _, 
                    _
                ) = model.forward_critical_sample(batch_data, data_loader, max_caption_len, greedy=True)

                # Generated captions
                captions_tokens =[]
                for caps_index in range(batch_size):
                    captions_tokens.append(self.text_tokenizer.decode(captions_tokens_ids[caps_index,:decode_lengths[caps_index].item()]))
                generated_greedy_captions.extend(captions_tokens)

                # Target captions
                for j in range(all_captions_for_image.shape[0]):
                    #remove <end_token> and use end_token of the model
                    img_captions = [decode_caption(rm_caption_special_tokens(caption, word_map), word_map)
                                    for caption in all_captions_for_image[j].tolist()]
                    target_captions.append(img_captions)
        
            # Generated greedy captions
            
            id2greedy = {coco_id[ix]: [generated_greedy_captions[ix]] for ix in range(len(coco_id))}
            if reward_with_bleu:
                loss, reward_monitor = critical_training.get_loss_greedy_with_bleu(captions_tokens_ids, samples_scores, id2caption, id2greedy, id2targets)
            else:
                loss, reward_monitor = critical_training.get_loss_greedy(captions_tokens_ids, samples_scores, id2caption, id2greedy, id2targets)
        else:
            loss, reward_monitor = critical_training.get_loss_mean(captions_tokens_ids, samples_scores, id2caption, id2targets)
        
        backprop(decoder_optimizer, encoder_optimizer, loss, grad_clip, encoder_lr_scheduler, decoder_lr_scheduler)

        # Keep track of metrics
        losses.update(loss.item(), sum(decode_lengths).item())

        # Log status
        if i % print_freq == 0:
            logging.info("Epoch: {0}[Batch {1}/{2}]\t"
                        "Loss: {loss.val:.4f} (Average: {loss.avg:.4f})\t".format(
                            epoch, i, len(data_loader), loss=losses))

            logging.info("reward baseline - {:.3f}".format(reward_monitor))

        epoch_training_rewards.append(reward_monitor.item())

        if debug:
            break
            
    logging.info("\n * LOSS - {loss.avg:.3f}".format(loss=losses))
    return epoch_training_rewards



def validate(model, data_loader, max_caption_len, print_freq, debug, critical=False):
    """
    Perform validation of one training epoch.

    """
    word_map = model.word_map
    model.eval()
    target_captions, generated_captions, coco_ids = [], [], []

    bleu4 = Bleu(n=4)
    cider_score= Cider()

    # Loop over batches
    for i, (batch_data) in enumerate(data_loader):

        captions_tokens_ids, batch_size, caption_lengths, all_captions_for_image, coco_id  = model.forward_inference(batch_data, data_loader, max_caption_len)

        # Generated captions
        captions_tokens =[]            
        for caps_index in range(batch_size):
            #remove start token and go until the lenght of one (first) references captions
            captions_tokens.append(model.text_tokenizer.decode(captions_tokens_ids[caps_index,1:caption_lengths[caps_index][0]]))

        generated_captions.extend(captions_tokens)

        # Target captions
        for j in range(all_captions_for_image.shape[0]):
            img_captions = [decode_caption(rm_caption_special_tokens(caption, word_map), word_map)
                            for caption in all_captions_for_image[j].tolist()]
            target_captions.append(img_captions)

        for coco in coco_id:
            coco_ids.append(coco)

        assert len(target_captions) == len(generated_captions)
        if i % print_freq == 0:
            logging.info("Validation: [Batch {0}/{1}]\t".format(i, len(data_loader)))
        if debug:
            break

    id2targets = {coco_ids[ix]: target_captions[ix] for ix in range(len(coco_ids))}
    id2caption = {coco_ids[ix]: [generated_captions[ix]] for ix in range(len(coco_ids))}
    #bleus give overall [bleu1,bleu2,bleu3,bleu4]; the _ gives individual bleus for each cap
    if critical:
        cider, _ = cider_score.compute_score(id2targets, id2caption)
        logging.info("\n * CIDER - {cider}".format(cider=cider))
        return cider
        
    else:
        bleus, _ = bleu4.compute_score(id2targets, id2caption)
        bleu = bleus[-1]
        logging.info("\n * BLEU-4 - {bleu}".format(bleu=bleu))
        return bleu

def setup_gpt2():
    def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
        outputs = [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
        return outputs

    GPT2Tokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens
    text_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    text_tokenizer.eos_token = '.' #gpt has not end of token, so we put dot (id=13)
    text_tokenizer.pad_token = '!' #gpt has not pad of token, so we chose token_id =0 (=!)
    return text_tokenizer

# ==================================================================================================================== #
#                                                         MAIN                                                         #
# ==================================================================================================================== #
def main(args):
    #to be deterministic and reproduce
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Setup training stats
    start_epoch = 0
    epochs_since_last_improvement = 0
    best_gen_metric_score = 0.0
    best_gen_loss_score = float('inf')

    # Data loaders
    model_name = abbr2name[args.model]
    val_batch_size=5


    if model_name == MODEL_LXMERT_GPT_CLS_VL:
        text_tokenizer = setup_gpt2()

        enc_tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')

    else:
        feature_extractor = None
        text_tokenizer = None
        enc_tokenizer = None


    train_data_loader = get_data_loader("train", args.batch_size, args.dataset_splits_dir, args.image_features_filename,
                                        args.workers, text_tokenizer, model_name, enc_tokenizer=enc_tokenizer, args=args)
    
    val_data_loader = get_data_loader("val", val_batch_size, args.dataset_splits_dir, args.image_features_filename,
                                      args.workers, text_tokenizer=text_tokenizer, model_name=model_name,enc_tokenizer=enc_tokenizer, args=args)
    
    if args.critical or args.critical_greedy:
        critical_data_loader = get_data_loader("critical", args.batch_size, args.dataset_splits_dir, args.image_features_filename,
                                      args.workers, text_tokenizer=text_tokenizer, model_name=model_name,enc_tokenizer=enc_tokenizer, args=args)
    

    # Build model
    ckpt_filename = os.path.join(args.checkpoints_dir, "checkpoint.best.pth.tar")
    
    if os.path.isfile(ckpt_filename):
        # Load checkpoint and update training stats
        checkpoint = torch.load(ckpt_filename, map_location=device)

        start_epoch = checkpoint["epoch"] + 1
        epochs_since_last_improvement = checkpoint["epochs_since_improvement"]
        best_gen_metric_score = checkpoint["gen_metric_score"]

        model = checkpoint["model"]

        model_name = checkpoint["model_name"]
        encoder_optimizer = checkpoint["encoder_optimizer"]
        decoder_optimizer = checkpoint["decoder_optimizer"]
       
        if args.critical or args.critical_greedy:
            start_epoch = 0
            epochs_since_last_improvement = 0
            encoder_optimizer, decoder_optimizer = build_optimizers(args, model)

            print("encoder_optimizer",encoder_optimizer)
            print("model.encoder",model.encoder)

            if args.restart_optim ==False:
                print("new optims")
                encoder_optimizer, decoder_optimizer = build_optimizers(args, model)
            else:
                print("restart optim")

            if args.enc_finetune:
                for p in model.encoder.parameters():
                    p.requires_grad = True
            else:
                for p in model.encoder.parameters():
                    p.requires_grad = False

        if "encoder_training" in args and args.encoder_training != "freeze" and encoder_optimizer is None:
            if args.encoder_training == "finetune":
                model.encoder.finetune()
            elif args.encoder_training == "train":
                model.encoder.unfreeze()
            encoder_optimizer, _ = build_optimizers(args, model)
        if args.lr_scheduler:
            encoder_lr_scheduler = create_scheduler(encoder_optimizer, len(train_data_loader))
            decoder_lr_scheduler=None
            #decoder_lr_scheduler = create_scheduler(decoder_optimizer, len(train_data_loader))
        else:
            encoder_lr_scheduler=None
            decoder_lr_scheduler=None

    else:
        # No checkpoint given, initialize the model
        model_name = abbr2name[args.model]
        model = build_model(args, model_name, text_tokenizer)
        encoder_optimizer, decoder_optimizer = build_optimizers(args, model)
        if args.lr_scheduler:
            print("enc optimizer", encoder_optimizer)
            print("enc optimizer", decoder_optimizer)
            encoder_lr_scheduler = create_scheduler(encoder_optimizer, len(train_data_loader))
            decoder_lr_scheduler=None
            #decoder_lr_scheduler = create_scheduler(decoder_optimizer, len(train_data_loader))
        else:
            encoder_lr_scheduler=None
            decoder_lr_scheduler=None

    
    # Move to GPU, if available
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model = model.to(device)

    # Log configuration
    logging.info("Model params: %s", vars(model))


    criterion = create_criterion(args.criterion, device)
    #reg_param, reg_func = create_regularizer(args)

    # Start Training
    final_epoch = 0

    if args.critical or args.critical_greedy:
        logging.info("Starting critical training on device: %s", device)
        
        critical_training= CriticalTraining(device)

        for epoch in range(start_epoch, args.max_epochs):
            training_losses = train_critical(model, critical_training, model, critical_data_loader, encoder_optimizer, decoder_optimizer, args.max_caption_len, args.print_freq, epoch, args.debug, args.grad_clip, encoder_lr_scheduler, decoder_lr_scheduler, args.critical_greedy, args.reward_with_bleu)
            all_training_losses.extend(training_losses)
            gen_loss_score=torch.tensor(training_losses).mean()

            # Validate
            gen_metric_score = validate(model, val_data_loader, args.max_caption_len, args.print_freq, args.debug, True)

            #gen_metric_score = model.validate(val_data_loader, args.max_caption_len, args.print_freq, args.debug)
            all_validation_bleu.append(gen_metric_score)

            # Update stats
            ckpt_is_best = gen_metric_score > best_gen_metric_score
            if ckpt_is_best:
                best_gen_metric_score = gen_metric_score
                epochs_since_last_improvement = 0
                logging.info("Improving\n")
            else:
                epochs_since_last_improvement += 1
                logging.info("Epochs since last improvement: {}".format(epochs_since_last_improvement))
                logging.info("Best generation score: {}\n".format(best_gen_metric_score))

            # ckpt_is_best = gen_loss_score <= best_gen_loss_score
            # if ckpt_is_best:
            #     best_gen_loss_score = gen_loss_score
            #     epochs_since_last_improvement = 0
            #     logging.info("Critical Improving\n")
            # else:
            #     epochs_since_last_improvement += 1
            #     logging.info("Critical epochs since last improvement: {}".format(epochs_since_last_improvement))
            #     logging.info("Critical Best generation score: {}\n".format(best_gen_loss_score))

            extras = dict()
            if args.checkpoints_critical_dir:
                save_checkpoint(args.checkpoints_dir, model_name, model, epoch, epochs_since_last_improvement, encoder_optimizer, decoder_optimizer, gen_metric_score, ckpt_is_best, args.checkpoints_critical_dir, **extras)
            else:
                print("error you need arg checkpoint", stop)
            final_epoch = epoch

    else: 
        start_time = datetime.now() 
        logging.info("Starting training on device: %s", device)
        for epoch in range(start_epoch, args.max_epochs):
            if epochs_since_last_improvement >= args.epochs_early_stopping:
                logging.info("No improvement since {} epochs, stopping training".format(epochs_since_last_improvement))
                break

            # Train for one epoch
            #if args.objective == OBJECTIVE_GENERATION:
            training_losses = train(model, train_data_loader, encoder_optimizer, decoder_optimizer,
                    criterion, args.grad_clip,
                    epoch, args.print_freq,
                    args.debug, encoder_lr_scheduler, decoder_lr_scheduler)


            all_training_losses.extend(training_losses)
            extras = dict()
            
            # Validate
            gen_metric_score = validate(model, val_data_loader, args.max_caption_len, args.print_freq, args.debug)

            #gen_metric_score = model.validate(val_data_loader, args.max_caption_len, args.print_freq, args.debug)
            all_validation_bleu.append(gen_metric_score)

            # Update stats
            ckpt_is_best = gen_metric_score > best_gen_metric_score
            if ckpt_is_best:
                best_gen_metric_score = gen_metric_score
                epochs_since_last_improvement = 0
                logging.info("Improving\n")
            else:
                epochs_since_last_improvement += 1
                logging.info("Epochs since last improvement: {}".format(epochs_since_last_improvement))
                logging.info("Best generation score: {}\n".format(best_gen_metric_score))

            # Save checkpoint
            save_checkpoint(args.checkpoints_dir, model_name, model, epoch, epochs_since_last_improvement,
                            encoder_optimizer, decoder_optimizer, gen_metric_score, ckpt_is_best, **extras)
            final_epoch = epoch
        time_elapsed = datetime.now() - start_time 

        

    print("all_training_losses", all_training_losses)
    print("all_validation_bleu", all_validation_bleu)

    final_dict = {
        "train_loss": all_training_losses,
        "val_bleu": all_validation_bleu
    }
    if all_training_losses:
        print("saving progress")
        if args.critical or args.critical_greedy:
            name="progress_critical.json"
            with open(args.checkpoints_critical_dir + name, 'w+') as f:
                json.dump(final_dict, f, indent=2)
        else:
            name="progress.json"
            with open(args.checkpoints_dir + name, 'w+') as f:
                json.dump(final_dict, f, indent=2)

            time_name="time.json"
            with open(args.checkpoints_dir + time_name, 'w+') as f:
                json.dump({"time": str(time_elapsed)}, f, indent=2)

            n_params= sum(p.numel() for p in model.parameters() if p.requires_grad)
            params_name="params.json"
            with open(args.checkpoints_dir + params_name, 'w+') as f:
                json.dump({"n_params": str(n_params)}, f, indent=2)

    logging.info("\n\nFinished training.")
    print("last epoch", final_epoch)
    print('Time elapsed (hh:mm:ss.ms) {}'.format(time_elapsed))
    print('Number of params', n_params)



if __name__ == "__main__":
    args = check_args(sys.argv[1:])
    print(transformers.__file__)
    print(transformers.__version__)


    logging.basicConfig(
            format='%(levelname)s: %(message)s', level=logging.INFO)

    #'%(levelname)s: %(message)s'
    logging.info(args)
    main(args)
