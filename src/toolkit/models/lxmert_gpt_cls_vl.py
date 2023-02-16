from transformers import GPT2Model, LxmertModel, LxmertTokenizer, EncoderDecoderModel, AutoModelForCausalLM
import torch
from torch import nn
from toolkit.utils import AverageMeter, clip_gradients, decode_caption, WORD_MAP_FILENAME, rm_caption_special_tokens
import logging
from pycocoevalcap.bleu.bleu import Bleu
import os.path
import json
from collections import OrderedDict
import torch.nn.functional as F

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LxMertGPTCLSVLModel(nn.Module):
    def __init__(self, args, text_tokenizer):
        super(LxMertGPTCLSVLModel, self).__init__()
        
        if  args.dec_not_pretrained:
            print("LxMertGPTTruthModel dec not pretrained ")
            enc_dec_pretrained = EncoderDecoderModel.from_encoder_decoder_pretrained("unc-nlp/lxmert-base-uncased", args.gpt2_type)
            config_decoder = enc_dec_pretrained.decoder.config
            config_decoder.n_layer=args.n_layer
            self.enc_dec = EncoderDecoderModel(encoder=LxmertModel.from_pretrained("unc-nlp/lxmert-base-uncased"), decoder=AutoModelForCausalLM.from_config(config_decoder))
        else:
            print("LxMertGPTTruthModel dec pretrained ")
            self.enc_dec = EncoderDecoderModel.from_encoder_decoder_pretrained("unc-nlp/lxmert-base-uncased", args.gpt2_type)
        self.encoder = self.enc_dec.encoder
        self.decoder = self.enc_dec.decoder
        self.enc_dec.decoder.config.use_cache = False

        self.text_tokenizer = text_tokenizer
        self.enc_text_tokenizer = LxmertTokenizer.from_pretrained('unc-nlp/lxmert-base-uncased')


        word_map_filename = os.path.join(args.dataset_splits_dir, WORD_MAP_FILENAME)
        with open(word_map_filename) as f:
            self.word_map = json.load(f)

        if args.freeze_dec:
            for name, p in self.decoder.named_parameters():
                if name in ['transformer.h.5.crossattention.bias', 'transformer.h.2.ln_cross_attn.weight', 'transformer.h.9.crossattention.c_proj.bias', 'transformer.h.5.crossattention.masked_bias', 'transformer.h.11.crossattention.c_proj.bias', 'transformer.h.2.crossattention.c_proj.weight', 'transformer.h.0.crossattention.bias', 'transformer.h.5.crossattention.q_attn.weight', 'transformer.h.5.crossattention.c_proj.bias', 'transformer.h.10.crossattention.q_attn.weight', 'transformer.h.8.crossattention.masked_bias', 'transformer.h.2.crossattention.bias', 'transformer.h.8.crossattention.q_attn.weight', 'transformer.h.0.ln_cross_attn.weight', 'transformer.h.6.crossattention.bias', 'transformer.h.8.crossattention.bias', 'transformer.h.8.crossattention.c_proj.weight', 'transformer.h.8.ln_cross_attn.weight', 'transformer.h.10.crossattention.bias', 'transformer.h.3.crossattention.q_attn.weight', 'transformer.h.1.crossattention.c_proj.bias', 'transformer.h.9.crossattention.q_attn.weight', 'transformer.h.7.crossattention.c_attn.weight', 'transformer.h.3.crossattention.bias', 'transformer.h.7.crossattention.c_proj.bias', 'transformer.h.4.crossattention.c_attn.weight', 'transformer.h.3.ln_cross_attn.weight', 'transformer.h.7.crossattention.q_attn.weight', 'transformer.h.1.crossattention.bias', 'transformer.h.7.ln_cross_attn.weight', 'transformer.h.9.crossattention.c_attn.weight', 'transformer.h.3.crossattention.c_proj.bias', 'transformer.h.8.crossattention.c_proj.bias', 'transformer.h.11.crossattention.c_proj.weight', 'transformer.h.3.crossattention.c_proj.weight', 'transformer.h.11.crossattention.bias', 'transformer.h.0.crossattention.c_proj.weight', 'transformer.h.2.crossattention.q_attn.weight', 'transformer.h.6.ln_cross_attn.weight', 'transformer.h.1.crossattention.masked_bias', 'transformer.h.4.crossattention.q_attn.weight', 'transformer.h.4.ln_cross_attn.weight', 'transformer.h.11.crossattention.q_attn.weight', 'transformer.h.11.crossattention.c_attn.weight', 'transformer.h.7.crossattention.bias', 'transformer.h.6.crossattention.masked_bias', 'transformer.h.2.crossattention.c_proj.bias', 'transformer.h.9.ln_cross_attn.weight', 'transformer.h.10.crossattention.c_attn.weight', 'transformer.h.1.crossattention.c_attn.weight', 'transformer.h.1.ln_cross_attn.weight', 'transformer.h.6.crossattention.c_proj.weight', 'transformer.h.7.crossattention.c_proj.weight', 'transformer.h.8.crossattention.c_attn.weight', 'transformer.h.6.crossattention.c_proj.bias', 'transformer.h.0.crossattention.masked_bias', 'transformer.h.5.ln_cross_attn.weight', 'transformer.h.10.crossattention.masked_bias', 'transformer.h.11.ln_cross_attn.weight', 'transformer.h.2.crossattention.c_attn.weight', 'transformer.h.1.crossattention.c_proj.weight', 'transformer.h.5.crossattention.c_proj.weight', 'transformer.h.10.ln_cross_attn.weight', 'transformer.h.0.crossattention.q_attn.weight', 'transformer.h.11.crossattention.masked_bias', 'transformer.h.2.crossattention.masked_bias', 'transformer.h.3.crossattention.masked_bias', 'transformer.h.1.crossattention.q_attn.weight', 'transformer.h.4.crossattention.c_proj.weight', 'transformer.h.10.crossattention.c_proj.weight', 'transformer.h.0.crossattention.c_proj.bias', 'transformer.h.7.crossattention.masked_bias', 'transformer.h.10.crossattention.c_proj.bias', 'transformer.h.9.crossattention.masked_bias', 'transformer.h.4.crossattention.masked_bias', 'transformer.h.6.crossattention.c_attn.weight', 'transformer.h.4.crossattention.c_proj.bias', 'transformer.h.4.crossattention.bias', 'transformer.h.6.crossattention.q_attn.weight', 'transformer.h.0.crossattention.c_attn.weight', 'transformer.h.9.crossattention.c_proj.weight', 'transformer.h.3.crossattention.c_attn.weight', 'transformer.h.9.crossattention.bias', 'transformer.h.5.crossattention.c_attn.weight']:
                    print("requires grad", name)
                    p.requires_grad = True
                else:
                    p.requires_grad = False

        if args.enc_finetune:
            print("fine tuning")
            for p in self.encoder.parameters():
                p.requires_grad = True
        else:
            for p in self.encoder.parameters():
                p.requires_grad = False

        self.args=args

    def encoder_forward(self, text_input_ids, text_attention_mask, visual_inputs_ids, bounding_boxes):
        encoder_outputs=self.encoder(
            input_ids=text_input_ids,
            visual_feats=visual_inputs_ids,
            visual_pos=bounding_boxes,
            attention_mask=text_attention_mask,
            output_hidden_states=True,
            output_attentions=True
        )

        order=OrderedDict()
        order.last_hidden_state=torch.cat((encoder_outputs.vision_output, encoder_outputs.language_output), dim=1) # same as vision_output

        order[0]=order.last_hidden_state
        order.hidden_states=encoder_outputs.vision_hidden_states #does not matter since it just for hugginface to output -> not used inside decoder 
        order.attentions=encoder_outputs.cross_encoder_attentions #does not matter since it just for hugginface to output-> not used inside decoder 
        encoder_outputs=order
        
        return encoder_outputs

    def forward(self, batch_data, data_loader):
        #inputs
        text_input_ids, text_attention_mask, visual_inputs_ids, bounding_boxes, dec_inputs_ids, dec_attention_mask, caption_lengths = batch_data
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)            
        visual_inputs_ids= visual_inputs_ids.to(device)
        bounding_boxes= bounding_boxes.to(device)
        visual_attention_masks= torch.ones(visual_inputs_ids.size()[0], visual_inputs_ids.size()[1]).to(device)
        dec_inputs_ids= dec_inputs_ids.to(device)
        dec_attention_mask= dec_attention_mask.to(device)
        decode_lengths = caption_lengths.squeeze(1) - 1

        if self.args.image_blind:
            visual_inputs_ids= torch.zeros(visual_inputs_ids.shape).to(device)
        
        encoder_outputs = self.encoder_forward(text_input_ids, text_attention_mask, visual_inputs_ids, bounding_boxes)
        enc_attention_masks=torch.cat((visual_attention_masks, text_attention_mask), dim=1)   
        #the decoder labels are equal to the inputs since the hugginface sifts it inside        
        dec_labels = dec_inputs_ids.clone()
        #but it's important to replace the pad by -100 for the model to ignore it in the loss
        dec_labels[dec_labels==0]= -100      

        # no need encoder_attention_mask=enc_att since Lxmert gives all same inputs sizes
        loss = self.enc_dec(
            encoder_outputs = encoder_outputs,
            encoder_attention_mask=enc_attention_masks,
            decoder_input_ids=dec_inputs_ids, 
            decoder_attention_mask=dec_attention_mask, 
            labels=dec_labels
        ).loss        

        return loss, decode_lengths


    @staticmethod
    def add_args(parser):
        """Add model-specific arguments to the parser"""
        group = parser.add_argument_group("EXTRA")
        group.add_argument("--enc_finetune", action="store_true")
        group.add_argument("--dec_not_pretrained", action="store_true")
        group.add_argument("--image_blind", action="store_true")
        group.add_argument("--gpt2_type", type=str, default="gpt2", choices=["gpt2", "gpt2-medium","gpt2-large"])
        group.add_argument("--decoder-learning-rate", type=float, default=1e-4)
        group.add_argument("--encoder-learning-rate", default=1e-4)
        group.add_argument("--freeze_dec", action="store_true")
        group.add_argument("--n_layer", type=int, default=12)
        group.add_argument("--language_inputs", type=str, default=None, choices=[
            None, "clip_resnet_4_with_ground_truth", "fasterrcnn_l2_five", "clip_resnet_three","clip_resnet_five", "model_caps_five", "clip_resnet_five_external_data","clip_resnet_five_mix_current_and_external_data", "clip_resnet_pairs","clip_vit_three", "clip_resnet", "clip_vit_five", "clip_vit_five_img2img", "clip_vit","cls_end", "tags", "ground_truth", "ground_truth_5", "any"
        ])
        return group

    def forward_inference(self, batch_data, data_loader, max_caption_len):
        #inputs
        text_input_ids,text_attention_mask, visual_inputs_ids, bounding_boxes, all_captions_for_image, caption_lengths, coco_id = batch_data
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)   
        visual_inputs_ids = visual_inputs_ids.to(device)
        visual_attention_masks= torch.ones(visual_inputs_ids.size()[0], visual_inputs_ids.size()[1]).to(device)
        bounding_boxes = bounding_boxes.to(device)
        batch_size=visual_inputs_ids.size(0)
        decode_lengths = torch.full((batch_size,), max_caption_len, dtype=torch.int64, device=device)

        #greedy decoding during validation
        scores = torch.zeros((batch_size, max(decode_lengths), self.text_tokenizer.vocab_size), device=device)
        dec_input_ids = torch.ones((batch_size, 1),dtype=torch.long, device=device) * self.text_tokenizer.bos_token_id
 
        enc_out = self.encoder_forward(text_input_ids, text_attention_mask, visual_inputs_ids, bounding_boxes)
        enc_attention_masks=torch.cat((visual_attention_masks, text_attention_mask), dim=1) 

        for t in range(max(decode_lengths)):
           
            # Find all sequences where an <end> token has been produced in the last timestep

            ind_end_token = torch.nonzero(dec_input_ids == self.text_tokenizer.eos_token_id)
            #ind_end_token givesfor instance [[0,3], [1,3]]

            decode_lengths[ind_end_token[:,0]]=ind_end_token[:,1]

            # Check if all sequences are finished:
            incomplete_sequences_ixs = torch.nonzero(decode_lengths > t).view(-1)
            if len(incomplete_sequences_ixs) == 0:
                break

            # Forward prop.
            outputs =self.enc_dec(
                encoder_outputs=enc_out,
                encoder_attention_mask=enc_attention_masks,
                decoder_input_ids=dec_input_ids,
            )

            lm_logits = outputs.logits
            next_decoder_input_ids = torch.argmax(lm_logits[:, -1:], axis=-1)
            dec_input_ids = torch.cat([dec_input_ids, next_decoder_input_ids], axis=-1)
    
            scores[incomplete_sequences_ixs, t, :] = lm_logits[incomplete_sequences_ixs, t].squeeze(1)

        return dec_input_ids, batch_size, caption_lengths, all_captions_for_image, coco_id

    def forward_critical_sample(self, batch_data, data_loader, max_caption_len, greedy=False):
        # inputs
        text_input_ids,text_attention_mask, visual_inputs_ids, bounding_boxes, all_captions_for_image, caption_lengths, coco_id = batch_data
        text_input_ids = text_input_ids.to(device)
        text_attention_mask = text_attention_mask.to(device)   
        visual_inputs_ids = visual_inputs_ids.to(device)
        visual_attention_masks= torch.ones(visual_inputs_ids.size()[0], visual_inputs_ids.size()[1]).to(device)
        bounding_boxes = bounding_boxes.to(device)                      
        batch_size=visual_inputs_ids.size(0)            
        decode_lengths = torch.full((batch_size,), max_caption_len, dtype=torch.int64, device=device)
            
        #sampling
        samples_scores = torch.zeros((batch_size, max(decode_lengths)), device=device)
        dec_input_ids = torch.ones((batch_size, 1),dtype=torch.long, device=device) * self.text_tokenizer.bos_token_id
        dec_input_ids_padding = torch.zeros((batch_size, max(decode_lengths)),dtype=torch.long, device=device)

        enc_out = self.encoder_forward(text_input_ids, text_attention_mask, visual_inputs_ids, bounding_boxes)

        for t in range(max(decode_lengths)):
            # Find all sequences where an <end> token has been produced in the last timestep
            ind_end_token = torch.nonzero(dec_input_ids_padding == self.text_tokenizer.eos_token_id)
            #ind_end_token givesfor instance [[0,3], [1,3]]
            decode_lengths[ind_end_token[:,0]]=ind_end_token[:,1]
            # Check if all sequences are finished:
            incomplete_sequences_ixs = torch.nonzero(decode_lengths > t).view(-1)

            if len(incomplete_sequences_ixs) == 0:
                break

            # Forward prop.
            outputs =self.enc_dec(
                encoder_outputs=enc_out,
                decoder_input_ids=dec_input_ids,
            )

            lm_logits = outputs.logits[:, -1:].squeeze(1) #lm_logits[:, -1:] -1: for the last time step            
            
            if not greedy:

                log_probs= F.log_softmax(lm_logits, dim=-1)
                #sample
                #next_decoder_input_ids = torch.distributions.Categorical(logits=lm_logits.detach()).sample().unsqueeze(1)
                #sampleLogprobs = F.log_softmax(lm_logits, dim=-1).gather(1, next_decoder_input_ids) # gather the logprobs at sampled positions
                next_decoder_input_ids = torch.distributions.Categorical(logits=log_probs.detach()).sample().unsqueeze(1)
                sampleLogprobs = log_probs.gather(1, next_decoder_input_ids) # gather the logprobs at sampled positions
                samples_scores[incomplete_sequences_ixs, t] = sampleLogprobs[incomplete_sequences_ixs, :].squeeze(1)

            else:
                next_decoder_input_ids = lm_logits.argmax(dim=-1).unsqueeze(1)
            
            dec_input_ids = torch.cat([dec_input_ids, next_decoder_input_ids], axis=-1)
            dec_input_ids_padding[incomplete_sequences_ixs, t] = next_decoder_input_ids[incomplete_sequences_ixs, :].squeeze(1)

        return dec_input_ids_padding, decode_lengths, samples_scores, batch_size, all_captions_for_image, coco_id
