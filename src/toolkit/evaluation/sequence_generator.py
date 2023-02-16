import torch
import torch.nn.functional as F

from toolkit.utils import TOKEN_START, TOKEN_END, TOKEN_PAD, DATA_CAPTION_LENGTHS, rm_caption_special_tokens, decode_caption
from collections import OrderedDict
import math
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def general_beam(model,encoder_output, beam_size, max_caption_len,  store_beam, enc_attention_masks=None):
    current_beam_width = beam_size

    enc_image_size = encoder_output.last_hidden_state.size(1)

    # Tensor to store top k sequences; now they're just <start>
    top_k_sequences = torch.full((beam_size, 1), model.text_tokenizer.bos_token_id, dtype=torch.int64, device=device)
    top_k_scores = torch.zeros(beam_size, device=device)

    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []
    beam = []

    for step in range(0, max_caption_len - 1):
        prev_words = top_k_sequences[:, step]

        outputs =model.enc_dec(
            encoder_outputs=encoder_output,
            encoder_attention_mask=enc_attention_masks, #can be None
            decoder_input_ids=top_k_sequences,
        )

        predictions = outputs.logits
        predictions = predictions[:,-1,:] #just consider the preds for the last token
        scores = F.log_softmax(predictions, dim=-1)

        # Add the new scores
        scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

        # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
        # different sequences, we should only look at one branch
        if step == 0:
            scores = scores[0]

        # Find the top k of the flattened scores
        top_k_scores, top_k_words = scores.view(-1).topk(current_beam_width, 0, largest=True, sorted=True)
        #print("alternative argmax", torch.argmax(outputs.logits[:, -1:], axis=-1))

        # Convert flattened indices to actual indices of scores
        prev_seq_inds = top_k_words / model.decoder.config.vocab_size  # (k)
        next_words = top_k_words % model.decoder.config.vocab_size  # (k)

        # Add new words to sequences
        top_k_sequences = torch.cat((top_k_sequences[prev_seq_inds.long()], next_words.unsqueeze(1)), dim=1)
        #print('top_k_sequences', top_k_sequences.size(), top_k_sequences)

        if store_beam:
            beam.append(top_k_sequences)


        # Check for complete and incomplete sequences (based on the <end> token)
        incomplete_inds = torch.nonzero(next_words != model.text_tokenizer.eos_token_id).view(-1).tolist()
        complete_inds = torch.nonzero(next_words == model.text_tokenizer.eos_token_id).view(-1).tolist()
        #print('incomplete_inds', incomplete_inds, 'complete_inds', complete_inds)

        # Set aside complete sequences and reduce beam size accordingly
        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_sequences[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])
           
        # Stop if k captions have been completely generated
        current_beam_width = len(incomplete_inds)
        if current_beam_width == 0:
            break
        
        #print("current_beam_width", current_beam_width)

        # Proceed with incomplete sequences
        top_k_sequences = top_k_sequences[incomplete_inds]
        encoder_output.last_hidden_state=encoder_output.last_hidden_state[prev_seq_inds[incomplete_inds].long()]
        encoder_output[0]=encoder_output.last_hidden_state

        top_k_scores = top_k_scores[incomplete_inds]

    if len(complete_seqs) < beam_size:
        complete_seqs.extend(top_k_sequences.tolist())
        complete_seqs_scores.extend(top_k_scores)

    sorted_sequences = [sequence for _, sequence in sorted(zip(complete_seqs_scores, complete_seqs), reverse=True)]

    sorted_alphas = None #not used
    return sorted_sequences, sorted_alphas, beam

def general_beam_with_attention(model,encoder_output, beam_size, max_caption_len,  store_beam, enc_attention_masks=None, size_of_cross=None):
    current_beam_width = beam_size

    enc_image_size = encoder_output.last_hidden_state.size(1)

    # Tensor to store top k sequences; now they're just <start>
    top_k_sequences = torch.full((beam_size, 1), model.text_tokenizer.bos_token_id, dtype=torch.int64, device=device)
    top_k_scores = torch.zeros(beam_size, device=device)

    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []
    beam = []
    
    seq_cross_attention = [None]*model.decoder.config.n_layer
    for dec_layer in range(model.decoder.config.n_layer):
        seq_cross_attention[dec_layer]=torch.ones(beam_size,12,1,size_of_cross)

    all_cross_attentions = []
    
    complete_cross_attention = [None]*model.decoder.config.n_layer
    for dec_layer in range(model.decoder.config.n_layer):
        complete_cross_attention[dec_layer]=[]
    complete_seqs = []
    complete_seqs_alpha = []
    complete_seqs_scores = []


    for step in range(0, max_caption_len - 1):
        prev_words = top_k_sequences[:, step]

        outputs =model.enc_dec(
            encoder_outputs=encoder_output,
            encoder_attention_mask=enc_attention_masks, #can be None
            decoder_input_ids=top_k_sequences,
            output_attentions=True
        )

        predictions = outputs.logits
        predictions = predictions[:,-1,:] #just consider the preds for the last token
        scores = F.log_softmax(predictions, dim=-1)

        # Add the new scores
        scores = top_k_scores.unsqueeze(1).expand_as(scores) + scores

        # For the first timestep, the scores from previous decoding are all the same, so in order to create 5
        # different sequences, we should only look at one branch
        if step == 0:
            scores = scores[0]

        # Find the top k of the flattened scores
        top_k_scores, top_k_words = scores.view(-1).topk(current_beam_width, 0, largest=True, sorted=True)
        #print("alternative argmax", torch.argmax(outputs.logits[:, -1:], axis=-1))

        # Convert flattened indices to actual indices of scores
        prev_seq_inds = top_k_words / model.decoder.config.vocab_size  # (k)
        next_words = top_k_words % model.decoder.config.vocab_size  # (k)

        # Add new words to sequences
        top_k_sequences = torch.cat((top_k_sequences[prev_seq_inds.long()], next_words.unsqueeze(1)), dim=1)
        #print('top_k_sequences', top_k_sequences.size(), top_k_sequences)

        if step == 0:
            for dec_layer in range(model.decoder.config.n_layer):
                cross_att=outputs.cross_attentions[dec_layer][:,:,-1,:].unsqueeze(2)
                seq_cross_attention[dec_layer]= cross_att[prev_seq_inds.long()]

                #print("seq_cross_attention after",seq_cross_attention[dec_layer].size())
        else:
            for dec_layer in range(model.decoder.config.n_layer):
                cross_att=outputs.cross_attentions[dec_layer][:,:,-1,:].unsqueeze(2)
                seq_cross_attention[dec_layer]= torch.cat((seq_cross_attention[dec_layer][prev_seq_inds.long()],cross_att[prev_seq_inds.long()]  ), dim=2)



        if store_beam:
            beam.append(top_k_sequences)

        # Check for complete and incomplete sequences (based on the <end> token)
        incomplete_inds = torch.nonzero(next_words != model.text_tokenizer.eos_token_id).view(-1).tolist()
        complete_inds = torch.nonzero(next_words == model.text_tokenizer.eos_token_id).view(-1).tolist()
        #print('incomplete_inds', incomplete_inds, 'complete_inds', complete_inds)

        # Set aside complete sequences and reduce beam size accordingly
        if len(complete_inds) > 0:
            complete_seqs.extend(top_k_sequences[complete_inds].tolist())
            complete_seqs_scores.extend(top_k_scores[complete_inds])

            for dec_layer in range(model.decoder.config.n_layer):
                #print("complete_inds", complete_inds)
                #print("i am going after a ind", seq_cross_attention[dec_layer].size())
                #print("i am going after a ind2", seq_cross_attention[dec_layer][complete_inds].size())
                for ind in complete_inds: 
                    complete_cross_attention[dec_layer].append(seq_cross_attention[dec_layer][ind].unsqueeze(0))
                


        # Stop if k captions have been completely generated
        current_beam_width = len(incomplete_inds)
        if current_beam_width == 0:
            break
        
        #print("current_beam_width", current_beam_width)

        # Proceed with incomplete sequences
        top_k_sequences = top_k_sequences[incomplete_inds]
        encoder_output.last_hidden_state=encoder_output.last_hidden_state[prev_seq_inds[incomplete_inds].long()]
        encoder_output[0]=encoder_output.last_hidden_state

        top_k_scores = top_k_scores[incomplete_inds]

    if len(complete_seqs) < beam_size:
        complete_seqs.extend(top_k_sequences.tolist())
        complete_seqs_scores.extend(top_k_scores)

    sorted_sequences = [sequence for _, sequence in sorted(zip(complete_seqs_scores, complete_seqs), reverse=True)]

    sorted_att = [None]*model.decoder.config.n_layer
    for dec_layer in range(model.decoder.config.n_layer):
        sorted_att[dec_layer] = [alpha for _, alpha in sorted(zip(complete_seqs_scores, complete_cross_attention[dec_layer]), reverse=True)]
    
    return sorted_sequences, sorted_att, beam

def beam_search_transformers_gpt_vl(model, text_input_ids, text_attention_mask, images, bounding_boxes, beam_size, max_caption_len=20, store_beam=False, coco=None, captions_text=None, image_retrieval=None):
    """Generate and return the top k sequences using beam search."""

    # the max beam size is the dictionary size - 1, since we never select pad
    beam_size = min(beam_size, model.decoder.config.vocab_size - 1)

    images = images.expand(beam_size, images.size(1), images.size(2))
    bounding_boxes = bounding_boxes.expand(beam_size, bounding_boxes.size(1), bounding_boxes.size(2))
    text_attention_mask = text_attention_mask.expand(beam_size, text_attention_mask.size(1))
    text_input_ids = text_input_ids.expand(beam_size, text_input_ids.size(1))

    encoder_output = model.encoder_forward(text_input_ids, text_attention_mask, images, bounding_boxes)

    visual_attention_masks= torch.ones(images.size()[0], images.size()[1]).to(device)
    enc_attention_masks=torch.cat((visual_attention_masks, text_attention_mask), dim=1) 
    #sorted_sequences, sorted_alphas, beam = general_beam(model,encoder_output, beam_size, max_caption_len, store_beam, print_beam, enc_attention_masks)
    
    size_of_cross_attention=bounding_boxes.size()[1] + text_input_ids.size()[1]
    sorted_sequences, sorted_alphas, beam = general_beam_with_attention(model,encoder_output, beam_size, max_caption_len, store_beam, enc_attention_masks, size_of_cross_attention)    
    return sorted_sequences, sorted_alphas, beam