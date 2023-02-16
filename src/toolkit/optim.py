import torch.optim
from transformers import get_scheduler, AdamW 
from torch import nn

# ============================================================================ #
#                             OPTIMIZERS                                       #
# ============================================================================ #


def create_optimizer(model, lr):
    optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()), lr=float(lr))
    return optimizer

def create_adamw_optimizer(model, lr1):
    optimizer = AdamW(params=filter(lambda p: p.requires_grad, model.parameters()), lr=3e-5)
    return optimizer

def create_scheduler(optimizer, len_train_dataloader, num_epochs=15):
    num_training_steps = num_epochs * len_train_dataloader
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=len_train_dataloader,
        num_training_steps=len_train_dataloader
    )
    return lr_scheduler

# ============================================================================ #
#                             Loss                                             #
# ============================================================================ #

def create_criterion(name, device):
    if name == "cross_entropy":
        loss = nn.CrossEntropyLoss().to(device)
    elif name == "contrastive_loss":
        loss = ContrastiveLoss().to(device)
    elif name == "l1":
        loss = nn.L1Loss().to(device)
    return loss


# ============================================================================ #
#                             Optimization Loop                                #
# ============================================================================ #

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

