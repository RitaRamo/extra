from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.bleu.bleu import Bleu

import sys
sys.path.append("src/cider")
sys.path.append("src/cider/data")

from pyciderevalcap.ciderD.ciderD import CiderD

#from pyciderevalcap.ciderD.ciderD import CiderD
import torch

class CriticalTraining():

    def __init__(self, device):
        super().__init__()

        self.device = device
        self.scorer = Cider()
        #self.scorer = CiderD(df="coco-val")
        #self.bleu_score = Bleu()


    def get_loss_mean(self, gen_captions_tokens, scores, id2caption, id2targets):
        #reward dim -> torch.tensor(n_batch)
        _, reward = self.scorer.compute_score(id2targets,id2caption)
        reward= torch.tensor(reward, device=self.device)

        reward_baseline = torch.mean(reward)

        loss = self.get_loss(reward, reward_baseline, gen_captions_tokens, scores)

        return loss, reward_baseline

    def get_loss_greedy(self, gen_captions_tokens, scores, id2samples, id2greedy, id2targets):
        #reward samples
        _, reward = self.scorer.compute_score(id2targets,id2samples)
        reward= torch.tensor(reward.data, device=self.device)
        
        #reward greedy
        _,reward_baseline = self.scorer.compute_score(id2targets,id2greedy)
        reward_baseline= torch.tensor(reward_baseline.data, device=self.device)

        loss = self.get_loss(reward, reward_baseline, gen_captions_tokens, scores)

        return loss, torch.mean(reward_baseline)

    def get_loss_greedy_with_bleu(self, gen_captions_tokens, scores, id2samples, id2greedy, id2targets):
        #reward samples
        _, reward_cider = self.scorer.compute_score(id2targets,id2samples)
        reward_cider= torch.tensor(reward_cider.data, device=self.device)
        
        _, bleus = self.bleu_score.compute_score(id2targets, id2samples)
        reward_bleu4 = bleus[-1]
        reward_bleu4 = torch.tensor(reward_bleu4, device=self.device)
        
        reward= reward_cider + reward_bleu4
        
        #reward greedy
        _,reward_baseline_cider = self.scorer.compute_score(id2targets,id2greedy)
        reward_baseline_cider= torch.tensor(reward_baseline_cider.data, device=self.device)
        
        _, reward_baseline_bleus = self.bleu_score.compute_score(id2targets, id2greedy)
        reward_baseline_bleu4 = reward_baseline_bleus[-1]
        reward_baseline_bleu4 = torch.tensor(reward_baseline_bleu4, device=self.device)

        reward_baseline = reward_baseline_cider + reward_baseline_bleu4

        loss = self.get_loss(reward, reward_baseline, gen_captions_tokens, scores)

        return loss, torch.mean(reward_baseline)

    # def get_loss(self,reward, reward_baseline, gen_captions_tokens, scores):
    #     seq_len = scores.size()[1]

    #     critical_reward = reward - reward_baseline

    #     #critical_reward.unsqueeze(1) -> (n_batch,1)
    #     #critical_reward.unsqueeze(1).repeat(1, seq_len) -> (n_batch,seq_len)
    #     critical_reward= critical_reward.unsqueeze(1).repeat(1, seq_len)

    #     #reshape to flatten
    #     critical_reward = critical_reward.reshape(-1)
    #     scores = scores.reshape(-1)

    #     #masking to just consider the tokens that are not padding
    #     mask = (gen_captions_tokens>0).to(self.device)
    #     mask = mask.reshape(-1)

    #     loss = - scores * critical_reward * mask
    #     loss = torch.sum(loss) / torch.sum(mask)

    #     return loss

    
    def get_loss(self,reward, reward_baseline, gen_captions_tokens, scores):
        seq_len = scores.size()[1]

        critical_reward = reward - reward_baseline
        #critical_reward.unsqueeze(1) -> (n_batch,1)
        #critical_reward.unsqueeze(1).repeat(1, seq_len) -> (n_batch,seq_len)
        #critical_reward= critical_reward.unsqueeze(1).repeat(1, seq_len)

        #masking to just consider the tokens that are not padding
        mask = (gen_captions_tokens>0).to(self.device)

        scores = scores* mask
        scores = torch.sum(scores, dim=-1) / torch.sum(mask, dim=-1)
        loss = - scores * critical_reward 

        loss = loss.mean()
        
        return loss