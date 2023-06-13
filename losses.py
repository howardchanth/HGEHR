import torch
from torch import nn
import torch.nn.functional as F

def calculate_kl(mu_q, sig_q, mu_p, sig_p):
    kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).mean()
    return kl

class KLDivergence(nn.Module):
    def __init__(self):
        super(KLDivergence, self).__init__()

    def forward(self, P, Q):
        p = F.softmax(P, dim=-1)
        kl = torch.sum(p * (F.log_softmax(P, dim=-1) - F.log_softmax(Q, dim=-1)))

        return torch.mean(kl)