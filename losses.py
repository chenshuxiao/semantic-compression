import torch
from torch.nn import functional as F
from torch import Tensor
from vae import VAE

def loss_function(recons: Tensor,
                input: Tensor,
                mu: Tensor,
                logvar: Tensor,
                kld_weight: float= 0.05,
                vae: VAE= None,
                enc_ortho_coeff:float= 0.0,
                dec_ortho_coeff:float= 0.0) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss = F.mse_loss(recons, input)

        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        enc_ortho_loss = 0
        if enc_ortho_coeff > 0:
            enc_ortho_loss = ortho(vae.encoder)

        dec_ortho_loss = 0
        if dec_ortho_coeff > 0:
            dec_ortho_loss = ortho(vae.decoder)

        return {'recon': recons_loss,
                'KLD': kld_weight * kld_loss,
                'enc_ortho': enc_ortho_coeff * enc_ortho_loss,
                'dec_ortho': dec_ortho_coeff * dec_ortho_loss}

def ortho(model):
    ortho_penalty = []
    cnt = 0
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            if m.kernel_size == (7, 7) or m.weight.shape[1] == 3:
                continue
            o = ortho_conv(m)
            cnt += 1
            ortho_penalty.append(o)
    ortho_penalty = sum(ortho_penalty)
    return ortho_penalty

def ortho_conv(m, device='cuda'):
    operator = m.weight
    operand = torch.cat(torch.chunk(m.weight, m.groups, dim=0), dim=1)
    transposed = m.weight.shape[1] < m.weight.shape[0]
    num_channels = m.weight.shape[1] if transposed else m.weight.shape[0]
    if transposed:
        operand = operand.transpose(1, 0)
        operator = operator.transpose(1, 0)
    gram = F.conv2d(operand, operator, padding=(m.kernel_size[0] - 1, m.kernel_size[1] - 1),
                    stride=m.stride, groups=m.groups)
    identity = torch.zeros(gram.shape).to(device)
    identity[:, :, identity.shape[2] // 2, identity.shape[3] // 2] = torch.eye(num_channels).repeat(1, m.groups)
    out = torch.sum((gram - identity) ** 2.0) / 2.0
    return out

class BetaSchedulerCyclic:
    """linear cyclic
    """

    def __init__(self, start=0.0, stop=1.0,  period=1000, ratio=0.5):
        # names are constant, cyclic, and monotonic
        self.start = start
        self.stop = stop
        self.period = period
        self.ratio = ratio
        self.step = (stop-start)/(period*ratio)
        self.v, self.i = start - self.step, 0
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.period:
            self.i += 1
            if self.v < self.stop:
                self.v += self.step
        else:
            self.v, self.i = self.start, 0
        return self.v

class BetaSchedulerMono:
    """linear monotonically increasing to max
    """

    def __init__(self, start=0.0, stop=1.0,  period=1000):
        # names are constant, cyclic, and monotonic
        self.start = start
        self.stop = stop
        self.period = period
        self.ratio = ratio
        self.step = (stop-start)/(period*ratio)
        self.v, self.i = start - self.step, -1
    
    def __iter__(self):
        return self

    def __next__(self):
        if self.i < self.period:
            self.i += 1
            if self.v < self.stop:
                self.v += self.step
        return self.v
