import torch
from torch.nn import functional as F
from torch import Tensor

def loss_function(self,
                      recons: Tensor,
                      input: Tensor,
                      mu: Tensor,
                      logvar: Tensor,
                      kld_weight: float = 0.05) -> dict:
        """
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """

        recons_loss = F.mse_loss(recons, input)

        kld_loss = -0.5 * torch.mean(torch.sum(1 + logvar - mu ** 2 - logvar.exp(), dim = 1), dim = 0)

        return {'reconstruction_loss': recons_loss, 'KLD':kld_weight * kld_loss}
    
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
