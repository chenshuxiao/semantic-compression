import torch
from torch import nn
from torch import Tensor
from blocks import *
from typing import *

""" contains vae classes.
"""

# NOTE: Interestingly, it may be the case that relu for encoding and leaky relu for decoding may provide
#       better results, https://arxiv.org/pdf/1511.06434.pdf


class VAE(nn.Module):
    """ 
    inspired by: https://github.com/AntixK/PyTorch-VAE/ and https://github.com/podgorskiy/VAE
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 res: int,
                 stage_count: int = 4,
                 layer_mult: int = 64,
                 d: int = 1,
                 resnet: bool = False,
                 kwargs_enc: dict = {
                                        'HAS_BN': True,
                                        'SReLU': False,
                                        'HAS_ST': False,
                                        'DROPOUT': 0,
                                        'CIFAR': True,
                                        'DIRAC_INIT': False
                                    },
                 kwargs_dec: dict = {
                                        'HAS_BN': False,
                                        'SReLU': False,
                                        'HAS_ST': False,
                                        'DROPOUT': 0,
                                        'CIFAR': True,
                                        'DIRAC_INIT': False
                                    }
                ) -> None:
        """inits a VAE

        Parameters
        ----------
        in_channels : int
            number of channels in image data
        latent_dim : int
            dimensionality of latent(z) code
        res : int
            length of resolution of (square)image data
            for now only deals with powers of two...
        stage_count : int
            number of hidden stage layers of half of the network, by default 4
        layer_mult : int
            first hidden layer dimension, by default 64
        d : int or array
            # of transforms per block, by default 1
        resnet : bool
            flag to add batch_normalization and skip connections, by default False
        """

        super().__init__()

        self.latent_dim = latent_dim

        hidden_dims = [layer_mult*(2**i) for i in range(stage_count)]
        hidden_dims = hidden_dims
        self.last_hdim = hidden_dims[-1]
        self.image_channels = in_channels

        # TODO: might have to add extra d for just the decoder
        if type(d) == int:
            d = [d]*stage_count
        
        print (hidden_dims)
        print (d)

        # Build Encoder
        self._construct_encoder(in_channels, hidden_dims, d, **kwargs_enc)
        # self._init_coder(self.encoder, **kwargs_enc)

        if kwargs_enc['DROPOUT'] > 0:
            self.dropout = nn.Dropout(p=kwargs_enc['DROPOUT'], inplace=True)
        else:
            self.dropout = None

        # NOTE: honestly variational inference might not be super useful. Perhaps
        #   later see how a plain autoencoder does
        self.code_len = res//(2**(stage_count-1))
        if not kwargs_enc['CIFAR']:
            # stride 2 conv and stride 2 max pool stem
            self.code_len = self.code_len // 4

        p_zlen = hidden_dims[-1]*self.code_len**2
        self.fc_mu = nn.Linear(p_zlen, latent_dim)
        self.fc_var = nn.Linear(p_zlen, latent_dim)

        
        # Build Decoder
        hidden_dims.reverse()
        modules = []

        self.decoder_input = nn.Linear(latent_dim, p_zlen)

        self._construct_decoder(hidden_dims, d, **kwargs_dec)
        # self._init_coder(self.encoder, **kwargs_dec)

    def _construct_encoder(self, in_channels, hidden_dims, d, **kwargs):
        modules = []
        # stem
        modules.append(ResStem(w_in= in_channels, w_out= hidden_dims[0], **kwargs))
        # stages
        w_in = hidden_dims[0]
        for i, w_out in enumerate(hidden_dims):
            print (i)
            stride = 2
            if w_in == w_out:
                stride = 1
            modules.append(ResStage(w_in=w_in, w_out=w_out, stride=stride, d=d[i], **kwargs))
            w_in = w_out
        self.encoder = nn.Sequential(*modules)
        # head is fc_mu fc_var

    def _construct_decoder(self, hidden_dims, d, **kwargs):
        modules = []
        # first let's ignore the stem
        # TODO: change this to WDSR or EDSR, currently slightly broken, see jupyter notebook
        for i, w in enumerate(hidden_dims[:-2]):
            modules.append(ResStage(w_in=w, w_out=w*2, stride=1, d=d[i], skip_relu=True, **kwargs))
            modules.append(nn.PixelShuffle(2))
            # may be beneficial to not have this ReLU
            # modules.append(nn.ReLU(True) if not kwargs['SReLU'] else SReLU(w_out))
        
        modules.append(ResStage(w_in=hidden_dims[-2], w_out=self.image_channels*4, stride=1, d=d[i], skip_relu=True, **kwargs))
        modules.append(nn.PixelShuffle(2))
        # TODO: investigate this activation
        # modules.append(nn.Tanh())
        # TODO: add stem based on kwargs cifar
        self.decoder = nn.Sequential(*modules)

    # TODO: investigate init_coder, currently commented out in __init__
    def _init_coder(self, coder, **kwargs):
        for m in coder.modules():
            if isinstance(m, nn.Conv2d):
                if kwargs['DIRAC_INIT']:
                    # the first 7x7 convolution we use pytorch default initialization
                    # and not enforce orthogonality since the large input/output channel difference
                    if m.kernel_size != (7, 7):
                        nn.init.dirac_(m.weight)
            # otherwise just default kaiming
            elif isinstance(m, nn.BatchNorm2d):
                zero_init_gamma = (
                    hasattr(m, 'final_bn') and m.final_bn
                )
                m.weight.data.fill_(0.0 if zero_init_gamma else 1.0)
                m.bias.data.zero_()

    def encode(self, input: Tensor) -> List[Tensor]:
        """
        Encodes the input by passing through the encoder network
        and returns the latent codes.
        :param input: (Tensor) Input tensor to encoder [N x C x H x W]
        :return: (Tensor) List of latent codes
        """
        result = self.encoder(input)
        result = torch.flatten(result, start_dim=1)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        if self.dropout:
            result = self.dropout(result)
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        return [mu, log_var]

    def decode(self, z: Tensor) -> Tensor:
        """
        Maps the given latent codes
        onto the image space.
        :param z: (Tensor) [N x D] , D is latent dim
        :return: (Tensor) [N x C x H x W]
        """
        result = self.decoder_input(z)
        result = result.view(z.shape[0], self.last_hdim, self.code_len, self.code_len)
        result = self.decoder(result)
        # result = self.final_layer(result)
        return result

    def reparameterize(self, mu: Tensor, logvar: Tensor) -> Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return eps * std + mu
        else:
            return mu

    def forward(self, input: Tensor) -> List[Tensor]:
        mu, log_var = self.encode(input)
        z = self.reparameterize(mu, log_var)
        return  [self.decode(z), mu, log_var]


    def sample(self,
               num_samples:int,
               current_device) -> Tensor:
        """
        Samples from the latent space and return the corresponding
        image space map.
        :param num_samples: (Int) Number of samples
        :param current_device: (Int) Device to run the model
        :return: (Tensor)
        """
        z = torch.randn(num_samples,
                        self.latent_dim)

        z = z.to(current_device)

        samples = self.decode(z)
        return samples

    def generate(self, x: Tensor) -> Tensor:
        """
        Given an input image x, returns the reconstructed image
        :param x: (Tensor) [B x C x H x W]
        :return: (Tensor) [B x C x H x W]
        """

        return self.forward(x)[0]
   

