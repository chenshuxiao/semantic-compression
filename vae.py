import torch
from torch import nn
from torch import Tensor
from typing import *

""" contains vae classes.
"""
# TODO: make blocks of regular, resnet, iso, R-iso
# TODO: we will make regular and resnet first, then iso and R-iso as flags?

# NOTE: Interestingly, it may be the case that relu for encoding and leaky relu for decoding may provide
#       better results, https://arxiv.org/pdf/1511.06434.pdf

class VanillaVAE(nn.Module):
    """ 
    adapted from https://github.com/AntixK/PyTorch-VAE/ and https://github.com/podgorskiy/VAE
    """

    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 res: int,
                 block_count: int,
                 layer_mult: int = 64) -> None:
        """inits a vanilla VAE

        Parameters
        ----------
        in_channels : int
            number of channels in image data
        latent_dim : int
            dimensionality of latent(z) code
        res : int
            length of resolution of (square)image data
            for now only deals with powers of two...
        layer_count : int
            number of hidden layers of half of the network
        layer_mult : int
            first hidden layer dimension
        """
        super(VanillaVAE, self).__init__()

        self.latent_dim = latent_dim

        modules = []
        hidden_dims = [layer_mult*(2**i) for i in range(layer_count)]
        # print (hidden_dims)
        self.last_hdim = hidden_dims[-1]
        image_channels = in_channels

        # TODO: add resnet likeness 
        # https://github.com/julianstastny/VAE-ResNet18-PyTorch/blob/master/model.py

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size= 3, stride= 2, padding  = 1),
                    nn.BatchNorm2d(h_dim),
                    nn.ReLU())
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.code_len = res//(2**layer_count)
        self.fc_mu = nn.Linear(hidden_dims[-1]*self.code_len*self.code_len, latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1]*self.code_len*self.code_len, latent_dim)

        # Build Decoder
        modules = []

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1] * self.code_len * self.code_len)

        hidden_dims.reverse()

        # transpose convolution
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=3,
                                       stride = 2,
                                       padding=1,
                                       output_padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU())
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
                            nn.ConvTranspose2d(hidden_dims[-1],
                                               hidden_dims[-1],
                                               kernel_size=3,
                                               stride=2,
                                               padding=1,
                                               output_padding=1),
                            nn.BatchNorm2d(hidden_dims[-1]),
                            nn.LeakyReLU(),
                            nn.Conv2d(hidden_dims[-1], out_channels=image_channels,
                                      kernel_size= 3, padding= 1),
                            nn.Tanh())

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
        # print (result.shape)
        result = result.view(z.shape[0], self.last_hdim, self.code_len, self.code_len)
        result = self.decoder(result)
        result = self.final_layer(result)
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
   

   
class IsoVAE(VanillaVAE):
    def __init__(self,
                 in_channels: int,
                 latent_dim: int,
                 res: int,
                 layer_count: int,
                 layer_mult: int,
                 down: bool = True) -> None:
        """[summary]

        Parameters
        ----------
        same as VanillaVAE, see above

        down : bool, optional
            true if only isometry in first half of network, by default True
        """
        # super super init
        super().__init__(in_channels,
                 latent_dim,
                 res,
                 layer_count,
                 layer_mult)
        # TODO: Initialize to delta
    
    # TODO: change loss