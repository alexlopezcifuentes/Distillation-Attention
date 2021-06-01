import torch
import torch.nn as nn
import torch.fft

"""
Using a DCT-driven Loss in Attention-based Knowledge-Distillation for Scene Recognition

DCT.py
Python class that codes the proposed DCT-driven method.

Fully developed by Anonymous Code Author.
"""


class DFTOurs(nn.Module):
    """
    Our implementation of DCT Loss between a pair of features
    """

    def __init__(self):
        super(DFTOurs, self).__init__()

    def forward(self, features_student, features_teacher):
        # Get the batch size
        bs, _, _, _ = features_student[-1].shape

        # Check the number of scales to use. L parameter in the paper
        n_scales = len(features_teacher)

        # Initialize the loss to zero
        scale_loss = torch.zeros(bs).float().cuda()

        # Loop over the L scales
        for scale in range(n_scales):
            # For each tensor of features build the activation map
            AMs_student = self.returnAM(features_student[scale])
            AMs_teacher = self.returnAM(features_teacher[scale])

            ffts_s = self.computeDCT(AMs_student)
            ffts_t = self.computeDCT(AMs_teacher)

            ffts_s = ffts_s.reshape(ffts_s.shape[0], -1)
            ffts_t = ffts_t.reshape(ffts_t.shape[0], -1)

            destillation_loss = torch.sqrt(torch.sum(torch.pow(ffts_s - ffts_t, 2), dim=1))

            scale_loss += destillation_loss

        return scale_loss

    def computeDCT(self, activation_map):

        # Get the size
        bs, h, w = activation_map.shape

        # Subtract mean value for the activation map so after the first DCT coefficient is 0.
        mean = torch.repeat_interleave(torch.unsqueeze(torch.mean(activation_map.view(bs, -1), dim=1), dim=1), h, dim=1)
        mean = torch.repeat_interleave(torch.unsqueeze(mean, dim=2), w, dim=2)

        activation_map = activation_map - mean

        # Compute 2D-DCT for the Student as separable transform. First in one dimension then in the other.
        # As there is no PyTorch implementation of DCT we use the DFT. DCT = abs(real(DFT))
        dct_am = torch.fft.fft(torch.fft.fft(activation_map, dim=1), dim=2)
        dct_am = torch.abs(torch.real(dct_am))

        # Normalize between 0-1 the 2D-DCT with min-max normalization.
        min_s = torch.repeat_interleave(torch.unsqueeze(torch.min(dct_am.view(bs, -1), dim=1)[0], dim=1), h, dim=1)
        min_s = torch.repeat_interleave(torch.unsqueeze(min_s, dim=2), w, dim=2)
        max_s = torch.repeat_interleave(torch.unsqueeze(torch.max(dct_am.view(bs, -1), dim=1)[0], dim=1), h, dim=1)
        max_s = torch.repeat_interleave(torch.unsqueeze(max_s, dim=2), w, dim=2)

        dct_am = (dct_am - min_s) / max_s

        return dct_am

    def returnAM(self, feature_conv):
        """
        Compute the CAM given features
        :param feature_conv: features of shape [BatchSize, Chns, T, H, W]
        :return:
        """

        # Obtain sizes from the features
        bs, nc, h, w = feature_conv.shape

        # Obtain activation maps as the mean aggregation of features.
        cam = feature_conv.pow(2).mean(1)

        # Min-max normalization for Activation Map.
        cam = cam.view(bs, -1)
        min = torch.min(cam, dim=1)[0]
        max = torch.max(cam, dim=1)[0]
        cam = cam - min[:, None]
        cam = cam / max[:, None]
        cam = cam.view(bs, h, w)

        return cam

