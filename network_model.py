"""
This is the module where the model is defined. It uses the nn.Module as a backbone to create the network structure
"""
# Own modules

# Built in
import math
import numpy as np

# Pytorch module
import torch.nn as nn
import torch.nn.functional as F
import torch
from torch import pow, add, mul, div, sqrt, exp

# Define class and computation
class Lorentz(nn.Module):
    # Initialization
    def __init__(self, flags, num_spectra=300):
        super(Lorentz, self).__init__()
        self.num_lor = flags.num_lor  # Number of Lorentzian
        self.num_spectra = num_spectra  # Number of spectra point
        self.freq_low = flags.freq_low
        self.freq_high = flags.freq_high
        w_numpy = np.arange(flags.freq_low, flags.freq_high, (flags.freq_high - flags.freq_low) / self.num_spectra)
        cuda = True if torch.cuda.is_available() else False
        if cuda:
            self.w = torch.tensor(w_numpy).cuda()
        else:
            self.w = torch.tensor(w_numpy)

    # Forward function of the calculation
    def forward(self, w0_in, wp_in, g_in, eps_inf, d):
        # Make sure the shape is correct
        # assert np.shape(w0)[0] == self.num_lor, "Your w0 is not the correct size"
        # assert len(wp) == self.num_lor, "Your wp is not the correct size"
        # assert len(g) == self.num_lor, "Your g is not the correct size"

        # Expand operators
        w0 = w0_in.unsqueeze(1).expand(self.num_lor, self.num_spectra)
        wp = wp_in.unsqueeze(1).expand_as(w0)
        g = g_in.unsqueeze(1).expand_as(w0)
        w_expand = self.w.expand_as(g)

        e2_num = mul(pow(wp, 2), mul(w_expand, g))
        e1_num = mul(pow(wp, 2), add(pow(w0, 2), -pow(w_expand, 2)))
        denom = add(pow(add(pow(w0, 2), -pow(w_expand, 2)), 2), mul(pow(w_expand, 2), pow(g, 2)))

        # Sum up the Oscillators
        e2 = torch.sum(div(e2_num, denom), 0)
        e1 = torch.sum(div(e1_num, denom), 0) + eps_inf

        # Get n, k
        e1_e2_sqrt_sum = sqrt(add(pow(e1, 2), pow(e2, 2)))
        n = sqrt(add(e1_e2_sqrt_sum, e1) / 2)
        k = sqrt(add(e1_e2_sqrt_sum, -e1) / 2)

        # Get T(w)
        T = mul(div(4 * n, add(pow(n + 1, 2), pow(k, 2))), exp(mul(-4 * np.pi * mul(d, k), 0.0033 * self.w)))

        return T, e2, e1