import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class FreqEncoder(nn.Module):
    def __init__(self, input_dim, max_freq_log2, N_freqs,
                 log_sampling=True, include_input=True,
                 periodic_fns=(torch.sin, torch.cos)):

        super().__init__()

        self.input_dim = input_dim
        self.include_input = include_input
        self.periodic_fns = periodic_fns

        self.output_dim = 0
        if self.include_input:
            self.output_dim += self.input_dim

        self.output_dim += self.input_dim * N_freqs * len(self.periodic_fns)

        if log_sampling:
            self.freq_bands = 2. ** torch.linspace(0., max_freq_log2, N_freqs)
        else:
            self.freq_bands = torch.linspace(
                2. ** 0., 2. ** max_freq_log2, N_freqs)

        self.freq_bands = self.freq_bands.numpy().tolist()

    def forward(self, input, **kwargs):
        '''
        weights (inpired by BARF)
        '''
        def w(alpha, k):

            if (alpha < k):
                return 0
            elif ((alpha-k) >= 0 and (alpha - k) < 1):
                return ((1-np.cos((alpha - k) * 3.141))/2)
            elif ((alpha - k) >= 1):
                return 1
            else:
                AssertionError

        out = []

        # print()
        # print(kwargs)
        step = kwargs["step"]

        # alpha = 3*(2.718)**(step//2000)  # TODO: used to be 2000
        alpha = 500  # TODO: used to be 2000

        if alpha <= 3:
            alpha = 3

        # print("alpha: {}".format(alpha))

        if self.include_input:
            out.append(input)

        for i in range(len(self.freq_bands)):
            # print(w(alpha, self.freq_bands[i]))
            freq = w(alpha, self.freq_bands[i]) * self.freq_bands[i]
            for p_fn in self.periodic_fns:
                out.append(p_fn(input * freq))

        out = torch.cat(out, dim=-1)

        return out


def get_encoder(encoding, input_dim=3,
                multires=6,
                degree=4,
                num_levels=16, level_dim=2, base_resolution=16, log2_hashmap_size=19, desired_resolution=2048, align_corners=False,
                **kwargs):

    if encoding == 'None':
        return lambda x, **kwargs: x, input_dim

    elif encoding == 'frequency':
        encoder = FreqEncoder(
            input_dim=input_dim, max_freq_log2=multires-1, N_freqs=multires, log_sampling=True)

    elif encoding == 'sphere_harmonics':
        from shencoder import SHEncoder
        encoder = SHEncoder(input_dim=input_dim, degree=degree)

    elif encoding == 'hashgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution,
                              log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='hash', align_corners=align_corners)

    elif encoding == 'tiledgrid':
        from gridencoder import GridEncoder
        encoder = GridEncoder(input_dim=input_dim, num_levels=num_levels, level_dim=level_dim, base_resolution=base_resolution,
                              log2_hashmap_size=log2_hashmap_size, desired_resolution=desired_resolution, gridtype='tiled', align_corners=align_corners)

    elif encoding == 'ash':
        from ashencoder import AshEncoder
        encoder = AshEncoder(input_dim=input_dim, output_dim=16,
                             log2_hashmap_size=log2_hashmap_size, resolution=desired_resolution)

    else:
        raise NotImplementedError(
            'Unknown encoding mode, choose from [None, frequency, sphere_harmonics, hashgrid, tiledgrid]')

    return encoder, encoder.output_dim
