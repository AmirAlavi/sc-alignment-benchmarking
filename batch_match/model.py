import torch
import torch.nn as nn
import torch.nn.functional as F

activations = {
    'tanh': nn.Tanh,
    'sigmoid': nn.Sigmoid,
    'relu': nn.ReLU
}

class DomainInvariantAutoencoder(nn.Module):
    def __init__(self, input_size, layer_sizes, act='tanh', dropout=0.0, batch_norm=False):
        super(DomainInvariantAutoencoder, self).__init__()
        self.input_size = input_size
        self.layer_sizes = layer_sizes
        self.act = activations[act]
        self.dropout = dropout
        self.batch_norm = batch_norm

        self.encoder = nn.Sequential()
        self.decoder = nn.Sequential()
        prev_size = self.input_size
        for layer, size in enumerate(layer_sizes):
            # Apply dropout
            if self.dropout > 0:
                self.encoder.add_module('enc_dropout_{}'.format(layer), nn.Dropout(p=self.dropout))
            # Linearity
            self.encoder.add_module('enc_lin_{}'.format(layer), nn.Linear(prev_size, size))
            # BN
            if self.batch_norm:
                self.encoder.add_module('enc_batch_norm_{}'.format(layer), nn.BatchNorm1d(size))
            # Finally, non-linearity
            self.encoder.add_module('enc_{}_{}'.format(act, layer), activations[act]())
            prev_size = size

        reversed_layer_list = list(self.encoder.named_modules())[::-1]
        decode_layer_count = 0
        for name, module in reversed_layer_list:
            if 'lin_' in name:
                size = module.weight.data.size()[1]
                if self.dropout > 0:
                    self.decoder.add_module('dec_dropout_{}'.format(decode_layer_count), nn.Dropout(p=self.dropout))
                # Linearity
                linearity = nn.Linear(prev_size, size)
                linearity.weight.data = module.weight.data.transpose(0, 1)
                self.decoder.add_module('dec_lin_{}'.format(decode_layer_count), linearity)
                if decode_layer_count < len(self.layer_sizes) - 1:
                    # BN
                    if self.batch_norm:
                        self.decoder.add_module('dec_batch_norm_{}'.format(decode_layer_count), nn.BatchNorm1d(size))
                    # Finally, non-linearity
                    self.decoder.add_module('dec_{}_{}'.format(act, decode_layer_count), activations[act]())
                prev_size = size
                decode_layer_count += 1

    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return reconstructed
