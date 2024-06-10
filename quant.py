import torch
import torch.nn as nn


def lookup_quant(x, lookup_values):
    """
    x: floating point tensor to be quantized
    lookup_values: tensor of predefined constant values for look-up quantization
    """
    # TODO: check on efficiency of frequent transfers
    lookup_values = lookup_values.to(x.device)

    # Expanding dimensions for broadcasting in distance calculation
    expanded_x = x.unsqueeze(-1)
    expanded_lookup = lookup_values.unsqueeze(0)

    # Calculate the distance from each value in 'x' to each value in 'lookup_values'
    distances = torch.abs(expanded_x - expanded_lookup)

    # Get the index of the minimum distance for each value in 'x'
    _, min_indices = torch.min(distances, dim=-1)
    
    # Use these indices to get the corresponding quantized values from 'lookup_values'
    quant_x = lookup_values[min_indices]

    return quant_x

def quantize_qfna(x, scale, zero, maxq, lookup_values=None):
    q = (x / scale) + zero
    if lookup_values is not None:
        q = lookup_quant(q, lookup_values)
    else: 
        q = torch.clamp(torch.round(q), 0, maxq)
    q = scale * (q - zero)
    return q

def quantize_qfnb(x, scale, maxq, lookup_values=None):
    q = x / scale
    q = ((q+1)/2) * maxq
    if lookup_values is not None:
        q = lookup_quant(q, lookup_values)
    else:
        q = torch.clamp(torch.round(q), 0, maxq)
    q = (q / maxq) * 2 - 1
    q = q * scale
    return q

class Quantizer(nn.Module):

    def __init__(self, shape=1):
        super(Quantizer, self).__init__()
        self.register_buffer('maxq', torch.tensor(0))
        self.register_buffer('scale', torch.zeros(shape))
        self.register_buffer('zero', torch.zeros(shape))

    def configure(self,
                  bits,
                  perchannel=False,
                  sym=True,
                  qfn='a',
                  mse=False,
                  norm=2.4,
                  grid=100,
                  maxshrink=.8,
                  lookup_values=None,
                  group_size=-1):
        # TODO: Handle lookup with mse
        assert mse == False
        self.maxq = torch.tensor(2**bits - 1)
        self.perchannel = perchannel
        self.sym = sym
        self.qfn = qfn
        self.norm = norm
        self.grid = grid
        self.maxshrink = maxshrink
        self.lookup_values = lookup_values
        if self.lookup_values is not None:
            lookup_values = torch.tensor(lookup_values)
            self.lookup_values = lookup_values
        self.group_size = group_size

    def find_params(self, x, weight=False, group_size=-1):
        if self.qfn == 'a':
            self.find_params_qfna(x, weight=weight)
        elif self.qfn == 'b':
            self.find_params_qfnb(x)
        else: 
            raise NotImplementedError()

    def find_params_qfna(self, x, weight=False):
        assert self.perchannel
        assert weight

        self.maxq = self.maxq.to(x.device)
        shape = x.shape
        x = x.flatten(1)
        
        if self.group_size == -1:
            self.group_size = x.shape[1]
        num_groups = x.shape[1] // self.group_size
        assert num_groups * self.group_size == x.shape[1], "Group size must divide the number of columns evenly"

        x = x.reshape(-1, num_groups, self.group_size)

        tmp = torch.zeros((x.shape[0], num_groups), device=x.device)
        xmin = torch.minimum(x.min(2)[0], tmp)
        xmax = torch.maximum(x.max(2)[0], tmp)

        if self.sym:
            xmax = torch.maximum(torch.abs(xmin), xmax)
            xmin = -xmax
        tmp = (xmin == 0) & (xmax == 0)
        xmin[tmp] = -1
        xmax[tmp] = +1

        self.scale = (xmax - xmin) / self.maxq
        if self.sym:
            zp = (self.maxq + 1) / 2
            zp = zp.expand_as(self.scale)
        else:
            zp = -xmin / self.scale
        # Round the zero-point to ensure no error when quantizing zeros
        if self.lookup_values is not None:
            self.zero = lookup_quant(zp, self.lookup_values)
        else:
            self.zero = torch.round(zp)

        # Reshape scale and zero to match the original input shape
        # Create the desired shape for broadcasting
        scale_shape = [-1] + [1] * (len(shape) - 2) + [num_groups, 1]
        self.scale = self.scale.reshape(scale_shape)
        self.zero = self.zero.reshape(scale_shape)

        # Expand the scales and zero points to match the original shape
        self.scale = self.scale.expand(-1, -1, self.group_size).reshape(shape)
        self.zero = self.zero.expand(-1, -1, self.group_size).reshape(shape)

    def find_params_qfnb(self, x):
        dev = x.device
        self.maxq  = self.maxq.to(dev)
        self.scale = None  #needs to be calculated after preproc
        self.zero  = None

    def quantize(self, x):
        if self.qfn == 'a':
            assert self.ready()
            return quantize_qfna(x, self.scale, self.zero, self.maxq, self.lookup_values)
        elif self.qfn == 'b':
            assert torch.all(self.maxq != 0)
            self.scale = 2.4 * x.square().mean().sqrt() + 1e-16
            return quantize_qfnb(x, self.scale, self.maxq, self.lookup_values)
        else:
            return NotImplementedError()

    def ready(self):
        return torch.all(self.scale != 0)
