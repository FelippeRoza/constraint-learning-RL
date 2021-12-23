from functional import seq
from torch.nn import Linear, Module, ModuleList
import torch.nn.functional as F


class Net(Module):
    def __init__(self,
                 in_dim,
                 out_dim,
                 layer_dims):
        super(Net, self).__init__()

        _layer_dims = [in_dim] + layer_dims + [out_dim]
        self._layers = ModuleList(seq(_layer_dims[:-1])
                                  .zip(_layer_dims[1:])
                                  .map(lambda x: Linear(x[0], x[1]))
                                  .to_list())

    def forward(self, inp):
        out = inp

        for layer in self._layers:
            out = F.relu(layer(out))

        return out.double()
