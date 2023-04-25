from torch import Tensor
from torch.nn import Module, Sequential, Linear, ReLU, Sigmoid


class DirectedAcyclicGraph(Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.input_size = input_size
        self.network = Sequential(
            Linear(input_size, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, input_size * input_size),
            Sigmoid()
        )

    def forward(self, data: Tensor):
        """
        Data format need to be [batch_idx, visit_idx, item_idx]
        """
        out = self.network(data)
        return out


class AcyclicDirectedMixGraph(Module):
    def __init__(self, input_size: int, hidden: int):
        super().__init__()
        self.input_size = input_size
        self.directed_network = Sequential(
            Linear(input_size, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, input_size * input_size),
            Sigmoid()
        )

        self.bi_directed_network = Sequential(
            Linear(input_size, hidden),
            ReLU(),
            Linear(hidden, hidden),
            ReLU(),
            Linear(hidden, input_size * input_size),
            Sigmoid()
        )

    def forward(self, data: Tensor):
        """
        Data format need to be [batch_idx, visit_idx, item_idx]
        """
        directed_output = self.directed_network(data)
        bi_directed_output = self.bi_directed_network(data)
        return directed_output, bi_directed_output
