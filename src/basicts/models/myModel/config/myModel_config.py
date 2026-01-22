from dataclasses import dataclass,field
from basicts.configs import BasicTSModelConfig

@dataclass

class MyModelConfig(BasicTSModelConfig):
    """
    MyModelConfig 的 Docstring
    """
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_layers2imf: int = field(default=3, metadata={"help": "Number of mixer layers."})
    centor_method: str = field(default='mean', metadata={"help": "Centering method for input data."})
    cnn_out_channels: int = field(default=128, metadata={"help": "Number of output channels in CNN layers."})
    hidden_layers: int=field(default=64,metadata='')
    n_heads: int=field(default=4,metadata='')
    dropout: float=field(default=0.1,metadata='')
    nums_DMS: int=field(default=5,metadata='')
    hidden_sizes: list=field(default_factory=lambda: [16,32,64, 128], metadata={"help": "Number of hidden units in each layer of the MLP."})