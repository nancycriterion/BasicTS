from dataclasses import dataclass,field
from basicts.configs import BasicTSModelConfig

@dataclass

class myModelConfig(BasicTSModelConfig):
    """
    myModelConfig 的 Docstring
    """
    input_len: int = field(default=None, metadata={"help": "Input sequence length."})
    output_len: int = field(default=None, metadata={"help": "Output sequence length."})
    num_features: int = field(default=None, metadata={"help": "Number of features."})
    num_layers2imf: int = field(default=2, metadata={"help": "Number of mixer layers."})