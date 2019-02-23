from overrides import overrides

import torch

from allennlp.modules.seq2vec_encoders.seq2vec_encoder import Seq2VecEncoder
from allennlp.nn.util import get_lengths_from_binary_sequence_mask

@Seq2VecEncoder.register("select_time_step")
class SelectTimeStepEncoder(Seq2VecEncoder):
    """
    A ``SelectTimeStepEncoder`` is a simple :class:`Seq2VecEncoder` which simply returns a single timestep from
    a sequence across the time dimension. The input to this module is of shape ``(batch_size, num_tokens,
    embedding_dim)``, and the output is of shape ``(batch_size, embedding_dim)``.

    Parameters
    ----------
    embedding_dim: ``int``
        This is the input dimension to the encoder.
    timestep: ``int`` (default=``0``)
        The timestep to be selected
    """
    def __init__(self,
                 embedding_dim: int,
                 timestep: int) -> None:
        super(SelectTimeStepEncoder, self).__init__()
        self._embedding_dim = embedding_dim
        self._timestep = timestep

    @overrides
    def get_input_dim(self) -> int:
        return self._embedding_dim

    @overrides
    def get_output_dim(self) -> int:
        return self._embedding_dim

    def forward(self, tokens: torch.Tensor, mask: torch.Tensor = None):  #pylint: disable=arguments-differ
        
        # Our input has shape `(batch_size, num_tokens, embedding_dim)`, so we sum out the `num_tokens`
        # dimension.
        selected = tokens[:,self._timestep,:].squeeze(1)
        return selected
