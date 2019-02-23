from typing import Dict, Optional

import numpy
from overrides import overrides
import torch
import torch.nn.functional as F
import torch.nn as nn

from allennlp.common import Params
from allennlp.common.checks import ConfigurationError
from allennlp.data import Vocabulary
from allennlp.modules import FeedForward, Seq2VecEncoder, TextFieldEmbedder
from allennlp.models.model import Model
from allennlp.nn import InitializerApplicator, RegularizerApplicator
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy, Metric, F1Measure


@Model.register("text_classifier")
class TextClassifier(Model):
    """
    This ``Model`` performs multi class text classification for a text.  We assume we're given a
    text and we predict some output label.
    The basic model structure: we'll embed the text and encode it with
    a Seq2VecEncoder, getting a single vector representing the content.  We'll then
    the result through a feedforward network, the output of
    which we'll use as our scores for each label.
    Parameters
    ----------
    vocab : ``Vocabulary``, required
        A Vocabulary, required in order to compute sizes for input/output projections.
    model_text_field_embedder : ``TextFieldEmbedder``, required
        Used to embed the ``tokens`` ``TextField`` we get as input to the model.
    internal_text_encoder : ``Seq2VecEncoder``
        The encoder that we will use to convert the input text to a vector.
    classifier_feedforward : ``FeedForward``
    embedding_token_dropout: ``float``, optional (default=``None``)
        Dropout entire embedded tokens with the given probability
    embedding_dropout: ``float``, optional (default=``None``)
        Dropout each value the hidden dimension of embedded tokens with given probability
    initializer : ``InitializerApplicator``, optional (default=``InitializerApplicator()``)
        Used to initialize the model parameters.
    regularizer : ``RegularizerApplicator``, optional (default=``None``)
        If provided, will be used to calculate the regularization penalty during training.
    """
    def __init__(self, vocab: Vocabulary,
                 model_text_field_embedder: TextFieldEmbedder,
                 internal_text_encoder: Seq2VecEncoder,
                 classifier_feedforward: FeedForward,
                 use_batch_norm: bool = False,
                 embedding_token_dropout: Optional[float] = None,
                 embedding_dropout: Optional[float] = None,
                 initializer: InitializerApplicator = InitializerApplicator(),
                 regularizer: Optional[RegularizerApplicator] = None) -> None:
        super().__init__(vocab, regularizer)

        self._model_text_field_embedder = model_text_field_embedder
        self._num_classes = self.vocab.get_vocab_size("labels")
        self._internal_text_encoder = internal_text_encoder
        self._classifier_feedforward = classifier_feedforward
        self._embedding_token_dropout = nn.Dropout(embedding_token_dropout) if embedding_token_dropout else None
        self._embedding_dropout = nn.Dropout(embedding_dropout) if embedding_dropout else None
        self._batch_norm = nn.modules.BatchNorm1d(num_features=internal_text_encoder.get_output_dim()) if use_batch_norm else None

        if model_text_field_embedder.get_output_dim() != internal_text_encoder.get_input_dim():
            raise ConfigurationError("The output dimension of the model_text_field_embedder must match the "
                                     "input dimension of the title_encoder. Found {} and {}, "
                                     "respectively.".format(model_text_field_embedder.get_output_dim(),
                                                            internal_text_encoder.get_input_dim()))

        self._metrics = {
            "accuracy": CategoricalAccuracy(),
            "f1": F1Measure(1) # Assuming binary classification and we set to 1 suggestion which is what semeval task is about.
        }
        self._loss = torch.nn.CrossEntropyLoss()

        initializer(self)

    @overrides
    def forward(self,  # type: ignore
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # pylint: disable=arguments-differ
        """
        Parameters
        ----------
        text : Dict[str, Variable], required
            The output of ``TextField.as_array()``.
        label : Variable, optional (default = None)
            A variable representing the label for each instance in the batch.
        Returns
        -------
        An output dictionary consisting of:
        class_probabilities : torch.FloatTensor
            A tensor of shape ``(batch_size, num_classes)`` representing a distribution over the
            label classes for each instance.
        loss : torch.FloatTensor, optional
            A scalar loss to be optimised.
        """
        embedded_text = self._model_text_field_embedder(text)
        text_mask = util.get_text_field_mask(text)
        if self._embedding_dropout:
            embedded_text = self._embedding_dropout(embedded_text)
        if self._embedding_token_dropout:
            embedded_text = self._embedding_token_dropout(embedded_text)
        encoded_text = self._internal_text_encoder(embedded_text, text_mask)
        if self._batch_norm:
            encoded_text = self._batch_norm(encoded_text)
        logits = self._classifier_feedforward(encoded_text)
        output_dict = {'logits': logits}
        if label is not None:
            loss = self._loss(logits, label)
            for metric in self._metrics.values():
                metric(logits, label)
            output_dict["loss"] = loss

        return output_dict

    @overrides
    def decode(self, output_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Does a simple argmax over the class probabilities, converts indices to string labels, and
        adds a ``"label"`` key to the dictionary with the result.
        """
        class_probabilities = F.softmax(output_dict['logits'], dim=-1)
        output_dict['class_probabilities'] = class_probabilities

        predictions = class_probabilities.cpu().data.numpy()
        argmax_indices = numpy.argmax(predictions, axis=-1)
        labels = [self.vocab.get_token_from_index(x, namespace="labels")
                  for x in argmax_indices]
        output_dict['label'] = labels
        return output_dict

    @overrides
    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        metrics = {}
        for metric_name, metric in self._metrics.items():
            if metric_name == 'f1':
                precision, recall, f1 = metric.get_metric(reset)
                metrics['precision'] = precision
                metrics['recall'] = recall
                metrics['f1'] = f1
            else:
                metrics[metric_name] = metric.get_metric(reset)
        return metrics
