import logging
import csv
import json
import io

from typing import Tuple
from overrides import overrides

from allennlp.data import Instance, DatasetReader
from allennlp.models import Model
from allennlp.predictors.predictor import Predictor, JsonDict

logger = logging.getLogger(__name__)

def csv2string(data):
    si = io.StringIO()
    cw = csv.writer(si)
    cw.writerow(data)
    return si.getvalue().strip('\r\n')

@Predictor.register('suggestion_mining')
class SuggestionMiningPredictor(Predictor):

    def load_line(self, line: str) -> JsonDict:  # pylint: disable=no-self-use
        """
        If your inputs are not in JSON-lines format (e.g. you have a CSV)
        you can override this function to parse them correctly.
        """
        sentence_id, sentence, gold = list(csv.reader([line]))[0]

        return {
            'sentence_id': sentence_id,
            'sentence': sentence,
            'gold_label': int(gold)
        }

    def dump_line(self, outputs: JsonDict) -> str:  # pylint: disable=no-self-use
        """
        If you don't want your outputs in JSON-lines format
        you can override this function to output them differently.
        """
        csv_data = [outputs["sentence_id"], outputs["sentence"], outputs["label"]]
        return f'{csv2string(csv_data)}\n'

    def predict_json(self, inputs: JsonDict) -> JsonDict:
        instance = self._json_to_instance(inputs)
        outputs = inputs

        if len(instance.fields['text'].tokens) > 4:
            prediction = self.predict_instance(instance)
            outputs['label'] = int(prediction['label'])
            outputs['suggestion_probability'] = prediction['class_probabilities'][1]
        else:
            outputs['label'] = 0
            outputs['suggestion_probability'] = 0.0
        return inputs

    @overrides
    def _json_to_instance(self, json_dict: JsonDict) -> Tuple[Instance, JsonDict]:
        sentence = json_dict['sentence']
        instance = self._dataset_reader.text_to_instance(text=sentence)
        return instance

