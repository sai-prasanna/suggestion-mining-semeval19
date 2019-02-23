from typing import Dict, Optional
import json
import csv
import logging
import textacy
import re
import random
from overrides import overrides

from allennlp.common.file_utils import cached_path
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.fields import Field, TextField, LabelField, MetadataField
from allennlp.data.instance import Instance
from allennlp.data.token_indexers import SingleIdTokenIndexer, TokenIndexer
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name



@DatasetReader.register("suggestion_mining")
class SuggestionMiningReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 max_length: Optional[int] = 400,
                 oversample_n: int = 0) -> None:
        super().__init__(False)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}
        self._max_length = max_length
        self._oversample_n = oversample_n

    @overrides
    def _read(self, file_path: str):
        # if `file_path` is a URL, redirect to the cache
        file_path = cached_path(file_path)
        with open(file_path, 'rt', errors='ignore', encoding='utf-8') as sem_file:
            reader = csv.reader(sem_file, delimiter=',')
            logger.info("Reading Sem instances from csv dataset at: %s", file_path)
            next(reader, None)  # skip the headers
            
            instances = []
            for record in reader:
                _, sentence, label = record[0], record[1], record[2]
                instance = self.text_to_instance(sentence, label)
                if len(instance.fields['text'].tokens) > 4: # Logic causes validating and training vary bit with test
                    instances.append(instance)
        
        if 'train' in file_path.lower(): # We could rather use different dataset readers for train and val/test
            suggestions = []
            non_suggestions = []
            for instance in instances:
                if int(instance.fields['label'].label) == 1:
                    suggestions.append(instance)
                else:
                    non_suggestions.append(instance)
            instances = []
            oversample_times = self._oversample_n
            for _ in range(oversample_times + 1):
                instances.extend(suggestions)
            logger.info(f"suggestions {len(suggestions)}")
            logger.info(f"instances {len(instances)}")
            logger.info(f"non_suggestions {len(non_suggestions)}")
            instances.extend(non_suggestions)
            random.shuffle(instances)

        for instance in instances:
            yield instance

    @overrides
    def text_to_instance(self,  # type: ignore
                         text: str,
                         label: str = None) -> Instance:
        # pylint: disable=arguments-differ
        text = _normalize_text(text)
        fields: Dict[str, Field] = {}
        sentence_tokens = self._tokenizer.tokenize(text)
        if self._max_length:
            sentence_tokens = sentence_tokens[:self._max_length]
        fields['text'] = TextField(sentence_tokens, self._token_indexers)
        if label is not None:
            fields['label'] = LabelField(label)
        return Instance(fields)

def _normalize_text(text: str) -> str:
    soup = BeautifulSoup(text, 'html.parser')
    text = soup.get_text()
    text = text.replace('\n', ' ').strip('\'').strip('"')
    text = text.replace('___', ' ')
    text = text.replace('__', ' ')
    text = text.replace('""""""', '"')
    text = text.replace('"""""', '"')
    text = text.replace('""""', '"')
    text = text.replace('"""', '"')
    text = textacy.preprocess.remove_accents(text)
    text = textacy.preprocess.transliterate_unicode(text)
    text = textacy.preprocess.normalize_whitespace(text)
    return text

NUMBER_TOKEN = "@@NUMBER@@"
EMAIL_TOKEN = "@@EMAIL@@"
URL_TOKEN = "@@URL@@"

def _destructive_preprocessing(text: str) -> str:
    text = textacy.preprocess.unpack_contractions(text)
    text = textacy.preprocess.replace_emails(text, replace_with=EMAIL_TOKEN)
    text = textacy.preprocess.replace_numbers(text, replace_with=NUMBER_TOKEN)
    text = textacy.preprocess.replace_urls(text, replace_with=URL_TOKEN)
    text = textacy.preprocess.remove_punct(text, marks=r'`~!#$%^&*()-_=+[]{}\|;:<>/\"')
    return text