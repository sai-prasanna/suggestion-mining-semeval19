import sys
import pathlib
import json
from scipy import stats
from hinton.predictors import SuggestionMiningPredictor
from hinton.data.dataset_readers import SuggestionMiningReader
from hinton.model import TextClassifier
from allennlp.models.archival import load_archive
from allennlp.predictors import Predictor
from sklearn.metrics import precision_recall_fscore_support
import logging
import numpy as np

experiment_dir = pathlib.Path(sys.argv[1])
test_data_path = pathlib.Path(sys.argv[2])

if len(sys.argv) == 4 and sys.argv[3] == 'semi':
    semi_supervised_ensemble = True
    single_model_prefix = 'best_model'
else:
    semi_supervised_ensemble = False


model_dirs = [experiment_dir / f"run_{i}" for i in range(1,6)]

def get_preds(model_dir):
    metrics = {}

    
    logger = logging.getLogger()
    handlers = logger.handlers
    if len(handlers) > 0:
        logger.removeHandler(handlers[0])


    predictors = []
    if semi_supervised_ensemble:
        for i in range(3):
            archive = load_archive(model_dir / f'{single_model_prefix}_{i}' / 'model.tar.gz', cuda_device=0)
            predictor = Predictor.from_archive(archive, 'suggestion_mining')
            predictors.append(predictor)
    else:
        archive = load_archive(model_dir / 'model.tar.gz', cuda_device=0)
        predictor = Predictor.from_archive(archive, 'suggestion_mining')
        predictors.append(predictor)

    
    test, test_pred = [], []
    
    with test_data_path.open('rt', errors='ignore') as test_file:
        next(test_file)
        for l in test_file:
            data = predictor.load_line(l.rstrip())
            pred = predictor.predict_json(data)
            test.append(pred['gold_label'])
            test_pred.append(pred['label'])

    return np.array(test), np.array(test_pred)


test_gold = None
all_runs = []

for model_dir in model_dirs:
    test_gold, run_pred = get_preds(model_dir)
    all_runs.append(run_pred)
all_runs = np.array(all_runs)

ensemble_pred = [1 if x > 2 else 0 for x in np.sum(all_runs, 0)] # Majority voting on all runs

for pred, gold in zip(test_gold, ensemble_pred):
    if pred == gold:
        print(1)
    else:
        print(0)