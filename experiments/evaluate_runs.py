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
val_data_path =  pathlib.Path(sys.argv[2])
test_data_path = pathlib.Path(sys.argv[3])

if len(sys.argv) == 5:
    semi_supervised_ensemble = True
    single_model_prefix = 'best_model'
else:
    semi_supervised_ensemble = False


model_dirs = [experiment_dir / f"run_{i}" for i in range(1,6)]


req_metrics_keys = [
    "validation_precision", 
    "validation_recall", 
    "validation_f1",
    "test_precision", 
    "test_recall", 
    "test_f1",
]

required_metrics = {k:[] for k in req_metrics_keys}


def get_metrics(model_dir):
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

    
    val, val_pred = [], []
    test, test_pred = [], []
    
    with val_data_path.open('rt', errors='ignore') as val_file:
        next(val_file)
        for l in val_file:
            data = predictors[0].load_line(l.rstrip())
            
            preds = []
            for predictor in predictors:
                pred = predictor.predict_json(data)
                preds.append(pred['label'])
            pred = max(set(preds), key=preds.count) # Majority voting for ensemble
            val.append(data['gold_label'])
            val_pred.append(pred)
    p, r, f1, _ = precision_recall_fscore_support(val, val_pred)
    metrics['validation_precision'], metrics['validation_recall'], metrics['validation_f1'] = p[1], r[1], f1[1]
    
    with test_data_path.open('rt', errors='ignore') as test_file:
        next(test_file)
        for l in test_file:
            data = predictor.load_line(l.rstrip())
            pred = predictor.predict_json(data)
            test.append(pred['gold_label'])
            test_pred.append(pred['label'])
    p, r, f1, _ = precision_recall_fscore_support(test, test_pred)
    metrics['test_precision'], metrics['test_recall'], metrics['test_f1'] = p[1], r[1], f1[1]

    return metrics


for model_dir in model_dirs:
    metrics = get_metrics(model_dir)
    for k in req_metrics_keys:
        required_metrics[k].append(metrics[k])


print("{:^45}{:^45} ".format("Val", "Test"))
print("{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}".format("P", "R", "F1", "P", "R", "F1"))


formated_metrics = {}


for k in req_metrics_keys:
    metric_samples = required_metrics[k]
    mean = np.mean(metric_samples)
    confidence = 0.95
    n = len(metric_samples)
    standard_error = stats.sem(metric_samples)
    h =  standard_error * stats.t.ppf((1. + confidence) / 2., n - 1)
    mean = mean * 100
    h = h * 100

    formated_metrics[k] = "{:.2f}±{:.2f}".format(mean, h)

print("{:^15}{:^15}{:^15}{:^15}{:^15}{:^15}".format(
    formated_metrics["validation_precision"],
    formated_metrics["validation_recall"],
    formated_metrics["validation_f1"],
    formated_metrics["test_precision"],
    formated_metrics["test_recall"],
    formated_metrics["test_f1"]
))
