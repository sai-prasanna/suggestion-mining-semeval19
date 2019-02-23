import os
import os.path
import sys
import csv
import io
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from hinton.predictors import SuggestionMiningPredictor
from hinton.data.dataset_readers import SuggestionMiningReader
from hinton.model import text_classifier
import random
import subprocess
import torch
import logging
import json
import _jsonnet
import pathlib
from collections import defaultdict
from sklearn.metrics import precision_recall_fscore_support

MODEL_CONFIG = sys.argv[1]
MODELS_DIR = pathlib.Path(sys.argv[2])
SEMI_SUPERVISED_UNLABELLED_FILE = sys.argv[3]
SEED = sys.argv[4]
VAL_DATA_PATH = pathlib.Path(json.loads(_jsonnet.evaluate_file(MODEL_CONFIG))['validation_data_path'])

random.seed(SEED)



MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Generate the seeds for all the models in this tri training run.
# We opt to use same seed to control for intialization, so only tri-training dataset can affect the performance in a given run
PYTORCH_SEED = random.randint(0, 10000)
NUMPY_SEED = random.randint(0, 10000)
RANDOM_SEED = random.randint(0, 10000)


logger = logging.getLogger(__file__)
logger.info('START')

def load_data(path):
    with open(path, 'rt', errors='ignore') as sem_file:
        reader = csv.reader(sem_file, delimiter=',')
        next(reader, None)  # skip the headers

        sentences = []
        labels = []
        for record in reader:
            _, sentence, label = record[0], record[1], record[2]
            sentences.append(sentence)
            if label == 'X':
                label = None
            else:
                label = int(label)
            labels.append(label)
    return sentences, labels

def csv2string(data):
    si = io.StringIO()
    cw = csv.writer(si, delimiter=',')
    cw.writerow(data)
    return si.getvalue().strip('\r\n')

def train_model(model_id, tri_train_epoch, dataset):

    model_path = f"{MODELS_DIR}/tri_train_epoch_{tri_train_epoch}_model_{model_id}"
    training_data_path = f"{MODELS_DIR}/tri_train_epoch_{tri_train_epoch}_model_{model_id}.csv"
    
    with open(training_data_path, 'w') as f:
        for sent, label in dataset:
            f.write(csv2string(('dummy_id', sent, int(label))) + '\n')
    
    # Train New

    train_command = [
        "allennlp",
        "train",
        MODEL_CONFIG,
        "-s",
        model_path,
        f"--overrides={{ 'train_data_path': '{training_data_path}', 'pytorch_seed': {PYTORCH_SEED}, 'random_seed': {RANDOM_SEED}, 'numpy_seed': {NUMPY_SEED} }}",
        f"--include-package=hinton"
    ]
    exit_code = -1
    while exit_code != 0:
        subprocess.call(f"rm -rf {model_path}".split())
        exit_code = subprocess.call(train_command, stdout=subprocess.PIPE)

        

def load_predictor(model_id, tri_train_epoch):
    model_path = f"{MODELS_DIR}/tri_train_epoch_{tri_train_epoch}_model_{model_id}"
    archive_path = f'{model_path}/model.tar.gz'
    predictor = SuggestionMiningPredictor.from_archive(load_archive(archive_path, cuda_device=0), 'suggestion_mining')
    return predictor

def predict(predictor, sentence):
    return predictor.predict_json({'sentence': sentence})['label']

def bootstrap_sample(dataset, ratio=1.0):
    sample = []
    n_sample = round(len(dataset) * ratio)
    while len(sample) < n_sample:
        index = random.randrange(len(dataset))
        sample.append(dataset[index])
    return list(set(sample))

def load_unlabelled_data(data_file):
    dataset = []
    if '.csv' in data_file:
        dataset, _ = load_data(data_file) # For test set data
    elif '.txt' in data_file:
        with open(data_file) as f:
            for l in f:
                dataset.append(l.rstrip())
    else:
        raise Exception('Unable to load unlabelled data for semisupervised training')
    return dataset

TRAINING_DATASET = list(zip(*load_data('../../data/V1.4_Training.csv')))
UNLABELLED_DATASET = load_unlabelled_data(SEMI_SUPERVISED_UNLABELLED_FILE)

old_datasets = [None, None, None]
current_datasets = [bootstrap_sample(TRAINING_DATASET) for i in range(3)]
current_predictions = []

tri_train_epoch = 0
best_val_f1_score = 0.0
PATIENCE = 3
current_patience = PATIENCE

while current_patience > 0: 

    for model_id in range(3):
        train_model(model_id, tri_train_epoch, current_datasets[model_id])
    
    logger.info(f'----------------------------Starting Tri-Training epoch {tri_train_epoch} ------------------------------------------------')

    semi_supervised_label_preds = defaultdict(list)
    val_gold_labels = []
    val_ensemble_predictions = defaultdict(list)

    for model_id in range(3):
        predictor = load_predictor(model_id, tri_train_epoch)
        for x in UNLABELLED_DATASET:
            prediction = predict(predictor, x)
            semi_supervised_label_preds[model_id].append((x, prediction))

        with VAL_DATA_PATH.open('rt', errors='ignore') as val_file:
            next(val_file)
            for l in val_file:
                data = predictor.load_line(l.rstrip())
                pred = predictor.predict_json(data)
                if model_id == 0:
                    val_gold_labels.append(data['gold_label']) # Only once as val labels don't change
                val_ensemble_predictions[model_id].append(pred['label']) 

        del predictor
        torch.cuda.empty_cache()

    for model_id in range(3):

        logger.info(f'------------ Starting Model {model_id} dataset search ---------------')

        old_datasets[model_id] = current_datasets[model_id]
        current_datasets[model_id] = []
        current_datasets[model_id].extend(TRAINING_DATASET) #Add entire labelled data to semi_supervised

        other_model_ids = list(range(3))
        other_model_ids.remove(model_id)

        for semi_supervised_data_j, semi_supervised_data_k in zip(semi_supervised_label_preds[other_model_ids[0]],
                                                                  semi_supervised_label_preds[other_model_ids[1]]):

            _, y1 = semi_supervised_data_j
            _, y2 = semi_supervised_data_k
            if y1 == y2: # labels of other models agree
                # Add new data point from semi supervised labels as other two models agree
                current_datasets[model_id].append(semi_supervised_data_j) 
       
        current = set(current_datasets[model_id])
        old = set(old_datasets[model_id])
        new_labels =  (current - old) | (old - current)
        logger.info(f'---------------------- Augmenting with {len(current) - len(TRAINING_DATASET)} data points of which {len(new_labels)} are new or changed -------------------------')

    val_preds = []
    for ensemble_pred in zip(val_ensemble_predictions[0], val_ensemble_predictions[1], val_ensemble_predictions[2]):
        pred = max(set(ensemble_pred), key=ensemble_pred.count) # Majority voting for ensemble
        val_preds.append(pred)

    _, _, val_f1, _ = precision_recall_fscore_support(val_gold_labels, val_preds)
    val_f1 = val_f1[1] # Only suggestions label f1

    logger.info(f"Ensemble validation f1-score {val_f1}")
    if (val_f1 - best_val_f1_score) < 1e-3:
        current_patience -= 1
    else:
        current_patience = PATIENCE #Reset patience
        best_val_f1_score = val_f1
        
    tri_train_epoch += 1

for model_id in range(3):
    copy_best_command = f"cp -R {MODELS_DIR}/tri_train_epoch_{tri_train_epoch - 1 - PATIENCE}_model_{model_id} {MODELS_DIR}/best_model_{model_id}"
    subprocess.call(copy_best_command.split())