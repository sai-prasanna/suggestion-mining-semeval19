## Organizer Baseline
```
                     Val                                         Test
       P              R             F1              P              R             F1
    72.84          81.68          77.01           68.86          78.16          73.21
```

## dan+glove

```sh
cd dan_glove
allennlp train --include-package=hinton dan_glove.jsonnet -s run_1
allennlp train --include-package=hinton dan_glove.jsonnet -s run_2 --overrides="{"numpy_seed": 453, "pytorch_seed": 12, "random_seed": 193 }"
allennlp train --include-package=hinton dan_glove.jsonnet -s run_3 --overrides="{"numpy_seed": 78, "pytorch_seed": 32, "random_seed":  54 }"
allennlp train --include-package=hinton dan_glove.jsonnet -s run_4 --overrides="{"numpy_seed": 893, "pytorch_seed": 933, "random_seed": 177 }"
allennlp train --include-package=hinton dan_glove.jsonnet -s run_5 --overrides="{"numpy_seed": 88, "pytorch_seed": 938, "random_seed":  1111 }"
python ../../predict_test_correct.py . ../../data/SubtaskB_EvaluationData_labeled.csv > test_set_correct.txt
python ../../evaluate_runs.py . ../../data/SubtaskB_Trial_Test_Labeled.csv ../../data/SubtaskB_EvaluationData_labeled.csv
```

```
                     Val                                         Test
       P              R             F1              P              R             F1
  82.00±4.25     52.97±9.25     64.01±5.75     73.32±3.50     46.09±7.21     56.35±4.71
```

## dan+bert

``` sh
cd dan_bert
allennlp train --include-package=hinton dan_bert.jsonnet -s run_1
allennlp train --include-package=hinton dan_bert.jsonnet -s run_2 --overrides="{"numpy_seed": 88, "pytorch_seed": 938, "random_seed":  1111 }"
allennlp train --include-package=hinton dan_bert.jsonnet -s run_3 --overrides="{"numpy_seed": 8238, "pytorch_seed": 43345, "random_seed":  834 }"
allennlp train --include-package=hinton dan_bert.jsonnet -s run_4 --overrides="{"numpy_seed": 944, "pytorch_seed": 1221, "random_seed":  6 }"
allennlp train --include-package=hinton dan_bert.jsonnet -s run_5 --overrides="{"numpy_seed": 1114, "pytorch_seed": 261, "random_seed": 3336 }"
python ../../predict_test_correct.py . ../../data/SubtaskB_EvaluationData_labeled.csv > test_set_correct.txt
python ../../evaluate_runs.py . ../../data/SubtaskB_Trial_Test_Labeled.csv ../../data/SubtaskB_EvaluationData_labeled.csv
```
```
                     Val                                         Test
       P              R             F1              P              R             F1
  89.75±2.79     65.74±8.71     75.65±5.10     78.90±4.03     64.20±8.77     70.49±4.09
```

## dan+bert-upsampling

```
cd dan_bert_no_upsampling
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_1
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_2 --overrides="{"numpy_seed": 88, "pytorch_seed": 938, "random_seed":  1111 }"
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_3 --overrides="{"numpy_seed": 8238, "pytorch_seed": 43345, "random_seed":  834 }"
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_4 --overrides="{"numpy_seed": 944, "pytorch_seed": 1221, "random_seed":  6 }"
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_5 --overrides="{"numpy_seed": 1114, "pytorch_seed": 261, "random_seed": 3336 }"
python ../../predict_test_correct.py . ../../data/SubtaskB_EvaluationData_labeled.csv > test_set_correct.txt
python ../../evaluate_runs.py . ../../data/SubtaskB_Trial_Test_Labeled.csv ../../data/SubtaskB_EvaluationData_labeled.csv
```
```
                     Val                                         Test
       P              R             F1              P              R             F1
  94.26±1.87     31.73±5.73     47.31±6.27     87.98±3.41     31.09±7.17     45.62±7.47
```

## cnn+bert

```
cd cnn_bert
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_1
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_2 --overrides="{"numpy_seed": 2124, "pytorch_seed": 1621, "random_seed": 882 }"
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_3 --overrides="{"numpy_seed": 1324, "pytorch_seed": 31, "random_seed": 9277 }"
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_4 --overrides="{"numpy_seed": 777, "pytorch_seed": 666, "random_seed": 15 }"
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_5 --overrides="{"numpy_seed": 7277, "pytorch_seed": 16, "random_seed": 125 }"
python ../../predict_test_correct.py . ../../data/SubtaskB_EvaluationData_labeled.csv > test_set_correct.txt
python ../../evaluate_runs.py . ../../data/SubtaskB_Trial_Test_Labeled.csv ../../data/SubtaskB_EvaluationData_labeled.csv
```
```
                     Val                                         Test
       P              R             F1              P              R             F1
  93.77±1.34     51.88±6.88     66.65±5.68     90.17±2.45     50.34±8.71     64.31±6.72
```


## cnn+bert-upsampling

```
cdd cnn_bert_no_upsampling
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_1
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_2 --overrides="{"numpy_seed": 2124, "pytorch_seed": 1621, "random_seed": 882 }"
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_3 --overrides="{"numpy_seed": 1324, "pytorch_seed": 31, "random_seed": 9277 }"
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_4 --overrides="{"numpy_seed": 777, "pytorch_seed": 666, "random_seed": 15 }"
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_5 --overrides="{"numpy_seed": 7277, "pytorch_seed": 16, "random_seed": 125 }"
python ../../predict_test_correct.py . ../../data/SubtaskB_EvaluationData_labeled.csv > test_set_correct.txt
python ../../evaluate_runs.py . ../../data/SubtaskB_Trial_Test_Labeled.csv ../../data/SubtaskB_EvaluationData_labeled.csv
```
```
                     Val                                         Test
       P              R             F1              P              R             F1
  93.94±1.36     45.99±7.59     61.53±6.73     89.75±4.41     44.08±9.38     58.66±7.79
```

## cnn + bert + tritrain on unlabelled test set data (Evaluation Data)

```sh
cd cnn_bert_tritrain
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_1 ../../data/SubtaskB_EvaluationData.csv 1337 2>&1 | tee run_1.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_2 ../../data/SubtaskB_EvaluationData.csv 1331 2>&1 | tee run_2.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_3 ../../data/SubtaskB_EvaluationData.csv 141 2>&1 | tee run_3.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_4 ../../data/SubtaskB_EvaluationData.csv 17 2>&1 | tee run_4.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_5 ../../data/SubtaskB_EvaluationData.csv 554 2>&1 | tee run_5.log
python ../../predict_test_correct.py . ../../data/SubtaskB_EvaluationData_labeled.csv semi > test_set_correct.txt
python ../../evaluate_runs.py . ../../data/SubtaskB_Trial_Test_Labeled.csv ../../data/SubtaskB_EvaluationData_labeled.csv semi
```
```
                     Val                                         Test
       P              R             F1              P              R             F1
  91.91±2.06     88.32±2.05     90.05±0.76     81.26±1.63     83.16±1.40     82.19±1.03
```

## cnn + bert + tritrain on yelp reviews
```
cd cnn_bert_tri_train_yelp
python ../../tritrain.py cnn_bert_tri_train_yelp.jsonnet ./run_1 ../../data/yelp.txt 1337 2>&1 | tee run_1.log
python ../../tritrain.py cnn_bert_tri_train_yelp.jsonnet ./run_2 ../../data/yelp.txt 1331 2>&1 | tee run_2.log
python ../../tritrain.py cnn_bert_tri_train_yelp.jsonnet ./run_3 ../../data/yelp.txt 141 2>&1 | tee run_3.log
python ../../tritrain.py cnn_bert_tri_train_yelp.jsonnet ./run_4 ../../data/yelp.txt 17 2>&1 | tee run_4.log
python ../../tritrain.py cnn_bert_tri_train_yelp.jsonnet ./run_5 ../../data/yelp.txt 554 2>&1 | tee run_5.log
python ../../predict_test_correct.py . ../../data/SubtaskB_EvaluationData_labeled.csv semi > test_set_correct.txt
python ../../evaluate_runs.py . ../../data/SubtaskB_Trial_Test_Labeled.csv ../../data/SubtaskB_EvaluationData_labeled.csv semi
```
```
                     Val                                         Test
       P              R             F1              P              R             F1
  88.09±0.62     87.13±0.38     87.61±0.42     78.01±5.42     86.67±3.96     81.98±2.05
```

## Mcnemar 

```
python ../mcnemar_test.py .
```

