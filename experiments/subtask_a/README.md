##Organizer Baseline

                     Val                                         Test
       P              R             F1              P              R             F1
    0.5872          0.9324        0.7206         0.1569         0.9195         0.2680


##dan+glove

export PYTHONPATH=../../../../src
allennlp train --include-package=hinton dan_glove.jsonnet -s run_1
allennlp train --include-package=hinton dan_glove.jsonnet -s run_2 --overrides="{"numpy_seed": 453, "pytorch_seed": 12, "random_seed": 193 }"
allennlp train --include-package=hinton dan_glove.jsonnet -s run_3 --overrides="{"numpy_seed": 78, "pytorch_seed": 32, "random_seed":  54 }"
allennlp train --include-package=hinton dan_glove.jsonnet -s run_4 --overrides="{"numpy_seed": 893, "pytorch_seed": 933, "random_seed": 177 }"
allennlp train --include-package=hinton dan_glove.jsonnet -s run_5 --overrides="{"numpy_seed": 88, "pytorch_seed": 938, "random_seed":  1111 }"


python ../../evaluate_runs.py . ../../data/SubtaskA_Trial_Test_Labeled.csv ../../data/SubtaskA_EvaluationData_labeled.csv
                     Val                                         Test
       P              R             F1              P              R             F1
  68.51±2.43     87.30±5.00     76.69±1.06     25.40±3.56     84.60±9.87     38.84±3.10


##dan+bert

allennlp train --include-package=hinton dan_bert.jsonnet -s run_1
allennlp train --include-package=hinton dan_bert.jsonnet -s run_2 --overrides="{"numpy_seed": 88, "pytorch_seed": 938, "random_seed":  1111 }"
allennlp train --include-package=hinton dan_bert.jsonnet -s run_3 --overrides="{"numpy_seed": 8238, "pytorch_seed": 43345, "random_seed":  834 }"
allennlp train --include-package=hinton dan_bert.jsonnet -s run_4 --overrides="{"numpy_seed": 944, "pytorch_seed": 1221, "random_seed":  6 }"
allennlp train --include-package=hinton dan_bert.jsonnet -s run_5 --overrides="{"numpy_seed": 1114, "pytorch_seed": 261, "random_seed": 3336 }"


python ../../evaluate_runs.py . ../../data/SubtaskA_Trial_Test_Labeled.csv ../../data/SubtaskA_EvaluationData_labeled.csv

                     Val                                         Test
       P              R             F1              P              R             F1
  76.06±1.31     90.27±1.71     82.55±0.50     45.80±4.49     90.80±1.75     60.82±3.99


##dan+bert-upsampling

allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_1
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_2 --overrides="{"numpy_seed": 88, "pytorch_seed": 938, "random_seed":  1111 }"
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_3 --overrides="{"numpy_seed": 8238, "pytorch_seed": 43345, "random_seed":  834 }"
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_4 --overrides="{"numpy_seed": 944, "pytorch_seed": 1221, "random_seed":  6 }"
allennlp train --include-package=hinton dan_bert_no_upsampling.jsonnet -s run_5 --overrides="{"numpy_seed": 1114, "pytorch_seed": 261, "random_seed": 3336 }"


python ../../evaluate_runs.py . ../../data/SubtaskA_Trial_Test_Labeled.csv ../../data/SubtaskA_EvaluationData_labeled.csv


                     Val                                         Test
       P              R             F1              P              R             F1
  79.04±2.67     83.38±2.73     81.11±0.68     55.06±6.36     83.68±2.75     66.28±4.28

##cnn+bert


allennlp train --include-package=hinton cnn_bert.jsonnet -s run_1
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_2 --overrides="{"numpy_seed": 2124, "pytorch_seed": 1621, "random_seed": 882 }"
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_3 --overrides="{"numpy_seed": 1324, "pytorch_seed": 31, "random_seed": 9277 }"
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_4 --overrides="{"numpy_seed": 777, "pytorch_seed": 666, "random_seed": 15 }"
allennlp train --include-package=hinton cnn_bert.jsonnet -s run_5 --overrides="{"numpy_seed": 7277, "pytorch_seed": 16, "random_seed": 125 }"


python ../../evaluate_runs.py . ../../data/SubtaskA_Trial_Test_Labeled.csv ../../data/SubtaskA_EvaluationData_labeled.csv

                     Val                                         Test
       P              R             F1              P              R             F1
  80.34±4.21     89.93±4.23     84.76±0.52     50.34±6.70     91.72±2.55     64.81±4.86



##cnn+bert-upsampling


allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_1
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_2 --overrides="{"numpy_seed": 2124, "pytorch_seed": 1621, "random_seed": 882 }"
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_3 --overrides="{"numpy_seed": 1324, "pytorch_seed": 31, "random_seed": 9277 }"
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_4 --overrides="{"numpy_seed": 777, "pytorch_seed": 666, "random_seed": 15 }"
allennlp train --include-package=hinton cnn_bert_no_upsampling.jsonnet -s run_5 --overrides="{"numpy_seed": 7277, "pytorch_seed": 16, "random_seed": 125 }"



python ../../evaluate_runs.py . ../../data/SubtaskA_Trial_Test_Labeled.csv ../../data/SubtaskA_EvaluationData_labeled.csv

                     Val                                         Test
       P              R             F1              P              R             F1
  83.22±3.01     84.73±3.86     83.90±0.70     58.98±5.41     88.05±1.63     70.58±4.24


## cnn + bert + tritrain on unlabelled test set data 

python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_1 ../../data/SubtaskA_EvaluationData.csv 1337 2>&1 | tee run_1.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_2 ../../data/SubtaskA_EvaluationData.csv 1331 2>&1 | tee run_2.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_3 ../../data/SubtaskA_EvaluationData.csv 141 2>&1 | tee run_3.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_4 ../../data/SubtaskA_EvaluationData.csv 17 2>&1 | tee run_4.log
python ../../tritrain.py cnn_bert_tri_train.jsonnet ./run_5 ../../data/SubtaskA_EvaluationData.csv 554 2>&1 | tee run_5.log



python ../../evaluate_runs.py . ../../data/SubtaskA_Trial_Test_Labeled.csv ../../data/SubtaskA_EvaluationData_labeled.csv semi

                     Val                                         Test
       P              R             F1              P              R             F1
  83.06±1.96     89.19±1.88     86.00±0.35     52.89±2.69     90.80±2.02     66.81±1.90



## Mcnemar 

Run this in all except tri train task folders and baseline
python ../../predict_test_correct.py . ../../data/SubtaskA_EvaluationData_labeled.csv > test_set_correct.txt

Run this in tri train task folder
python ../../predict_test_correct.py . ../../data/SubtaskA_EvaluationData_labeled.csv > test_set_correct.txt semi



python mcnemar_test.py subtask_a

     dan_glove               NA                 0.0                 0.0                 0.0                 0.0                 0.0                 0.0         
dan_bert_no_upsampling        0.0                  NA               0.09896              0.3778            2.061e-05              0.0                0.1088       
      cnn_bert              0.0               0.09896                NA                0.1083             0.04662               0.0              8.236e-06      
 cnn_bert_tri_train         0.0                0.3778              0.1083                NA               0.003609              0.0              0.0004184      
      dan_bert              0.0              2.061e-05            0.04662             0.003609               NA                 0.0               2.26e-07      
      baseline              0.0                 0.0                 0.0                 0.0                 0.0                  NA                 0.0         
cnn_bert_no_upsampling        0.0                0.1088            8.236e-06           0.0004184            2.26e-07              0.0                  NA 