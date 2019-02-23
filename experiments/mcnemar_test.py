
import sys
import numpy as np
from scipy import stats
import pathlib

# https://github.com/rtmdrr/testSignificanceNLP/blob/master/testSignificance.py
def calculateContingency(data_A, data_B):
    ABrr = 0
    ABrw = 0
    ABwr = 0
    ABww = 0
    for i in range(0, len(data_A)):
        if(data_A[i]==1 and data_B[i]==1):
            ABrr = ABrr+1
        if (data_A[i] == 1 and data_B[i] == 0):
            ABrw = ABrw + 1
        if (data_A[i] == 0 and data_B[i] == 1):
            ABwr = ABwr + 1
        else:
            ABww = ABww + 1
    return np.array([[ABrr, ABrw], [ABwr, ABww]])

def mcNemar(table):
    statistic = float(np.abs(table[0][1]-table[1][0]))**2/(table[1][0]+table[0][1])
    pval = 1-stats.chi2.cdf(statistic,1)
    return pval

subtask_dir = pathlib.Path(sys.argv[1])
model_dirs = [x for x in subtask_dir.iterdir() if x.is_dir()]

model_predictions = []

for model_dir in model_dirs:
    model_name = model_dir.name
    test_pred_file = model_dir / 'test_set_correct.txt'
    if test_pred_file.exists():
        with test_pred_file.open() as f:
            preds = [int(l.rstrip()) for l in f]
            model_predictions.append((model_name, preds))



for model_name, preds in model_predictions:
    p_values = []
    for other_model_name, other_preds in model_predictions:
        if other_model_name != model_name:
            table = calculateContingency(preds, other_preds)
            p = mcNemar(table)
            p_val_formatted = "{:^20.4}".format(p)
            #if p > 0.05:
                #print("Null hypothesis could be true in", model_name, other_model_name)
            print(p)
            print(table)
        else:
            p_val_formatted = "{:^20}".format('NA')
        p_values.append(p_val_formatted)

    print_stuff = ["{:^20}".format(model_name)]
    print_stuff.extend(p_values)
    print(''.join(print_stuff))
