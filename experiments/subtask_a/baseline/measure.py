from sklearn.metrics import precision_recall_fscore_support 
                                                                                                                                                                                     
                                                                                                                                                                      
def get_labels(path): 
    labels = [] 
    with open(path, 'rt', errors='ignore') as f: 
        next(f) 
        for l in f: 
            labels.append(int(l.rstrip()[-1])) 
    return labels 
                                                                                                                                                                                    

val_gold = get_labels('../../data/SubtaskA_Trial_Test_Labeled.csv')                                                                                                                  
val_preds = get_labels('validation_preds.csv')                                                                                                                                       
print('Validation')
print(precision_recall_fscore_support(val_gold, val_preds))  

test_gold = get_labels('../../data/SubtaskA_EvaluationData_labeled.csv')
test_preds = get_labels('test_preds.csv')
print('Test')
print(precision_recall_fscore_support(test_gold, test_preds))

with open('test_set_correct.txt', 'w') as f: 
    for p, g in zip(test_preds, test_gold): 
        if p == g: 
            f.write('1\n') 
        else: 
            f.write('0\n') 
