# Semeval-2019 Task 3 Emocontext
### Masters Project Lund University
##### Nate Joselson and Rasmus Hallén

Requires Python 3 and Keras

Model creation: `python3 testBaseline.py -conf testBaseline.config`

Model Evaluation: `python3 test_evaluation.py -conf testBaseline.config`

## Model scores
### Baseline
True Positives per class :  [4250.  143.  162.  245.]\
False Positives per class :  [222.  84. 145. 258.]\
False Negatives per class :  [427. 141.  88.  53.]\
Class happy : Precision : 0.630, Recall : 0.504, F1 : 0.56\
Class sad : Precision : 0.528, Recall : 0.648, F1 : 0.582\
Class angry : Precision : 0.487, Recall : 0.822, F1 : 0.612\
Ignoring the Others class, Macro Precision : 0.5482, Macro Recall : 0.6579, **Macro F1 : 0.5981** \
Ignoring the Others class, Micro TP : 550, FP : 487, FN : 282\
Accuracy : 0.8713, Micro Precision : 0.5304, Micro Recall : 0.6611, Micro F1 : 0.5886

### With smiley separation
True Positives per class :  [4268.  152.  163.  245.]\
False Positives per class :  [219.  98. 149. 215.]\
False Negatives per class :  [409. 132.  87.  53.]\
Class happy : Precision : 0.608, Recall : 0.535, F1 : 0.569\
Class sad : Precision : 0.522, Recall : 0.652, F1 : 0.580\
Class angry : Precision : 0.533, Recall : 0.822, F1 : 0.646\
Ignoring the Others class, Macro Precision : 0.5543, Macro Recall : 0.6698, **Macro F1 : 0.6066** \
Ignoring the Others class, Micro TP : 560, FP : 462, FN : 272 \
Accuracy : 0.8764, Micro Precision : 0.5479, Micro Recall : 0.6731, Micro F1 : 0.6041

### 3 lstm layers each treating one turn
True Positives per class :  [4312.  185.  144.  230.]\
False Positives per class :  [231. 216.  58. 133.]\
False Negatives per class :  [365.  99. 106.  68.]\
Class happy : Precision : 0.461, Recall : 0.651, F1 : 0.540\
Class sad : Precision : 0.713, Recall : 0.576, F1 : 0.637\
Class angry : Precision : 0.634, Recall : 0.772, F1 : 0.696\
Ignoring the Others class, Macro Precision : 0.6026, Macro Recall : 0.6664, **Macro F1 : 0.6329** \
Ignoring the Others class, Micro TP : 559, FP : 407, FN : 273 \
Accuracy : 0.8842, Micro Precision : 0.5787, Micro Recall : 0.6719, Micro F1 : 0.6218

### Extra meta data layer, and hidden
True Positives per class :  [4129.  191.  181.  248.] \
False Positives per class :  [156. 224. 152. 228.] \
False Negatives per class :  [548.  93.  69.  50.] \
Class happy : Precision : 0.460, Recall : 0.673, F1 : 0.546\
Class sad : Precision : 0.544, Recall : 0.724, F1 : 0.621\
Class angry : Precision : 0.521, Recall : 0.832, F1 : 0.641\
Ignoring the Others class, Macro Precision : 0.5083, Macro Recall : 0.7429, **Macro F1 : 0.6036** \
Ignoring the Others class, Micro TP : 620, FP : 604, FN : 212 \
Accuracy : 0.8620, Micro Precision : 0.5065, Micro Recall : 0.7452, Micro F1 : 0.6031

### 3 node - Meta data layer + smaller hidden layer
True Positives per class :  [3746.  202.  194.  258.]\
False Positives per class :  [102. 260. 299. 448.]\
False Negatives per class :  [931.  82.  56.  40.]\
Class happy : Precision : 0.437, Recall : 0.711, F1 : 0.542\
Class sad : Precision : 0.394, Recall : 0.776, F1 : 0.522\
Class angry : Precision : 0.365, Recall : 0.866, F1 : 0.514\
Ignoring the Others class, Macro Precision : 0.3987, Macro Recall : 0.7843, **Macro F1 : 0.5287**\
Ignoring the Others class, Micro TP : 654, FP : 1007, FN : 178\
Accuracy : 0.7987, Micro Precision : 0.3937, Micro Recall : 0.7861, Micro F1 : 0.5247
