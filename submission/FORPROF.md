# Given

dataset/
    includes given dataset .csv files

# Deliverables

main.ipynb - main Jupyter Notebook
    Jupyter  notebook  of  ANN  code  in  Python  for  Networks  A  and  B  with  self-
    documenting comments. Include another one for generating the balanced data 
    using SMOTE.
    Includes:
    i) how the dataset was partitioned and details on how you dealt with the 
    highly imbalanced dataset 
    ii) how the number of nodes in the hidden layers was selected, how you 
    tuned  the  hyperparameters  (learning  rate  parameter,  momentum 
    constant alpha) for Network A 
    iii) Separate plots of training and validation errors vs epoch number for 
    Networks A and B.
    iv) Timing Comparisons for Networks A and B 
    v) Confusion  matrix,  Accuracy,  Precision,  Recall,  F1  scores,  Matthews 
    Correlation Coefficient values and their interpretation (one each for 
    Networks A and B). You may include other metrics from our Classifier 
    Evaluation Lecture if you wish. 
    vi) Conclusion

export/training_set.csv - using SMOTE
export/training_labels.csv - using SMOTE
export/validation_set.csv - using SMOTE
export/validation_labels.csv - using SMOTE

predictions/predictions_for_test_tanh.csv - using best model for Network A
predictions/predictions_for_test_leakyrelu.csv - using best model for Network B