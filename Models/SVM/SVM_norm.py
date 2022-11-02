import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score

# get the current directory
cur_dir = os.getcwd()

# Import the dataset
df = pd.read_csv(os.path.join(cur_dir, '../../Datasets/normalized_banknote.csv'), header=None)
print(df.shape)

# split into x, y
X, y = df.drop(4, axis=1), df[4]

# Train/test split for model development
X_train, X_test, y_train, y_test = train_test_split(X, y)

# Define the model parameters
kernels = ['linear', 'poly', 'sigmoid', 'rbf']

# Open the output file 
f = open(os.path.join(cur_dir, '../../Results/normailzed_banknote.txt'), 'w')

for kernel in kernels:
    svc_classifier = SVC(kernel=kernel)
    svc_classifier.fit(X_train,y_train)

    # Predict t
    svc_classifier_prediction = svc_classifier.predict(X_test)
    
    # Output the Results
    f.write(kernel + ':\n')
    f.write(str(confusion_matrix(y_test,svc_classifier_prediction)) + "\n")
    f.write(str(classification_report(y_test, svc_classifier_prediction)))
    f.write("Accuracy:" + str(accuracy_score(y_test, svc_classifier_prediction)*100) + "%\n\n")
    
f.close()
    
