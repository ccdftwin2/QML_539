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
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Define the model parameters
kernels = ['linear', 'poly', 'sigmoid', 'rbf']

# Open the output file 
f = open(os.path.join(cur_dir, '../../Results/normalized_banknote.txt'), 'w')

from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay

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
    
    cm = confusion_matrix(y_test, svc_classifier_prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[-1,1])
    disp.text_.set_size(20)
    disp.plot()
    disp.im_.set_clim(0, 230)
    plt.savefig("../../Results/Confusion_Matrix_" + kernel + "_banknote.png")
    
f.close()
    
