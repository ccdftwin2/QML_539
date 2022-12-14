import numpy as np
import pandas as pd
import os
import time
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pennylane as qml
from pennylane.templates import AngleEmbedding, StronglyEntanglingLayers
from pennylane.operation import Tensor
import matplotlib.pyplot as plt

# get the current directory
cur_dir = os.getcwd()

# Import the dataset
df = pd.read_csv(os.path.join(cur_dir, '../../Datasets/normalized_banknote.csv'), header=None)

# split into x, y
X, y = df.drop(4, axis=1), df[4]

# Need to scale the data to normal distribution for embeddings to work
scaler = StandardScaler().fit(X)
X_scaled = scaler.transform(X)
print("Mean: ", X_scaled.mean(), "Standard Deviation: ", X_scaled.std())
y_scaled = 2 * (y - 0.5)
print("Unique labels:", np.unique(y_scaled))

# Train/test split for model development
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, random_state=42, test_size=0.3)

# Embed the data. First define the number of qubits
n_qubits = len(X_scaled[0])

# Set up the projector
dev_kernel = qml.device("default.qubit", wires=n_qubits)

projector = np.zeros((2**n_qubits, 2**n_qubits))
projector[0, 0] = 1

# Define the kernel
@qml.qnode(dev_kernel)
def kernel(x1, x2):
    """The quantum kernel."""
    AngleEmbedding(x1, wires=range(n_qubits))
    qml.adjoint(AngleEmbedding)(x2, wires=range(n_qubits))
    return qml.expval(qml.Hermitian(projector, wires=range(n_qubits)))

# Needed for SVC function
def kernel_matrix(A, B):
    return np.array([[kernel(a, b) for b in B] for a in A])

# Start time
start = time.time()

# Define the svm
svm = SVC(kernel=kernel_matrix, verbose=True).fit(X_train, y_train)

#train time
train_time = time.time() - start

# Predict the test set
prediction = svm.predict(X_test)

# predict time
predict_time = time.time() - train_time - start

# Write the results to the test file
f = open(os.path.join(cur_dir, '../../Results/qsvm_banknote.txt'), 'w')
f.write(str(confusion_matrix(y_test,prediction)) + "\n")
f.write(str(classification_report(y_test, prediction)))
f.write("Accuracy:" + str(accuracy_score(y_test, prediction)*100) + "%\n\n")
f.write("Training Time: " + str(train_time) + "\n")
f.write("Prediction Time: " + str(predict_time) + "\n")
cm = confusion_matrix(y_test,prediction)
from sklearn.metrics import confusion_matrix,ConfusionMatrixDisplay
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=[-1,1])
disp.plot()
disp.im_.set_clim(0, 230)
plt.savefig("../../Results/Confusion_Matrix_" + "lin_qsvm" + "_banknote.png")
f.close()

