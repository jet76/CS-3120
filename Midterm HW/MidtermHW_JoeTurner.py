import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression


def plot_confusion_matrix(cnf_matrix, name):
    plt.clf()
    cm = pd.DataFrame(cnf_matrix, index=np.unique(
        y), columns=np.unique(y))
    cm.index.name = "Actual"
    cm.columns.name = "Predicted"
    sns.heatmap(cm, annot=True, cmap="YlGnBu", fmt='g')
    plt.title(name + ' Confusion Matrix')
    plt.tight_layout()
    plt.show()
    # plt.savefig(name + '.png', dpi=150)


iris = pd.read_csv('iris.csv')

# optional output
# print(iris.head())
# sns.pairplot(data=iris, hue='variety', palette='Set2')
# plt.show()
# plt.savefig('pairplot.png', dpi=300)

iris = np.array(iris)

x = iris[:, :-1]
# print(x)
y = iris[:, -1]
# print(y)
le = preprocessing.LabelEncoder()
labels = le.fit_transform(y)
# print(le.classes_)
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.30, random_state=0)

# Support Vector Machine
Gamma = 0.001
C = 1
model = SVC(kernel='linear', C=C, gamma=Gamma)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
svm_cnf_matrix = confusion_matrix(y_test, y_pred)
print('\nSupport Vector Machine')
print(classification_report(y_test, y_pred))
print(svm_cnf_matrix)
# plot_confusion_matrix(svm_cnf_matrix, 'SVM')

# K Nearest Neighbors
x_knn_test, x_valid, y_knn_test, y_valid = train_test_split(
    x_test, y_test, test_size=0.33, random_state=0)
K = [3, 5, 7]
L = [1, 2]
best_k = best_l = best_acc = 0
for l in L:
    for k in K:
        model = KNeighborsClassifier(n_neighbors=k, p=l)
        model.fit(x_train, y_train)
        acc = metrics.accuracy_score(y_valid, model.predict(
            x_valid))
        # print("L" + str(l) + ", " + "k=" + str(k) + ", Accuracy=" + str(acc))
        if acc > best_acc:
            best_acc = acc
            best_k = k
            best_l = l
# print("Best: L" + str(best_l) + ", k=" + str(best_k) + ", Accuracy=" + str(best_acc))
model = KNeighborsClassifier(n_neighbors=best_k, p=best_l)
model.fit(x_train, y_train)
y_pred = model.predict(x_knn_test)
knn_cnf_matrix = metrics.confusion_matrix(y_knn_test, y_pred)
print("\nK Nearest Neighbors (L" + str(best_l) + ", k=" + str(best_k) + ")")
print(classification_report(y_knn_test, model.predict(
    x_knn_test), target_names=le.classes_))
print(knn_cnf_matrix)
# plot_confusion_matrix(knn_cnf_matrix, 'KNN')

# Logistic Regression
logreg = LogisticRegression()
logreg.fit(x_train, y_train)
y_pred = logreg.predict(x_test)
lr_cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print("\nLogistic Regression")
print(classification_report(y_test, y_pred))
print(lr_cnf_matrix)
# plot_confusion_matrix(lr_cnf_matrix, 'LR')
